import tqdm
from ..model_arc.modeling_gpt2 import GPT2PreTrainedModel, GPT2LMHeadModel, GPT2Block
from transformers import GenerationMixin
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
import torch
from torch import nn, einsum
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers.utils import ModelOutput
from dataclasses import dataclass
from einops import rearrange, reduce, repeat
import os
import math
from safetensors.torch import load_file
from vector_quantize_pytorch import SimVQ

from torch.nn import CrossEntropyLoss, MSELoss

import torch
import numpy as np

class Concept_Module_VQ_GPT(GPT2PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        config,
        hlm_config=None,
        type='hlm'
    ):
        super().__init__(config)
        self.config = config
        self.chunk_size = self.config.chunk_size
        self.type=type

        self.transformer = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(self.config.special_layers)])

        self.loss_fct = CrossEntropyLoss()
        self.embed_dim = self.config.hidden_size

        self.encoder_mask = torch.tril(torch.ones(self.config.n_positions, self.config.n_positions))
        self.encoder_mask = self.encoder_mask.masked_fill(self.encoder_mask == 0, float('-inf')) -1

        self.ctx_lm_head = nn.ModuleList([nn.Linear(int(self.embed_dim), self.config.vq_config['codebook_size']) for  i in range(self.config.n_head)])


    def forward(
        self,
        hlm_hidden_states: Optional[Tuple[torch.FloatTensor]],
        combined_vq_outputs: Optional[Tuple[torch.FloatTensor]],
        combined_vq_indices: Optional[Tuple[torch.IntTensor]],
        epoch = None, 
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        topk_ctx_token_ids: Optional[Tuple[torch.Tensor]] = None,
        ranked_distance: Optional[Tuple[torch.Tensor]] = None,
        pred_indices: Optional[Tuple[torch.Tensor]] = None,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:

        for layers in self.transformer:
            
            outputs = layers(hlm_hidden_states, attention_mask=self.encoder_mask[:hlm_hidden_states.shape[1], :hlm_hidden_states.shape[1]].to(hlm_hidden_states.device), output_attentions=True)
            hlm_hidden_states = outputs[0]
        
        hidden_states = hlm_hidden_states

        pred_hlm_hs_split = hidden_states
        
        lm_loss = []
        vq_code = []
        total_lm_logits = []
        for i in range(self.config.n_head):
            
            lm_logits = self.ctx_lm_head[i](pred_hlm_hs_split) 
            shift_lmlogits = lm_logits[..., :-1, :].contiguous()

            vq_code.append(lm_logits.argmax(-1))
            total_lm_logits.append(lm_logits)    

        combined_vq_code = torch.stack(vq_code, dim=2)
        # combined_vq_code
        total_lm_logits = torch.stack(total_lm_logits, dim=3)
        return combined_vq_code, torch.tensor([0.0]), torch.tensor([0.0]), None, None, total_lm_logits  # hidden_states, present, (attentions, cross_attentions)



def default(val, d):
    return val if exists(val) else d


def exists(val):
    return val is not None


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `torch.FloatTensor` tuples of length `config.n_layers`, with each tuple containing the cached key,
            value states of the self-attention and the cross-attention layers if model is used in encoder-decoder
            setting. Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    hlm_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    codebook_rec_loss: Optional[torch.FloatTensor]=None
    pred_rec_loss: Optional[torch.FloatTensor]=None
    VQ_MSE_loss: Optional[torch.FloatTensor] = None
    VQ_CE_loss: Optional[torch.FloatTensor] = None
    acc: Optional[torch.FloatTensor] = None
    tf_pred_acc: Optional[torch.FloatTensor]=None
    ar_pred_acc: Optional[torch.FloatTensor]=None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None



class ConceptLM_GPT(GPT2PreTrainedModel, GenerationMixin):

    def __init__(self, config, hlm_config=None, attn_resampling=True):
        super().__init__(config)
  

        self.base_model_config = config

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        transformer_kwargs = dict(
            dim = config.n_embd,
            heads = config.n_head,
            dim_head = int(config.n_embd / config.n_head)
        )
        self.mse_loss_fct = MSELoss()
        self.chunk_size = self.config.chunk_size
        self.hlm_type = self.config.hlm_type

        self.model_parallel = False
        self.device_map = None

        self.max_positions = config.n_positions

        self.cosine_loss = nn.CosineEmbeddingLoss()

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.ce_loss_fct = nn.CrossEntropyLoss()

        self.encoder = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.encoder_layers)])

        if 'baseline' in self.config.training_type:
            self.hlm = None
        elif 'vq_gpt2_ce' in self.hlm_type:
            self.vqs = nn.ModuleList([
                SimVQ(
                    dim=int(config.n_embd / config.n_head),
                    codebook_size=self.config.vq_config['codebook_size'],
                    rotation_trick=True,
                ) for _ in range(config.n_head)
            ])    
        
            self.hlm = Concept_Module_VQ_GPT(
                config, 
            )

        self.decoder = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.decoder_layers)])

        self.init_masks()

        self.post_init()


    def init_masks(self):

        self.encoder_mask = torch.tril(torch.ones(self.max_positions, self.max_positions)).to(dtype=torch.bfloat16)
        self.encoder_mask = self.encoder_mask.masked_fill(self.encoder_mask == 0, float('-inf')) -1

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head.to(self.transformer.last_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head.to('cpu')
        self.model_parallel = False
        torch.cuda.empty_cache()

    def forward(
        self,
        epoch = None,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if labels is None:
            labels = input_ids
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.encoder) + [None] * len(self.decoder))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # prepare embeddings
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        encoder_hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        # encoder forward
        for encoder_layers in self.encoder:
            if self.config._attn_implementation == 'flash_attention_2':
                outputs = encoder_layers(encoder_hidden_states)
            else:
                outputs = encoder_layers(encoder_hidden_states, attention_mask=self.encoder_mask[:hidden_states.shape[1], :hidden_states.shape[1]].to(device=encoder_hidden_states.device))
            encoder_hidden_states = outputs[0]

        if 'baseline' in self.config.training_type:
            decoder_hidden_states = encoder_hidden_states

            for decoder_layers in self.decoder:
                
                outputs = decoder_layers(decoder_hidden_states, attention_mask=self.encoder_mask[:decoder_hidden_states.shape[1], :decoder_hidden_states.shape[1]].to(decoder_hidden_states.device), output_attentions=True)
                decoder_hidden_states = outputs[0]
            
            hidden_states = decoder_hidden_states

        else:

            # mean pooling
            batch_size, seq_len, hidden_size = encoder_hidden_states.size()
            num_ctx_tokens = seq_len // self.chunk_size  # ctx tokens 的数量
            if self.config.chunk_merge_method == 'meanpooling':
                hidden_states_reshaped = encoder_hidden_states[:,:int(seq_len//self.chunk_size)*self.chunk_size]
                hlm_hidden_states = hidden_states_reshaped.reshape(batch_size,int(seq_len//self.chunk_size),self.chunk_size,hidden_size).mean(dim=-2)

            if 'default' not in self.config.detach_or_not:
                hlm_hidden_states_det = hlm_hidden_states.detach()
                # VQ, only quantize in training
                if self.training:
                    hlm_hs_split = hlm_hidden_states_det.view(hlm_hidden_states_det.size(0), hlm_hidden_states_det.size(1), self.config.n_head, -1)
                    
                    vq_outputs = []
                    vq_indices = []
                    total_commit_loss = 0.0
                    
                    for i in range(self.config.n_head):
                        # for each head
                        quantized, indices, commit_loss = self.vqs[i](hlm_hs_split[:, :, i, :])
                        codebook_rec_loss = ((quantized - hlm_hs_split[:, :, i, :]) ** 2).mean()   
                        vq_outputs.append(quantized)
                        vq_indices.append(indices)
                        total_commit_loss += (commit_loss+codebook_rec_loss)

                    combined_vq_outputs = torch.stack(vq_outputs, dim=2).view(hlm_hidden_states_det.size(0), hlm_hidden_states_det.size(1), -1)
                    combined_vq_indices = torch.stack(vq_indices, dim=2)
                else:
                    total_commit_loss = 0.0
                
                # concept predict
                combined_vq_code, vq_ce_loss, tf_pred_acc, ar_pred_acc, mixed_pred_indices, probs = self.hlm(
                    hlm_hidden_states=hlm_hidden_states,
                    combined_vq_outputs=None,
                    combined_vq_indices=None,
                    use_cache=use_cache,
                    epoch=epoch, 
                )  

                qualitized_hlm_hidden_states = []
                for i in range(int(self.config.n_head / self.config.vq_patch_ratio)):
                    if self.config.distribution_merge:
                        topk = self.config.dist_topk
                        if topk == self.config.vq_config['codebook_size']:
                            codes = self.vqs[i].codebook
                            this_probs = probs[:, :, :, i]
                            weighted_sum = torch.matmul(this_probs, codes)
                            qualitized_hlm_hidden_states.append(weighted_sum)
                    else:
                        qualitized_hlm_hidden_states.append(self.vqs[i].indices_to_codes(combined_vq_code[:, :, i]))

                qualitized_hlm_hidden_states = torch.stack(qualitized_hlm_hidden_states, dim=2).view(hlm_hidden_states_det.size(0), hlm_hidden_states_det.size(1), -1)

            if self.base_model_config.loss_type == 'hlm_MSE_loss':
                pred_MSE_loss = self.mse_loss_fct(qualitized_hlm_hidden_states[:, :-1, :], hlm_hidden_states_det[:, 1:, :].detach())

            hlm_hidden_states_det = hlm_hidden_states.detach()
            qualitized_hlm_hidden_states = torch.cat((hlm_hidden_states_det[:, :1, :], qualitized_hlm_hidden_states[:, :, :]), dim=1)
            qualitized_hlm_hidden_states[:, :1, :] = 0 
            shifted_hlm_hidden_states = qualitized_hlm_hidden_states

            # merge hlm-hs and encoder-hs

            hlm_hidden_states_repeated = shifted_hlm_hidden_states.repeat_interleave(self.chunk_size, dim=1)[:, :seq_len+1, :] # .detach()
            hlm_hidden_states_repeated = hlm_hidden_states_repeated[:, 1:, :]

            encoder_hidden_states = self.ln_f(encoder_hidden_states)
            hlm_hidden_states_repeated = self.ln_f(hlm_hidden_states_repeated)
            
            decoder_hidden_states = encoder_hidden_states + hlm_hidden_states_repeated

            for decoder_layers in self.decoder:
                
                outputs = decoder_layers(decoder_hidden_states, attention_mask=self.encoder_mask[:hidden_states.shape[1], :hidden_states.shape[1]].to(device=encoder_hidden_states.device), output_attentions=True)
                decoder_hidden_states = outputs[0]
            
            hidden_states = decoder_hidden_states
        
        hidden_states = self.ln_f(hidden_states).view(output_shape)
        
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            
            loss = self.ce_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        

        hlm_loss = torch.tensor([0.0]).to(lm_logits.device)

        if 'baseline' in self.config.training_type or 'test' in self.config.training_type:
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                hlm_loss=hlm_loss,
                logits=lm_logits,
                VQ_MSE_loss=loss.detach(),
                VQ_CE_loss=loss.detach(),
                acc=tf_pred_acc.mean(),
                hidden_states=hidden_states,
            )

        else:
            return CausalLMOutputWithCrossAttentions(
                loss=loss, # .detach(), 
                VQ_MSE_loss=total_commit_loss,
                VQ_CE_loss=pred_MSE_loss,
                acc=tf_pred_acc.mean(),
                logits=lm_logits,
                hidden_states=hidden_states,
            )  


    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
    
    @staticmethod
    def load_checkpoint(model, ckpt_dir):
        assert isinstance(model, HLTGPT2)
        # load model.safetensors
        model.load_state_dict(load_file(os.path.join(ckpt_dir,'model.safetensors')))

    def tie_weights(self):
        return
    