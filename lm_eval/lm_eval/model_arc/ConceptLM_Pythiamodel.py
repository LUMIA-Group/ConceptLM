import torch
import tqdm
import copy
import random

from torch import nn, einsum
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from ..model_arc.modeling_gpt_neox import GPTNeoXPreTrainedModel, GPTNeoXLayer, GPTNeoXConfig
from transformers import GenerationMixin
from typing import Optional, Tuple, Union

from vector_quantize_pytorch import SimVQ

from transformers.activations import ACT2FN
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from dataclasses import dataclass
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import get_torch_version, is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10, logging, ModelOutput
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

import torch
import torch.nn.functional as F




class Concept_Module_VQ_Pythia(GPTNeoXPreTrainedModel, GenerationMixin):
    def __init__(
        self,
        config,
        type='hlm', 
    ):
        super().__init__(config)
        self.config = config
        self.chunk_size = config.chunk_size
        self.type=type
        self._attn_implementation = 'flash_attention_2'

        self.transformer = nn.ModuleList([GPTNeoXLayer(config) for i in range(self.config.special_layers)])
        self.max_positions = config.max_position_embeddings
        self.embed_dim = self.config.hidden_size
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.loss_fct = CrossEntropyLoss()        

        self.encoder_mask = torch.tril(torch.ones(self.max_positions, self.max_positions))
        self.encoder_mask = self.encoder_mask.masked_fill(self.encoder_mask == 0, float('-inf')) -1

        self.ctx_lm_head = nn.ModuleList([nn.Linear(int(self.embed_dim), self.config.vq_config['codebook_size']) for  i in range(self.config.num_attention_heads)])


    def forward(
        self,
        hlm_hidden_states: Optional[Tuple[torch.FloatTensor]],
        combined_vq_outputs: Optional[Tuple[torch.FloatTensor]],
        combined_vq_indices: Optional[Tuple[torch.IntTensor]],
        epoch = None, 
        router_res = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        topk_ctx_token_ids: Optional[Tuple[torch.Tensor]] = None,
        ranked_distance: Optional[Tuple[torch.Tensor]] = None,
        pred_indices: Optional[Tuple[torch.Tensor]] = None,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:

        # Attention mask.
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions and head_mask is None:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, seq_length),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask=attention_mask,
                input_shape=(batch_size, seq_length),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )

        device = hlm_hidden_states.device
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.transformer))
        else:
            past_length = past_key_values[0][0].size(-2)

        position_ids = torch.arange(past_length, hlm_hidden_states.shape[-2] + past_length, dtype=torch.long, device=device) *self.chunk_size
        position_ids = position_ids.unsqueeze(0)


        for i, (layers, layer_past) in enumerate(zip(self.transformer, past_key_values)):
            outputs = layers(
                hlm_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=None,
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hlm_hidden_states = outputs[0]
        
        hidden_states = self.layer_norm(hlm_hidden_states)
        pred_hlm_hs_split = hidden_states
        
        lm_loss = []
        vq_code = []
        total_lm_logits = []
        mask_list=None
        
        for i in range(int(self.config.num_attention_heads / self.config.vq_patch_ratio)):
            lm_logits = self.ctx_lm_head[i](pred_hlm_hs_split)
            shift_lmlogits = lm_logits[..., :-1, :].contiguous()
            shift_labels = combined_vq_indices[:, 1:, i].contiguous()
            lm_loss.append(self.loss_fct(shift_lmlogits.view(-1, shift_lmlogits.size(-1)), shift_labels.view(-1)))
            vq_code.append(lm_logits.argmax(-1))
            total_lm_logits.append(lm_logits)

        combined_vq_code = torch.stack(vq_code, dim=2)
        total_lm_logits = torch.stack(total_lm_logits, dim=3)
        acc = (combined_vq_code[:, :-1, :] == combined_vq_indices[:, 1:, :]).sum()/(combined_vq_indices.shape[0]*combined_vq_indices.shape[1]*combined_vq_indices.shape[2])
        lm_loss = torch.stack(lm_loss).mean()
    
        return combined_vq_code, lm_loss, acc, None, None, total_lm_logits

    


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    VQ_MSE_loss: Optional[torch.FloatTensor] = None
    VQ_HLM_loss: Optional[torch.FloatTensor] = None
    acc: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    VQ_MSE_loss: Optional[torch.FloatTensor] = None



def exists(val):
    return val is not None




class ConceptLM_Pythia(GPTNeoXPreTrainedModel):

    def __init__(self, config, hlm_config=None, attn_resampling=True):
        super().__init__(config)

        self.base_model_config = config 
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.chunk_size = self.base_model_config.chunk_size

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.max_positions = config.max_position_embeddings
        self._attn_implementation = 'flash_attention_2'

        self.embed_in = nn.Embedding(config.vocab_size, self.embed_dim)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.emb_dropout = nn.Dropout(0.)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        

        self.encoder = nn.ModuleList([GPTNeoXLayer(config) for i in range(self.base_model_config.encoder_layers)])

        if 'baseline' in self.base_model_config.training_type:
            self.hlm = None

        elif 'vqaddfeature' not in self.base_model_config.training_type:
            self.hlm = Concept_Module_VQ_Pythia(
                config, 
            )
            if 'vq_gpt2_ce' in self.base_model_config.hlm_type:
                self.vqs = nn.ModuleList([
                    SimVQ(
                        dim=int((config.hidden_size / config.num_attention_heads)*(self.base_model_config.vq_patch_ratio)),
                        codebook_size=self.base_model_config.vq_config['codebook_size'],
                        rotation_trick=True,
                        codebook_transform = nn.Sequential(
                            nn.Linear(int((config.hidden_size * self.base_model_config.vq_patch_ratio / config.num_attention_heads)), int((config.hidden_size * self.base_model_config.vq_patch_ratio / config.num_attention_heads)) * 2),
                            nn.ReLU(),
                            nn.Linear(int((config.hidden_size * self.base_model_config.vq_patch_ratio / config.num_attention_heads)) * 2, int((config.hidden_size * self.base_model_config.vq_patch_ratio / config.num_attention_heads)))
                        ),
                    ) for _ in range(int(config.num_attention_heads/(self.base_model_config.vq_patch_ratio)))
                ])


        self.gradient_checkpointing = False
        self.decoder = nn.ModuleList([GPTNeoXLayer(config) for i in range(self.base_model_config.decoder_layers)])
        self.init_masks()
        self.post_init()
        self.MSE_loss = MSELoss()
        self.ce_loss_fct = nn.CrossEntropyLoss()


    def init_masks(self):

        self.encoder_mask = torch.tril(torch.ones(self.max_positions, self.max_positions))
        self.encoder_mask = self.encoder_mask.masked_fill(self.encoder_mask == 0, float('-inf')) -1

        self.decoder_mask = self.ctx_token_attention_mask(int(self.max_positions // self.chunk_size) + self.max_positions, self.chunk_size+1, self.chunk_size+1)

    def ctx_token_attention_mask(self, seq_len, window_size, window_position):
        
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = torch.ones(seq_len, seq_len)
        # 处理每个 window 的额外规则

        for start in range(0, seq_len, window_size):
            mask[start+window_size-1:, start] = 0

        mask *= causal_mask
        for i in range(mask.shape[0]):
            # for j in range(mask.shape[1]):
            if i % window_size==4 and i != seq_len-1: #  and j % (window_size+1)==0 and i>j:
                # 
                mask[i, i+1]=1

        mask = mask.masked_fill(mask == 0, float('-inf')) -1
        
        return mask

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.embed_out.to(self.transformer.last_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.embed_out.to('cpu')
        self.model_parallel = False
        torch.cuda.empty_cache()

    def forward(
        self,
        epoch = None,
        input_ids: Optional[torch.LongTensor] = None,
        inps: Optional[torch.LongTensor] = None,
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        input_ids = inps if input_ids is None else input_ids
        # input_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).cuda()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:

            if attention_mask is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                attention_mask = torch.ones(input_ids.shape[0], input_ids.shape[1], dtype=torch.long)

            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        # import pdb; pdb.set_trace()
        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.num_hidden_layers)
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        # Attention mask.
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions and head_mask is None:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, seq_length),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask=attention_mask,
                input_shape=(batch_size, seq_length),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
 
        encoder_hidden_states = self.emb_dropout(inputs_embeds)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # encoder forward
        for i, (encoder_layers, layer_past) in enumerate(zip(self.encoder, past_key_values)):

            outputs = encoder_layers(
                encoder_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            encoder_hidden_states = outputs[0]
        
        # baseline
        if 'baseline' in self.base_model_config.training_type:
            decoder_hidden_states = encoder_hidden_states
            for i, (decoder_layers, layer_past) in enumerate(zip(self.decoder, past_key_values)):
                outputs = decoder_layers(
                    decoder_hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                decoder_hidden_states = outputs[0]
            hidden_states = decoder_hidden_states
            hidden_states = self.final_layer_norm(hidden_states)        
            lm_logits = self.embed_out(hidden_states)
            loss = None

            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(lm_logits.device)
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(dtype=torch.long)
                # Flatten the tokens
                loss = self.ce_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss, 
                VQ_MSE_loss=loss,
                VQ_HLM_loss=loss,
                acc=loss,
                logits=lm_logits,
                hidden_states=hidden_states,
            )


        # mean pooling
        batch_size, seq_len, hidden_size = encoder_hidden_states.size()
        num_ctx_tokens = seq_len // self.chunk_size  # number of concepts
        if self.base_model_config.chunk_merge_method == 'meanpooling':
            hidden_states_reshaped = encoder_hidden_states[:,:int(seq_len//self.chunk_size)*self.chunk_size]
            hlm_hidden_states = hidden_states_reshaped.reshape(batch_size,int(seq_len//self.chunk_size),self.chunk_size,hidden_size).mean(dim=-2)
        
        if 'addvqfeature' in self.base_model_config.training_type:
            hlm_hidden_states_det = hlm_hidden_states.detach()
            # VQ
            vq_outputs = []
            vq_indices = []
            total_commit_loss = []
            hlm_hs_split = hlm_hidden_states_det.view(hlm_hidden_states_det.size(0), hlm_hidden_states_det.size(1), int(self.config.num_attention_heads / self.base_model_config.vq_patch_ratio), -1)
            for i in range(int(self.config.num_attention_heads / self.base_model_config.vq_patch_ratio)):
                # for each codebook
                quantized, indices, commit_loss = self.vqs[i](hlm_hs_split[:, :, i, :])
                codebook_rec_loss = ((quantized - hlm_hs_split[:, :, i, :]) ** 2).mean()   
                vq_outputs.append(quantized)
                vq_indices.append(indices)
                total_commit_loss.append(commit_loss+codebook_rec_loss)
            total_commit_loss = torch.stack(total_commit_loss).mean()
            combined_vq_outputs = torch.stack(vq_outputs, dim=2).view(hlm_hidden_states_det.size(0), hlm_hidden_states_det.size(1), -1)
            combined_vq_indices = torch.stack(vq_indices, dim=2)
        
            # concept level prediction
            combined_vq_code, vq_ce_loss, term, term, mixed_pred_indices, probs = self.hlm(
                hlm_hidden_states=hlm_hidden_states,
                combined_vq_outputs=combined_vq_outputs,
                combined_vq_indices=combined_vq_indices,
                use_cache=use_cache,
                epoch=epoch, 
            )  

            # information process
            qualitized_hlm_hidden_states = []
            qualitized_hlm_hidden_states_GT = []
            new_matirx_s = []
            for i in range(int(self.config.num_attention_heads / self.base_model_config.vq_patch_ratio)):
                if self.base_model_config.distribution_merge:
                    topk = self.base_model_config.dist_topk
                    if topk == self.base_model_config.vq_config['codebook_size']:
                        codes = self.vqs[i].codebook
                        this_probs = probs[:, :, :, i]
                        weighted_sum = torch.matmul(this_probs, codes)
                        qualitized_hlm_hidden_states.append(weighted_sum)
                        qualitized_hlm_hidden_states_GT.append(self.vqs[i].indices_to_codes(combined_vq_code[:, :, i]))
                    
                else:
                    qualitized_hlm_hidden_states.append(self.vqs[i].indices_to_codes(combined_vq_code[:, :, i]))

            if self.base_model_config.add_GT_ratio != 0:
                new_matirx_s = torch.stack(new_matirx_s, dim=-1)
            
            qualitized_hlm_hidden_states = torch.stack(qualitized_hlm_hidden_states, dim=2).view(hlm_hidden_states_det.size(0), hlm_hidden_states_det.size(1), -1)
            qualitized_hlm_hidden_states_GT = torch.stack(qualitized_hlm_hidden_states_GT, dim=2).view(hlm_hidden_states_det.size(0), hlm_hidden_states_det.size(1), -1)
            
            pred_MSE_loss = self.MSE_loss(qualitized_hlm_hidden_states[:, :-1, :], hlm_hidden_states_det[:, 1:, :].detach())


            if 'addvqfeature' in self.base_model_config.training_type:

                qualitized_hlm_hidden_states = torch.cat((hlm_hidden_states_det[:, :1, :], qualitized_hlm_hidden_states[:, :, :]), dim=1)
                qualitized_hlm_hidden_states[:, :1, :] = 0 
                shifted_hlm_hidden_states = qualitized_hlm_hidden_states


            hlm_hidden_states_repeated = shifted_hlm_hidden_states.repeat_interleave(self.chunk_size, dim=1)[:, :seq_len+1, :]

            if self.base_model_config.shift_feature:
                hlm_hidden_states_repeated = hlm_hidden_states_repeated[:, 1:, :]

            
            # information interaction
            if self.base_model_config.layer_norm_option == 'rawadd':
                decoder_hidden_states = encoder_hidden_states + hlm_hidden_states_repeated
            
            # decoder
            for i, (decoder_layers, layer_past) in enumerate(zip(self.decoder, past_key_values)):
                outputs = decoder_layers(
                    decoder_hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                decoder_hidden_states = outputs[0]
            hidden_states = decoder_hidden_states

        # extract hlm-hs
        hidden_states = self.final_layer_norm(hidden_states)        
        lm_logits = self.embed_out(hidden_states)
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(dtype=torch.long)
            # Flatten the tokens
            loss = self.ce_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output


        if self.base_model_config.add_GT_ratio != 0:
            acc = (combined_vq_indices == new_matirx_s).sum()/(new_matirx_s.shape[0]*new_matirx_s.shape[1]*new_matirx_s.shape[2])
        else:
            acc = (combined_vq_code[:, :-1, :]==combined_vq_indices[:, 1:, :]).sum()/(combined_vq_indices.shape[0]*combined_vq_indices.shape[1]*combined_vq_indices.shape[2])

        vq_hlm_loss = pred_MSE_loss

        return CausalLMOutputWithPast(
            loss=loss, # .detach(), 
            VQ_MSE_loss=total_commit_loss, 
            VQ_HLM_loss=vq_hlm_loss, 
            acc=acc,
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

