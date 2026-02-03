from functools import partial
from typing import Callable, Optional, Tuple, Union

import copy
import torch
import torch.utils.checkpoint
from torch import nn
from dataclasses import dataclass

from vector_quantize_pytorch import SimVQ
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    # CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
    ModelOutput,
)
from transformers.utils.deprecation import deprecate_kwarg
from .configuration_llama import LlamaConfig


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "LlamaConfig"


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
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    pred_loss: Optional[torch.FloatTensor] = None
    VQ_cmt_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None



import torch
from torch.nn import Module


def copy_module_parameters(src: Module, tgt: Module, verbose=True):
    """
    递归复制 src 模块的所有参数到 tgt 模块
    假设 src 和 tgt 结构完全一致，参数数量一致
    
    Args:
        src (Module): 源模块
        tgt (Module): 目标模块
        verbose (bool): 是否打印每个复制的参数
    """
    src_params = dict(src.named_parameters())
    tgt_params = dict(tgt.named_parameters())

    # 检查参数数量是否一致
    if len(src_params) != len(tgt_params):
        raise ValueError(
            f"Parameter count mismatch: src={len(src_params)}, tgt={len(tgt_params)}"
        )

    # 逐个复制
    for name, src_param in src_params.items():
        if name not in tgt_params:
            raise ValueError(f"Parameter {name} missing in target module")
        tgt_param = tgt_params[name]

        if src_param.numel() != tgt_param.numel():
            raise ValueError(
                f"Shape mismatch for {name}: {src_param.shape} vs {tgt_param.shape}"
            )
        tgt_param.data.copy_(src_param.data)
        if verbose:
            print(f"[copied] {name}")

    # 递归处理子模块
    for name, src_child in src.named_children():
        if hasattr(tgt, name):
            copy_module_parameters(src_child, getattr(tgt, name), verbose)

    # 删除源模块
    del src


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # try:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # except:
        
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # try:
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # except:
            

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        # import pdb; pdb.set_trace()
        # 如果是attention mask超长，截断就可以了？这里是漏了处理吗
        # try:
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask[:, :query_states.shape[-2]] if attention_mask is not None else attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        # except:
        #     import pdb; pdb.set_trace()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()



# @add_start_docstrings(
#     "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
#     LLAMA_START_DOCSTRING,
# )
class ConceptLM_Llama_BaseModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, base_model=None):
        super().__init__(config)
        self.MSE_loss = nn.MSELoss()
        self.register_buffer("cached_pred_high_level_hs", None)
        self.register_buffer("cached_encoder_hs_1", None)
        self.register_buffer("cached_encoder_hs_2", None)
        self.register_buffer("cached_encoder_hs_3", None)
        self.register_buffer("cached_encoder_hs_4", None)
        self.config._attn_implementation = 'flash_attention_2'
        if base_model is not None:
            
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size

            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            copy_module_parameters(base_model.model.embed_tokens, self.embed_tokens)

            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            copy_module_parameters(base_model.model.norm, self.norm)

            self.rotary_emb = LlamaRotaryEmbedding(config=config)
            copy_module_parameters(base_model.model.rotary_emb, self.rotary_emb)

            self.encoder = nn.ModuleList(
                [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.encoder_layers)]
            )
            self.decoder = nn.ModuleList(
                [LlamaDecoderLayer(config, layer_idx+config.encoder_layers+config.special_layers) for layer_idx in range(config.decoder_layers)]
            )

            self.hlm_layers = nn.ModuleList(
                [LlamaDecoderLayer(config, layer_idx+config.encoder_layers) for layer_idx in range(config.special_layers)]
            )

            assert(config.encoder_layers + config.decoder_layers == config.num_hidden_layers)

            for (idx, layers) in enumerate(self.encoder):
                copy_module_parameters(base_model.model.layers[idx], self.encoder[idx])

            for (idx, layers) in enumerate(self.decoder):
                copy_module_parameters(base_model.model.layers[idx+config.encoder_layers], self.decoder[idx])
            
            self.vqs = nn.ModuleList([
                SimVQ(
                    dim=int((config.hidden_size / config.num_attention_heads)*(self.config.vq_patch_ratio)),
                    codebook_size=self.config.vq_config['codebook_size'],
                    rotation_trick=True,
                    codebook_transform = nn.Sequential(
                        nn.Linear(int((config.hidden_size * self.config.vq_patch_ratio / config.num_attention_heads)), int((config.hidden_size * self.config.vq_patch_ratio / config.num_attention_heads)) * 2),
                        nn.ReLU(),
                        nn.Linear(int((config.hidden_size * self.config.vq_patch_ratio / config.num_attention_heads)) * 2, int((config.hidden_size * self.config.vq_patch_ratio / config.num_attention_heads)))
                    ),
                ) for _ in range(int(config.num_attention_heads/(self.config.vq_patch_ratio)))
            ])

            self.ctx_lm_head = nn.ModuleList([nn.Linear(int(config.hidden_size), config.vq_config['codebook_size']) for  i in range(config.num_attention_heads)])

            self.gradient_checkpointing = False
            # Initialize weights and apply final processing
            self.post_init()
        else:
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size

            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.rotary_emb = LlamaRotaryEmbedding(config=config)
            
            self.encoder = nn.ModuleList(
                [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.encoder_layers)]
            )
            self.decoder = nn.ModuleList(
                [LlamaDecoderLayer(config, layer_idx+config.encoder_layers+config.special_layers) for layer_idx in range(config.decoder_layers)]
            )

            self.hlm_layers = nn.ModuleList(
                [LlamaDecoderLayer(config, layer_idx+config.encoder_layers) for layer_idx in range(config.special_layers)]
            )

            self.vqs = nn.ModuleList([
                SimVQ(
                    dim=int((config.hidden_size / config.num_attention_heads)*(self.config.vq_patch_ratio)),
                    codebook_size=self.config.vq_config['codebook_size'],
                    rotation_trick=True,
                    codebook_transform = nn.Sequential(
                        nn.Linear(int((config.hidden_size * self.config.vq_patch_ratio / config.num_attention_heads)), int((config.hidden_size * self.config.vq_patch_ratio / config.num_attention_heads)) * 2),
                        nn.ReLU(),
                        nn.Linear(int((config.hidden_size * self.config.vq_patch_ratio / config.num_attention_heads)) * 2, int((config.hidden_size * self.config.vq_patch_ratio / config.num_attention_heads)))
                    ),
                ) for _ in range(int(config.num_attention_heads/(self.config.vq_patch_ratio)))
            ])

            self.ctx_lm_head = nn.ModuleList([nn.Linear(int(config.hidden_size), self.config.vq_config['codebook_size']) for  i in range(self.config.num_attention_heads)])

            self.gradient_checkpointing = False
            # Initialize weights and apply final processing
            self.post_init()

            # import pdb; pdb.set_trace()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        encoder_hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(encoder_hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for encoder_layer in self.encoder:
            layer_outputs = encoder_layer(
                encoder_hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            encoder_hidden_states = layer_outputs[0]
        # import pdb; pdb.set_trace()
        if len(past_key_values)==1:
            self.cached_encoder_hs_1 = encoder_hidden_states[:, -4:-3, :]
            self.cached_encoder_hs_2 = encoder_hidden_states[:, -3:-2, :]
            self.cached_encoder_hs_3 = encoder_hidden_states[:, -2:-1, :]
            self.cached_encoder_hs_4 = encoder_hidden_states[:, -1:, :]

        else:
            if (past_key_values[0][0].shape[2] % past_key_values[1][0].shape[2])%4 == 1:
                self.cached_encoder_hs_1 = encoder_hidden_states

            if (past_key_values[0][0].shape[2] % past_key_values[1][0].shape[2])%4 == 2:
                self.cached_encoder_hs_2 = encoder_hidden_states

            if (past_key_values[0][0].shape[2] % past_key_values[1][0].shape[2])%4 == 3:
                self.cached_encoder_hs_3 = encoder_hidden_states

            if (past_key_values[0][0].shape[2] % past_key_values[1][0].shape[2])%4 == 0:
                self.cached_encoder_hs_4 = encoder_hidden_states

        if 'baseline' in self.config.training_type:
            mid_layers_hidden_states = encoder_hidden_states
            
            for mid_layer in self.hlm_layers:
                layer_outputs = mid_layer(
                    mid_layers_hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )     

                mid_layers_hidden_states = layer_outputs[0]

            decoder_layers_hidden_states = mid_layers_hidden_states

            for decoder_layer in self.decoder:
                layer_outputs = decoder_layer(
                    decoder_layers_hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )     

                decoder_layers_hidden_states = layer_outputs[0]

            hidden_states = self.norm(decoder_layers_hidden_states)

            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            ), torch.tensor([0.0]).to(hidden_states.device), torch.tensor([0.0]).to(hidden_states.device)


        # mean pooling
        batch_size, seq_len, hidden_size = encoder_hidden_states.size()
        num_ctx_tokens = seq_len // self.config.chunk_size  # ctx tokens 的数量
        if self.config.chunk_merge_method == 'meanpooling':
            hidden_states_reshaped = encoder_hidden_states[:,:int(seq_len//self.config.chunk_size)*self.config.chunk_size]
            hlm_hidden_states = hidden_states_reshaped.reshape(batch_size,int(seq_len//self.config.chunk_size),self.config.chunk_size,hidden_size).mean(dim=-2)
        
        if 'Context' in self.config.training_type:
            hlm_hidden_states_det = hlm_hidden_states.detach()
            if len(past_key_values)==1:
                ctx_position_ids = torch.arange(0, hlm_hidden_states_det.shape[-2], dtype=torch.long, device=hlm_hidden_states_det.device) * self.config.chunk_size
                ctx_position_ids = ctx_position_ids.unsqueeze(0)
            else:
                ctx_position_ids = cache_position.detach().unsqueeze(0)

            if len(past_key_values) == self.config.encoder_layers or (past_key_values[0][0].shape[2] % past_key_values[1][0].shape[2])%4 == 1:
                vq_outputs = []
                vq_indices = []
                total_commit_loss = []
                if len(past_key_values) == self.config.encoder_layers:
                    pass
                else:
                    hlm_hidden_states = (self.cached_encoder_hs_1+self.cached_encoder_hs_2+self.cached_encoder_hs_3+self.cached_encoder_hs_4)/4
                    hlm_hidden_states_det = hlm_hidden_states.detach()

                hlm_hs_split = hlm_hidden_states_det.view(hlm_hidden_states_det.size(0), hlm_hidden_states_det.size(1), int(self.config.num_attention_heads / self.config.vq_patch_ratio), -1)

                if self.training:
                    for i in range(int(self.config.num_attention_heads / self.config.vq_patch_ratio)):
                        # 遍历每一个VQ实例并进行处理
                        quantized, indices, commit_loss = self.vqs[i](hlm_hs_split[:, :, i, :])
                        codebook_rec_loss = ((quantized - hlm_hs_split[:, :, i, :]) ** 2).mean()   
                        vq_outputs.append(quantized)
                        vq_indices.append(indices)
                        total_commit_loss.append(commit_loss+codebook_rec_loss)
                    total_commit_loss = torch.stack(total_commit_loss).mean()
                    combined_vq_outputs = torch.stack(vq_outputs, dim=2).view(hlm_hidden_states_det.size(0), hlm_hidden_states_det.size(1), -1)
                    combined_vq_indices = torch.stack(vq_indices, dim=2)
                else:
                    pass

                pred_hlm_hidden_states = hlm_hidden_states
                ctx_position_embeddings = self.rotary_emb(pred_hlm_hidden_states, ctx_position_ids)
                
                # VQ predict
                for hlm_layer in self.hlm_layers:
                    layer_outputs = hlm_layer(
                        pred_hlm_hidden_states,
                        attention_mask=causal_mask,
                        position_ids=ctx_position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=ctx_position_embeddings,
                        **flash_attn_kwargs,
                    )

                    pred_hlm_hidden_states = layer_outputs[0]
            
                vq_code = []
                total_lm_logits = []
                for i in range(int(self.config.num_attention_heads / self.config.vq_patch_ratio)):
                    lm_logits = self.ctx_lm_head[i](pred_hlm_hidden_states)
                    vq_code.append(lm_logits.argmax(-1))
                    total_lm_logits.append(lm_logits)
                combined_vq_code = torch.stack(vq_code, dim=2)
                total_lm_logits = torch.stack(total_lm_logits, dim=3)

                qualitized_hlm_hidden_states = []
                qualitized_hlm_hidden_states_GT = []
                new_matirx_s = []
                for i in range(int(self.config.num_attention_heads / self.config.vq_patch_ratio)):
                    if self.config.distribution_merge:
                        topk = self.config.dist_topk
                        if topk == self.config.vq_config['codebook_size']:
                            codes = self.vqs[i].codebook
                            this_probs = total_lm_logits[:, :, :, i]
                            weighted_sum = torch.matmul(this_probs, codes)
                            qualitized_hlm_hidden_states.append(weighted_sum)
                            qualitized_hlm_hidden_states_GT.append(self.vqs[i].indices_to_codes(combined_vq_code[:, :, i]))
                    else:
                        qualitized_hlm_hidden_states.append(self.vqs[i].indices_to_codes(combined_vq_code[:, :, i]))

                qualitized_hlm_hidden_states = torch.stack(qualitized_hlm_hidden_states, dim=2).view(hlm_hidden_states_det.size(0), hlm_hidden_states_det.size(1), -1)
                qualitized_hlm_hidden_states_GT = torch.stack(qualitized_hlm_hidden_states_GT, dim=2).view(hlm_hidden_states_det.size(0), hlm_hidden_states_det.size(1), -1)
                
                pred_loss = self.MSE_loss(qualitized_hlm_hidden_states[:, :-1, :], qualitized_hlm_hidden_states_GT[:, 1:, :].detach())
            
                qualitized_hlm_hidden_states = torch.cat((hlm_hidden_states_det[:, :1, :], qualitized_hlm_hidden_states[:, :, :]), dim=1)
                qualitized_hlm_hidden_states[:, :1, :] = 0 

                self.cached_pred_high_level_hs = qualitized_hlm_hidden_states.clone()
            else:
                qualitized_hlm_hidden_states = self.cached_pred_high_level_hs[:, -1:, :]

            shifted_hlm_hidden_states = qualitized_hlm_hidden_states
            hlm_hidden_states_repeated = shifted_hlm_hidden_states.repeat_interleave(self.config.chunk_size, dim=1).detach()
            
            if self.config.shift_feature and len(past_key_values) == (self.config.encoder_layers+self.config.special_layers):
                hlm_hidden_states_repeated = torch.cat((hlm_hidden_states_repeated[:, 1:, :], hlm_hidden_states_repeated[:, :1, :]), dim=1)
            decoder_hidden_states = encoder_hidden_states + hlm_hidden_states_repeated[:, :encoder_hidden_states.shape[1], :]

            for decoder_layer in self.decoder:
                layer_outputs = decoder_layer(
                    decoder_hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

                decoder_hidden_states = layer_outputs[0]
            hidden_states = decoder_hidden_states

        hidden_states = self.norm(hidden_states)
        if self.training:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            ), pred_loss,total_commit_loss
        else:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            ), torch.tensor([0.0]).to(hidden_states.device),torch.tensor([0.0]).to(hidden_states.device)

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            if isinstance(attention_mask, BlockMask):
                return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask



class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class ConceptLM_Llama_ForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, base_model=None):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if base_model is not None:
            copy_module_parameters(base_model.lm_head, self.lm_head)
            self.model = ConceptLM_Llama_BaseModel(config, base_model)
        else:
            self.model = ConceptLM_Llama_BaseModel(config)
    
        self.vocab_size = config.vocab_size
        

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs, pred_loss,VQ_cmt_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            pred_loss=pred_loss,
            VQ_cmt_loss=VQ_cmt_loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

