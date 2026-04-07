import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.import_utils import is_torch_fx_available
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv
)

from .kv_cache import DynamicCustomCache

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()
        
        self.past_key_value = None  # Will be initialized in init_cache

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[DynamicCustomCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[DynamicCustomCache]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        cache = past_key_value if past_key_value is not None else self.past_key_value
        
        # QKV projection on GPU
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Use position_embeddings if provided (transformers 4.46.2 style), otherwise compute locally
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if cache is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = cache.update(
                key_states,
                value_states,
                layer_idx=self.layer_idx,
                cache_kwargs=cache_kwargs,
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Prepare attention mask like original LlamaAttention
        # Original: causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        causal_mask = None
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        
        if cache is not None:
            attn_output = cache.flash_attention(
                layer_idx=self.layer_idx,
                query=query_states,
                key=key_states,
                value=value_states,
                attn_mask=causal_mask,
                head_dim=self.head_dim,
            )
        else:
            # No-cache path (e.g. use_cache=False forward). Causal masking must
            # still apply: when no explicit 4D mask is provided, delegate to SDPA
            # with is_causal=True so prefill cannot attend to future tokens.
            if causal_mask is None and q_len > 1:
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    attn_mask=None, dropout_p=0.0, is_causal=True,
                )
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                if causal_mask is not None:
                    attn_weights = attn_weights + causal_mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights = nn.functional.dropout(
                    attn_weights,
                    p=self.attention_dropout,
                    training=self.training,
                )
                attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, cache


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
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
        past_key_value: Optional[DynamicCustomCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[DynamicCustomCache]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
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

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Initialize rotary embedding for position embeddings (transformers 4.46.2 style)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.compression_config = None
        # Initialize weights and apply final processing
        self.post_init()
    
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
        This is compatible with transformers 4.46.2 and uses cache_position.
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
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask
    
    def init_cache(self, compression_config):
        """
        Initialize cache configuration. This stores the compression_config which will be used
        by _prepare_cache_for_generation to automatically create the appropriate cache.
        """
        self.compression_config = compression_config
        self.compression_method = None if compression_config is None else compression_config.compression_method
        # Note: Actual cache creation is done in _prepare_cache_for_generation
        # This method just stores the config for later use
        
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCustomCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and past_key_values is not None and not isinstance(past_key_values, DynamicCustomCache):
            raise TypeError("`past_key_values` must be `DynamicCustomCache` for KVLlama.")

        if use_cache and past_key_values is None:
            cache_device = (
                inputs_embeds.device
                if inputs_embeds is not None
                else (input_ids.device if input_ids is not None else self.device)
            )
            past_key_values = DynamicCustomCache(
                config=self.config,
                compression_config=getattr(self, "compression_config", None),
                device=cache_device,
            )

        # Get past_key_values_length from cache
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        device = inputs_embeds.device
        
        # Handle cache_position (transformers 4.46.2)
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=device
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create position embeddings to be shared across decoder layers (transformers 4.46.2)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        
        # attention_mask = attention_mask.to(inputs_embeds.dtype)
        # Use _update_causal_mask for transformers 4.46.2 compatibility
        # Since we use per-layer caches, we need to handle this differently
        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # Issue #6: skip the [B,1,Sq,Sk] mask materialization when there's no
            # padding to encode. Causal masking is generated per-block inside
            # flash_attention (or handled by SDPA's is_causal in the decode path).
            no_padding = (
                attention_mask is None
                or (attention_mask.dim() == 2 and bool((attention_mask == 1).all()))
            )
            if no_padding:
                attention_mask = None
            else:
                target_length = (
                    attention_mask.shape[-1]
                    if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() >= 2
                    else past_key_values_length + seq_length
                )
                attention_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
                    attention_mask=attention_mask,
                    sequence_length=seq_length,
                    target_length=target_length,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                )
        
        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if next_cache is not None:
            # 외부 코드(RL 등) 하위 호환
            self.layer_compressors = next_cache.layer_compressors
            self.layer_caches = self.layer_compressors
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class KVLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
        # Store compression config for use in _prepare_cache_for_generation
        self.compression_config = None
        self.layer_compressors = None
        self.layer_caches = None
    
    def init_cache(self, compression_config):
        """Initialize cache with compression config. Can also be called automatically via _prepare_cache_for_generation."""
        self.compression_config = compression_config
        self.model.init_cache(compression_config)
    
    def _prepare_cache_for_generation(
        self,
        generation_config,
        model_kwargs,
        assistant_model=None,
        batch_size=None,
        max_cache_length=None,
        device=None,
    ):
        """
        Override to automatically create compressed cache. compression_config must be set via init_cache().
        """
        cache_name = "past_key_values"
        
        # If user specifies a cache, use it
        user_defined_cache = model_kwargs.get(cache_name)
        if user_defined_cache is not None:
            return
        
        # If use_cache is False, nothing to do
        if generation_config.use_cache is False:
            return
        
        inferred_device = device
        if inferred_device is None:
            input_ids = model_kwargs.get("input_ids")
            inputs_embeds = model_kwargs.get("inputs_embeds")
            if isinstance(inputs_embeds, torch.Tensor):
                inferred_device = inputs_embeds.device
            elif isinstance(input_ids, torch.Tensor):
                inferred_device = input_ids.device
            else:
                inferred_device = self.device

        model_kwargs[cache_name] = DynamicCustomCache(
            config=self.model.config,
            compression_config=self.compression_config,
            device=inferred_device,
        )
        self.layer_compressors = model_kwargs[cache_name].layer_compressors
        self.layer_caches = self.layer_compressors

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        if past_key_values is not None:
            # DynamicCustomCache follows DynamicCache API including get_seq_length.
            past_length = past_key_values.get_seq_length()

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )

        # Forward additional generation kwargs (e.g., num_logits_to_keep)
        # so memory-saving generation options are not dropped.
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        return model_inputs