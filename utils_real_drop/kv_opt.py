"""KV-compression-enabled OPT (mirrors kv_llama.py, but OPT is quite different).

OPT vs Llama:
  - NO RoPE; absolute learned positional embeddings added at the model input.
  - NO GQA (num_kv_heads == num_heads); standard multi-head attention.
  - nn.LayerNorm (not RMSNorm); ReLU FFN with separate fc1/fc2.
  - projection names: q_proj/k_proj/v_proj/out_proj, bias = config.enable_bias.
  - optional project_in/project_out when word_embed_proj_dim != hidden_size.
  - do_layer_norm_before controls pre/post norm placement.

Integration notes:
  - OPT scales the query by 1/sqrt(head_dim). We do NOT pre-scale here; instead
    compressed_attention applies the same 1/sqrt(head_dim) via its head_dim arg,
    giving identical scores.
  - cache.update is rotary-agnostic (it just concatenates), so no sin/cos needed.
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import (
    OPTLearnedPositionalEmbedding,
    OPTPreTrainedModel,
    OPTForCausalLM,
)

from .cache import CompressedKVCache
from .attention import compressed_attention

logger = logging.get_logger(__name__)


class OPTAttention(nn.Module):
    def __init__(self, config: OPTConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        # OPT has no GQA; expose num_key_value_heads for cache/scorer symmetry.
        self.num_key_value_heads = self.num_heads
        self.enable_bias = config.enable_bias
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.past_key_value = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[CompressedKVCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[CompressedKVCache]]:
        bsz, q_len, _ = hidden_states.size()
        cache = past_key_value if past_key_value is not None else self.past_key_value

        # No query pre-scaling: compressed_attention applies 1/sqrt(head_dim).
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if cache is not None:
            key_states, value_states = cache.update(
                key_states, value_states, layer_idx=self.layer_idx,
                cache_kwargs={"cache_position": cache_position},
            )

        causal_mask = None
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        scorer = cache.get_scorer(self.layer_idx) if cache is not None else None
        selector = cache.selector if cache is not None else None
        if selector is not None and not selector.needs_scores(self.layer_idx):
            scorer = None
        attn_output, scores = compressed_attention(
            query_states, key_states, value_states,
            scorer=scorer, attn_mask=causal_mask, head_dim=self.head_dim,
        )
        is_prefill = query_states.size(-2) > 1
        if cache is not None and selector is not None and is_prefill:
            cache.compress(self.layer_idx, scores, seq_len_k=key_states.size(-2))

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None, cache


class OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(config=config, layer_idx=layer_idx)
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[CompressedKVCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            past_key_value=past_key_value, output_attentions=output_attentions,
            use_cache=use_cache, cache_position=cache_position,
        )
        hidden_states = residual + hidden_states
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully connected (with OPT's reshape-to-2D dance)
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = (residual + hidden_states).view(hidden_states_shape)
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class OPTDecoder(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        self.project_out = (
            nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
            if config.word_embed_proj_dim != config.hidden_size else None
        )
        self.project_in = (
            nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
            if config.word_embed_proj_dim != config.hidden_size else None
        )
        self.final_layer_norm = (
            nn.LayerNorm(config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine)
            if (config.do_layer_norm_before and not config._remove_final_layer_norm) else None
        )

        self.layers = nn.ModuleList([OPTDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.compression_config = None
        self.post_init()

    def init_cache(self, compression_config):
        self.compression_config = compression_config
        self.compression_method = None if compression_config is None else compression_config.compression_method

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[CompressedKVCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            cache_device = input_ids.device if input_ids is not None else inputs_embeds.device
            past_key_values = CompressedKVCache(
                config=self.config, compression_config=getattr(self, "compression_config", None),
                device=cache_device,
            )
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        mask_seq_length = past_key_values_length + seq_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        device = inputs_embeds.device

        # 2D attention mask for the learned positional embeddings (ones if absent).
        attn_2d = attention_mask
        if attn_2d is None or attn_2d.dim() != 2:
            attn_2d = torch.ones(batch_size, mask_seq_length, device=device)
        pos_embeds = self.embed_positions(attn_2d, past_key_values_length, position_ids=position_ids)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)
        hidden_states = inputs_embeds + pos_embeds

        # Causal mask: skip materialization with no padding (compressed_attention
        # handles causal internally); otherwise build the standard 4D mask.
        no_padding = attention_mask is None or (attention_mask.dim() == 2 and bool((attention_mask == 1).all()))
        if no_padding:
            layer_mask = None
        else:
            layer_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        if position_ids is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=device
            )
        else:
            cache_position = None

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states, attention_mask=layer_mask, past_key_value=past_key_values,
                output_attentions=output_attentions, use_cache=use_cache, cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=next_cache,
            hidden_states=all_hidden_states, attentions=all_self_attns,
        )


class OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoder(config)
        self.post_init()

    def init_cache(self, compression_config):
        self.decoder.init_cache(compression_config)

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    def forward(self, *args, **kwargs) -> Union[Tuple, BaseModelOutputWithPast]:
        return_dict = kwargs.get("return_dict", None)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        decoder_outputs = self.decoder(*args, **kwargs)
        if not return_dict:
            return decoder_outputs
        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class KVOPTForCausalLM(OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        self.post_init()
        self.compression_config = None

    def init_cache(self, compression_config):
        self.compression_config = compression_config
        self.model.init_cache(compression_config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[CompressedKVCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,   # accepted + ignored (OPT has no RoPE)
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model.decoder(
            input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, position_ids=position_ids,
        )
        hidden = outputs.last_hidden_state if return_dict else outputs[0]
        if num_logits_to_keep:
            hidden = hidden[:, -num_logits_to_keep:, :]
        logits = self.lm_head(hidden).contiguous()
        if not return_dict:
            return (logits,) + (outputs[1:] if isinstance(outputs, tuple) else ())
        return CausalLMOutputWithPast(
            loss=None, logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, assistant_model=None,
        batch_size=None, max_cache_length=None, device=None,
    ):
        if model_kwargs.get("past_key_values") is not None:
            return
        if generation_config.use_cache is False:
            return
        inferred_device = device
        if inferred_device is None:
            input_ids = model_kwargs.get("input_ids")
            inferred_device = input_ids.device if isinstance(input_ids, torch.Tensor) else self.device
        model_kwargs["past_key_values"] = CompressedKVCache(
            config=self.config, compression_config=self.compression_config, device=inferred_device,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        inputs_embeds=None, **kwargs,
    ):
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value
        return model_inputs
