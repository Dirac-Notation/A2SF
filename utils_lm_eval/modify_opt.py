import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.opt.modeling_opt import OPTAttention


__all__ = ['convert_kvcache_opt_heavy_recent', 'OPTAttention_Mask']

def local_heavy_hitter_mask(attn_weights, heavy_budget, recent_budget, penalty, value=None):

    # attn_weights (head, query, keys)
    dtype_attn_weights = attn_weights.dtype
    seq_length = attn_weights.shape[-1]

    cache_budget = heavy_budget + recent_budget

    tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype_attn_weights)

    penalty_factor = torch.arange(cache_budget,0,-1).unsqueeze(1).to(dtype_attn_weights).to(attn_weights.device) - 1
    penalty_factor = penalty**penalty_factor

    penaltied_attn = tmp_attn[:,:cache_budget,:] * penalty_factor
    accumulated_attention_score = torch.sum(penaltied_attn, dim=-2) #(head, keys)
    accumulated_attention_score[:,cache_budget:] = 0

    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom[:, :cache_budget, :cache_budget] = True

    for token_index in range(cache_budget, seq_length):
        local_index = token_index-recent_budget
        
        mask_bottom_index = torch.zeros_like(accumulated_attention_score, dtype=torch.bool)
        
        if heavy_budget > 0:
            accumulated_attention_score[:,:local_index] = mask_bottom[:,token_index-1,:local_index]*accumulated_attention_score[:,:local_index]
            if value is not None:
                accumulated_attention_score_value = accumulated_attention_score * torch.mean(torch.abs(value), dim=-1)
                _, tmp_topk_index = torch.topk(accumulated_attention_score_value[:,:local_index], k=heavy_budget, dim=-1)
            else:
                _, tmp_topk_index = torch.topk(accumulated_attention_score[:,:local_index], k=heavy_budget, dim=-1)
            mask_bottom_index = mask_bottom_index.scatter(-1, tmp_topk_index, True) # (head, keys)
        
        mask_bottom_index[:,local_index:] = True # recent
        
        mask_bottom[:,token_index,:] = mask_bottom_index

        tmp_attn_index = mask_bottom_index * attn_weights[:,token_index,:] + ~mask_bottom_index*torch.finfo(attn_weights.dtype).min
        tmp_attn_index = torch.softmax(tmp_attn_index, dim=-1, dtype=torch.float32).to(dtype_attn_weights)

        accumulated_attention_score = accumulated_attention_score * penalty + tmp_attn_index

    mask_bottom = torch.tril(mask_bottom)

    return mask_bottom

class OPTAttention_Mask(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        heavy_ratio: float,
        recent_ratio: float,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        penalty: float = 1.0,
        value_mode: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.heavy_budget_ratio = heavy_ratio
        self.recent_budget_ratio = recent_ratio
        self.penalty = penalty
        self.value_mode = value_mode

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        ### Heavy + Recent
        heavy_budget = int(self.heavy_budget_ratio * attn_weights.shape[-1])
        recent_budget = int(self.recent_budget_ratio * attn_weights.shape[-1])

        # Heavy Hitter Mask
        mask_bottom = local_heavy_hitter_mask(
            attn_weights=attn_weights,
            heavy_budget=heavy_budget,
            recent_budget=recent_budget,
            penalty=self.penalty,
            value=value_states if self.value_mode else None
        ) # Default: No padding applied to input

        attn_weights[~mask_bottom] = torch.min(attention_mask)


        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


def convert_kvcache_opt_heavy_recent(model, config):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_opt_heavy_recent(module, config)

        if isinstance(module, OPTAttention) or isinstance(module, OPTAttention_Mask):
            model._modules[name] = OPTAttention_Mask(
                embed_dim=module.embed_dim,
                num_heads=config.num_attention_heads,
                heavy_ratio = config.heavy_ratio,
                recent_ratio = config.recent_ratio,
                dropout=config.attention_dropout,
                is_decoder=True,
                bias=config.enable_bias,
                penalty=config.penalty,
                value_mode=config.value_mode
            )
    return model