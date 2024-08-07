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

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, LlamaDecoderLayer, apply_rotary_pos_emb

__all__ = ['convert_kvcache_llama_heavy_recent', 'LlamaAttention_heavy_hitter']

def heavy_hitter_ideal_mask(attn_weights, heavy_budget):

    # attn_weights (BS, head, query, keys)
    dtype_attn_weights = attn_weights.dtype
    seq_length = attn_weights.shape[-1]
    
    cache_budget = heavy_budget
    
    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom[:,:,:cache_budget+1,:cache_budget+1] = True # First Token
    mask_bottom = torch.tril(mask_bottom, diagonal=0)

    for token_index in range(cache_budget+1, seq_length):
        # Current Step Calculate
        
        tmp_attn = torch.softmax(attn_weights[:,:,token_index,:], dim=-1, dtype=torch.float32).to(dtype_attn_weights)
                    
        # Next Mask Make
        _, tmp_topk_index = torch.topk(tmp_attn, k=heavy_budget, dim=-1)
        mask_bottom[:,:,token_index,:] = mask_bottom[:,:,token_index,:].scatter(-1, tmp_topk_index, True) # (head, keys)
    
    return mask_bottom

class LlamaAttention_heavy_hitter_ideal(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        self.heavy_budget_ratio = config.heavy_ratio
        self.recent_budget_ratio = config.recent_ratio
        self.penalty = config.penalty

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
            
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        ### Heavy
        heavy_budget = int(self.heavy_budget_ratio * attn_weights.shape[-1])
        
        # Heavy Hitter Mask
        mask_bottom = heavy_hitter_ideal_mask(
            attn_weights=attn_weights,
            heavy_budget=heavy_budget,
        ) # Default: No padding applied to input
        
        attn_weights[~mask_bottom] = torch.min(attention_mask)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value

def convert_kvcache_llama_heavy_recent_ideal(model, config):
    from .modify_llama import LlamaAttention_heavy_hitter
    
    for name, module in reversed(model._modules.items()):
        if isinstance(module, LlamaDecoderLayer):
            tmp_heavy_ratio = [0.70, 0.40, 0.24, 0.18, 0.12, 0.17, 0.20, 0.19, 0.21, 0.19, 0.21, 0.22, 0.21, 0.20, 0.21, 0.20, 0.22, 0.19, 0.20, 0.18, 0.16, 0.12, 0.13, 0.13, 0.13, 0.12, 0.19, 0.14, 0.15, 0.12, 0.16, 0.22][int(name)]
            if config.recent_ratio > 0.0:
                tmp_heavy_ratio /= 2
                config.recent_ratio = tmp_heavy_ratio
            config.heavy_ratio = tmp_heavy_ratio
            
        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_llama_heavy_recent_ideal(module, config)

        if isinstance(module, LlamaAttention) or isinstance(module, LlamaAttention_heavy_hitter) or isinstance(module, LlamaAttention_heavy_hitter_ideal):
            model._modules[name] = LlamaAttention_heavy_hitter_ideal(config)

    return model