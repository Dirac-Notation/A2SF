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

def eviction_mask(attn_weights, streaming_budget, selecting_budget, recent_budget, forgetting_factor):

    # attn_weights (BS, head, query, keys)
    dtype_attn_weights = attn_weights.dtype
    seq_length = attn_weights.shape[-1]
    
    cache_budget = streaming_budget + selecting_budget + recent_budget
    score_shape = attn_weights[:,:,0,:].shape

    select_score = torch.zeros(score_shape, dtype=torch.float, device=attn_weights.device)
    
    attn_mask = torch.ones_like(attn_weights, dtype=torch.bool, device=attn_weights.device)
    attn_scores = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype_attn_weights)
    
    for token_index in range(cache_budget):
        select_score = forgetting_factor*select_score + attn_scores[:,:,token_index,:]

    for token_index in range(cache_budget, seq_length-1):
        # Current Step Calculate
        current_score = attn_scores[:,:,token_index,:]
        current_mask = attn_mask[:,:,token_index,:]
        
        current_score *= current_mask
        current_score /= current_score.sum(dim=-1).unsqueeze(dim=-1)
        
        if forgetting_factor != 0.0:
            select_score = forgetting_factor*select_score + current_score
        else:
            select_score[select_score != torch.inf] = 0 
            select_score += current_score
        
        # Next Mask Make
        local_index = token_index - recent_budget
        min_index = torch.argmin(select_score[:,:,streaming_budget:local_index+1], dim=-1).unsqueeze(dim=-1) + streaming_budget
        select_score.scatter_(-1, min_index, torch.inf)
        attn_mask[:,:,token_index+1,:] = current_mask.scatter(-1, min_index, False)
    
    return attn_mask

class LlamaAttention_heavy_hitter(nn.Module):
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

        self.streaming_ratio = config.streaming_ratio
        self.selecting_ratio = config.selecting_ratio
        self.recent_ratio = config.recent_ratio
        self.forgetting_factor = config.forgetting_factor

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
        
        ### Heavy + Recent
        streaming_budget = math.floor(self.streaming_ratio * hidden_states.shape[-2] + 0.5)
        selecting_budget = math.floor(self.selecting_ratio * hidden_states.shape[-2] + 0.5)
        recent_budget = math.floor(self.recent_ratio * hidden_states.shape[-2] + 0.5)

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
        
        ################################################################################################ start
        
        mask_bottom = eviction_mask(
            attn_weights=attn_weights,
            streaming_budget=streaming_budget,
            selecting_budget=selecting_budget,
            recent_budget=recent_budget,
            forgetting_factor=self.forgetting_factor,
        )

        attn_weights[~mask_bottom] = torch.min(attention_mask)

        ################################################################################################ end

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

def convert_kvcache_llama_heavy_recent(model, config):
    from .ideal_llama import LlamaAttention_heavy_hitter_ideal
    
    for name, module in model._modules.items():
        if isinstance(module, LlamaDecoderLayer):
        #     tmp_heavy_ratio = [0.61, 0.41, 0.24, 0.16, 0.12, 0.17, 0.19, 0.2, 0.21, 0.2, 0.21, 0.21, 0.23, 0.2, 0.21, 0.2, 0.21, 0.19, 0.18, 0.17, 0.17, 0.16, 0.14, 0.1, 0.15, 0.12, 0.17, 0.12, 0.17, 0.18, 0.18, 0.22][int(name)]
        #     if config.recent_ratio > 0.0:
        #         tmp_heavy_ratio /= 2
        #         config.recent_ratio = tmp_heavy_ratio
        #     config.heavy_ratio = tmp_heavy_ratio
            
            # tmp_factor = [0.2, 0.35, 0, 0.3, 0.3, 0.25, 0.35, 0.35, 0.5, 0.4, 0.25, 0.5, 0.45, 0.5, 0.5, 0.5, 0.5, 0.65, 0.5, 0.6, 0.5, 0.45, 0.45, 0.4, 0.5, 0.5, 0.4, 0.4, 0.05, 0.15, 0.05, 0.75][int(name)]
            # config.penalty = tmp_factor
            if config.tmp is not None:
                ratio = config.streaming_ratio + config.selecting_ratio + config.recent_ratio
                if int(name) == config.tmp:
                    config.forgetting_factor = 0.3
                    config.streaming_ratio = 0.0
                    config.selecting_ratio = ratio
                    config.recent_ratio = 0.0
                else:
                    config.forgetting_factor = 1.0
                    config.streaming_ratio = 0.0
                    config.selecting_ratio = ratio/2
                    config.recent_ratio = ratio/2
        
        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_llama_heavy_recent(module, config)
        
        if isinstance(module, LlamaAttention) or isinstance(module, LlamaAttention_heavy_hitter) or isinstance(module, LlamaAttention_heavy_hitter_ideal):
            model._modules[name] = LlamaAttention_heavy_hitter(config)

    return model