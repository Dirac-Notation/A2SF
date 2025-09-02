import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import copy
import math
import os

from typing import Optional, Tuple
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv


class SKLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
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
        
        self.register_buffer("skew_matrix", torch.eye(self.head_dim, self.head_dim)[None,:,:].expand(self.num_heads, -1, -1))
        self.skewed_dim = self.head_dim
        
        self.past_key_value = None  # Will be initialized in init_cache

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def flash_attention(self, query, key, value, attn_mask, head_dim, block_size=1024):
        """
        Real Flash Attention implementation with chunked processing
        Args:
            query: [batch_size, num_heads, seq_len_q, head_dim]
            key: [batch_size, num_heads, seq_len_k, head_dim]
            value: [batch_size, num_heads, seq_len_k, head_dim]
            attn_mask: [batch_size, 1, seq_len_q, seq_len_k] or None
            head_dim: dimension of each attention head
            block_size: size of chunks for memory-efficient processing
        Returns:
            output: [batch_size, num_heads, seq_len_q, head_dim]
        """
        device = query.device
        dtype = query.dtype
        
        batch_size, num_heads, seq_len_q, _ = query.shape
        _, _, seq_len_k, _ = key.shape
        
        sm_scale = torch.tensor(1.0 / math.sqrt(head_dim), device=device, dtype=dtype)
        
        output = torch.zeros((*query.shape[:-1], head_dim), device=device, dtype=dtype)
        running_max = torch.full((batch_size, num_heads, seq_len_q, 1), float('-inf'), device=device, dtype=dtype)
        running_sum = torch.zeros((batch_size, num_heads, seq_len_q, 1), device=device, dtype=dtype)
        
        # Process key-value pairs in chunks
        for k_start in range(0, seq_len_k, block_size):
            k_end = min(k_start + block_size, seq_len_k)
            
            # Extract current chunk of key and value
            k_chunk = key[:, :, k_start:k_end, :]  # [batch_size, num_heads, chunk_size, head_dim]
            v_chunk = value[:, :, k_start:k_end, :]  # [batch_size, num_heads, chunk_size, head_dim]
            
            # Compute attention scores for this chunk
            scores = torch.matmul(query, k_chunk.transpose(-2, -1)).mul_(sm_scale)  # [batch_size, num_heads, seq_len_q, chunk_size]
            
            # Apply attention mask if provided (includes causal masking)
            if attn_mask is not None:
                scores.add_(attn_mask[:, :, :, k_start:k_end])
            
            # Update running statistics using flash attention algorithm
            new_max = torch.maximum(running_max, torch.max(scores, dim=-1, keepdim=True)[0])
            
            # Compute exponential terms and update running sum
            scores.sub_(new_max).exp_()  # Subtract new_max and apply exp in-place
            running_sum.mul_(torch.exp(running_max - new_max)).add_(torch.sum(scores, dim=-1, keepdim=True))
            
            # Update output
            output.mul_(torch.exp(running_max - new_max)).add_(torch.matmul(scores, v_chunk))
            
            # Update running max
            running_max.copy_(new_max)
        
        # Final normalization
        output = output / running_sum
        
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        # QKV projection on GPU
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        key_states = key_states @ self.skew_matrix

        # reuse k, v, self_attention
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
        
        key_states = key_states @ self.skew_matrix.transpose(1,2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        attn_output = self.flash_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attention_mask,
            head_dim=self.head_dim
        )

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

        return attn_output, attn_weights, past_key_value

def skew_model(model):
    for layer_idx, layer in enumerate(model.model.layers):
        device = layer.self_attn.q_proj.weight.device
        dtype = layer.self_attn.q_proj.weight.dtype
        
        checkpoint = copy.deepcopy(layer.self_attn.state_dict())
        layer.self_attn = SKLlamaAttention(layer.self_attn.config)
        layer.self_attn.load_state_dict(checkpoint, strict=False)
        k_skew_matrix = torch.load(os.path.join("models_skew", "skew_matrix", "llama3", f"layer{layer_idx}", "k_skew_matrix.pt"))
        skewed_dim = 120
        layer.self_attn.skew_matrix = k_skew_matrix[:,:,:skewed_dim]
        layer.self_attn.skewed_dim = skewed_dim
        layer.self_attn.to(device, dtype)