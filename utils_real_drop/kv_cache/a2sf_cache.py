import torch
import math
from . import KVCache

class A2SFCache(KVCache):
    """A2SF cache implementation (forgetting_factor != 1)"""
    
    def __init__(self, num_key_value_heads: int, seq_dim: int = 2):
        super().__init__(num_key_value_heads, seq_dim)
        self.forgetting_factor = None
        self.exponents = None
        self.input_ids = None
        self.prompt = False
    
    def init_cache(self, compression_config, layer_idx):
        """Initialize A2SF cache settings"""
        self.seq_length = 0
        self.total_budget = max(round(compression_config.total_budget * compression_config.layerwise_ratios[layer_idx]), 2)
        self.recent_budget = round(self.total_budget * compression_config.local_ratios)
        self.select_budget = self.total_budget - self.recent_budget
        self.forgetting_factor = compression_config.forgetting_factors[layer_idx]
        self.input_ids = None
        self.prompt = False

    def select(self, scores):
        """Common selection logic for all cache implementations"""
        if self.seq_length <= self.total_budget:
            return
        
        # Select tokens to keep (common logic)
        selected_indices = scores[:,:,:-self.recent_budget].topk(self.select_budget, dim=-1).indices.sort().values
        
        # Update key-value cache
        selected_indices = selected_indices.unsqueeze(-1).expand(-1,-1,-1,self.key_data.size(-1))
        
        self.key_data = torch.cat((
            self.key_data.gather(self.seq_dim, selected_indices),
            self.key_data[:,:,-self.recent_budget:,:]
        ), dim=self.seq_dim)
        
        self.value_data = torch.cat((
            self.value_data.gather(self.seq_dim, selected_indices),
            self.value_data[:,:,-self.recent_budget:,:]
        ), dim=self.seq_dim)

    def flash_prepare_scores(self, attn_scores):
        seq_len = attn_scores.size(2)
        
        forgetting = (self.forgetting_factor ** self.exponents.to(attn_scores.device)).view(1, 1, seq_len, 1)
        return (forgetting * attn_scores).sum(dim=self.seq_dim)

    def flash_attention(self, query, key, value, attn_mask, head_dim, block_size=1024):
        if self.prompt:
            self.prompt = True
            return self.prompt_flash_attention(query, key, value, attn_mask, head_dim, block_size)
        else:
            return super().flash_attention(query, key, value, attn_mask, head_dim, block_size)