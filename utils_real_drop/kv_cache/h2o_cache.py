import torch
import math
from . import KVCache

class H2OCache(KVCache):
    """H2O cache implementation (forgetting_factor == 1)"""
    
    def __init__(self, num_key_value_heads: int, seq_dim: int = 2):
        super().__init__(num_key_value_heads, seq_dim)
        self.prompt = False
    
    def init_cache(self, compression_config, layer_idx):
        """Initialize H2O cache settings"""
        self.seq_length = 0
        self.total_budget = compression_config.total_budget
        self.recent_budget = round(self.total_budget * 0.5)
        self.select_budget = self.total_budget - self.recent_budget
        self.prompt = False
    
    def select(self, scores):
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
        return attn_scores.sum(self.seq_dim)

    def flash_attention(self, query, key, value, attn_mask, head_dim, block_size=1024):
        if self.prompt:
            self.prompt = True
            return self.prompt_flash_attention(query, key, value, attn_mask, head_dim, block_size)
        else:
            return super().flash_attention(query, key, value, attn_mask, head_dim, block_size)