import torch
import math
from . import KVCache

class PyramidCache(KVCache):
    """Pyramid cache implementation"""
    
    def __init__(self, num_key_value_heads: int, seq_dim: int = 2):
        super().__init__(num_key_value_heads, seq_dim)
        self.prompt = False
    
    def init_cache(self, compression_config, layer_idx):
        """Initialize Pyramid cache settings"""
        self.seq_length = 0
        default_budget = compression_config.total_budget
        beta = compression_config.beta
        self.recent_budget = compression_config.recent_budget
        
        min_budget = (default_budget - self.recent_budget) // beta
        max_budget = (default_budget - self.recent_budget) * 2 - min_budget
        
        step = (max_budget - min_budget) // 31
        self.total_budget = max_budget - step * layer_idx
        
        self.select_budget = self.total_budget - self.recent_budget
        self.prompt = False
    
    def select(self, scores):
        if self.prompt:
            return
        
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
        return attn_scores[:,:,-self.recent_budget:].sum(self.seq_dim)