import torch
from __init__ import BaseCache

class H2OCache(BaseCache):
    """H2O cache implementation (forgetting_factor == 1)"""
    
    def __init__(self, num_key_value_heads: int, seq_dim: int = 2):
        super().__init__(num_key_value_heads, seq_dim)
        self.use_compression = False
        self.total_budget = 0
        self.recent_budget = 0
        self.select_budget = 0
        self.score = None
    
    def init_cache(self, compression_config, layer_idx):
        """Initialize H2O cache settings"""
        self.use_compression = True
        self.total_budget = max(round(compression_config.total_budget * compression_config.layerwise_ratio[layer_idx]), 2)
        self.recent_budget = round(self.total_budget * 0.5)
        self.select_budget = self.total_budget - self.recent_budget
        self.score = None
    
    def update(self, attn_scores):
        """Update cache using H2O method (forgetting_factor == 1)"""
        if not (self.use_compression and self.seq_length > self.total_budget):
            return
        
        attn_scores_shape = attn_scores.shape
        
        attn_scores = attn_scores.view(attn_scores_shape[0], self.num_key_value_heads, -1, *attn_scores_shape[2:]).sum(dim=2)
        
        # For H2O, simply sum attention scores without forgetting factor
        current_score = attn_scores.sum(self.seq_dim)
        if self.score is not None:
            current_score[:,:,:-1] += self.score
        self.score = current_score
        
        # Select tokens to keep
        selected_indices = self.score[:,:,:-self.recent_budget].topk(self.select_budget, dim=-1).indices.sort().values
        
        # Update scores
        self.score = torch.cat((
            self.score.gather(self.seq_dim, selected_indices),
            self.score[:,:,-self.recent_budget:]
        ), dim=self.seq_dim)
        
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