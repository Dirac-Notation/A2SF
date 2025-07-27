import torch
from . import BaseCache

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
    
    def update(self, attn_scores=None):
        """Update cache using H2O method (forgetting_factor == 1)"""
        # First prepare scores, then select
        self.prepare_scores(attn_scores)
        self.select()
    
    def prepare_scores(self, attn_scores):
        """Prepare scores for H2O method (simple accumulation)"""
        attn_scores_shape = attn_scores.shape
        
        attn_scores = attn_scores.view(attn_scores_shape[0], self.num_key_value_heads, -1, *attn_scores_shape[2:]).sum(dim=2)
        
        # For H2O, simply sum attention scores without forgetting factor
        current_score = attn_scores.sum(self.seq_dim)
        if self.score is not None:
            current_score[:,:,:-1] += self.score
        self.score = current_score 
    
    def flash_prepare_scores(self, attn_scores):
        return attn_scores.sum(self.seq_dim)