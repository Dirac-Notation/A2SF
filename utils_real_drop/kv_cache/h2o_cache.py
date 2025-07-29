import torch
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
        self.score = None
        self.prompt = False
    
    def update(self, attn_scores=None):
        """Update cache using H2O method (forgetting_factor == 1)"""
        # First prepare scores, then select
        if not self.prompt:
            self.prepare_scores(attn_scores)
            self.select()
            self.prompt = True
        else:
            return
    
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