import torch
from . import BaseCache

class AverageCache(BaseCache):
    """Average-based selection cache implementation"""
    
    def __init__(self, num_key_value_heads: int, seq_dim: int = 2):
        super().__init__(num_key_value_heads, seq_dim)
        self.use_compression = False
        self.total_budget = 0
        self.recent_budget = 0
        self.select_budget = 0
        self.score = None
        self.cumulative_count = None
        self.is_prefill = True
    
    def init_cache(self, compression_config, layer_idx):
        """Initialize average cache settings"""
        self.use_compression = True
        self.total_budget = max(round(compression_config.total_budget * compression_config.layerwise_ratio[layer_idx]), 2)
        self.recent_budget = round(self.total_budget * 0.5)
        self.select_budget = self.total_budget - self.recent_budget
        self.score = None
        self.cumulative_count = None
        self.is_prefill = True
    
    def update(self, attn_scores):
        """Update cache using average-based selection method"""
        # First prepare scores, then select
        self.prepare_scores(attn_scores)
        self.select()
    
    def prepare_scores(self, attn_scores):
        """Prepare scores for average-based selection method"""
        attn_scores_shape = attn_scores.shape
        
        attn_scores = attn_scores.view(attn_scores_shape[0], self.num_key_value_heads, -1, *attn_scores_shape[2:]).sum(dim=2)
        
        # Calculate current score
        current_score = attn_scores.sum(self.seq_dim)
        current_count = (attn_scores > 0).sum(self.seq_dim).to(current_score.dtype)
        
        if self.score is None:
            self.score = current_score
            self.cumulative_count = current_count
        else:
            current_score[:,:,:-1] += self.score
            current_count[:,:,:-1] += self.cumulative_count
        self.score = current_score
        self.cumulative_count = current_count
        
        self.score = self.score.to(self.device)
        self.cumulative_count = self.cumulative_count.to(self.device)
        
        # Calculate average scores
        self.score = self.score / self.cumulative_count 