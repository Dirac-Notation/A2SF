import torch
from __init__ import BaseCache

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
        if not (self.use_compression and self.seq_length > self.total_budget):
            return
        
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
        average_score = self.score / self.cumulative_count
        
        # Select tokens to keep based on average scores
        selected_indices = average_score[:,:,:-self.recent_budget].topk(self.select_budget, dim=-1).indices.sort().values
        
        # Update scores and counts
        self.score = torch.cat((
            self.score.gather(self.seq_dim, selected_indices),
            self.score[:,:,-self.recent_budget:]
        ), dim=self.seq_dim)
        
        self.cumulative_count = torch.cat((
            self.cumulative_count.gather(self.seq_dim, selected_indices),
            self.cumulative_count[:,:,-self.recent_budget:]
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