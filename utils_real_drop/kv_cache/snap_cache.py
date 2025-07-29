import torch
from . import KVCache

class SnapCache(KVCache):
    """Snap cache implementation"""
    
    def __init__(self, num_key_value_heads: int, seq_dim: int = 2):
        super().__init__(num_key_value_heads, seq_dim)
        self.prompt = False
    
    def init_cache(self, compression_config, layer_idx):
        """Initialize Snap cache settings"""
        super().init_cache(compression_config, layer_idx)
        self.total_budget = max(round(compression_config.total_budget * compression_config.layerwise_ratio[layer_idx]), 2)
        self.recent_budget = 8 if self.total_budget > 8 else self.total_budget-1
        self.select_budget = self.total_budget - self.recent_budget
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
        current_score = attn_scores[:,:,-self.recent_budget:].sum(self.seq_dim)
        
        if self.score is not None:
            current_score[:,:,:-1] += self.score
        self.score = current_score
    
    def flash_prepare_scores(self, attn_scores):
        if not self.prompt:
            return attn_scores[:,:,-self.recent_budget:].sum(self.seq_dim)
        else:
            return torch.zeros_like(attn_scores.sum(self.seq_dim))
    
    def select(self):
        if self.prompt:
            return
        
        if not (self.use_compression and self.seq_length > self.total_budget):
            return
        
        # Select tokens to keep (common logic)
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
        
        self.prompt = True