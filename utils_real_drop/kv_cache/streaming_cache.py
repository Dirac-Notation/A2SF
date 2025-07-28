import torch
from . import KVCache

class StreamingCache(KVCache):
    """Streaming LLM cache implementation"""
    
    def __init__(self, num_key_value_heads: int, seq_dim: int = 2):
        super().__init__(num_key_value_heads, seq_dim)
        self.streaming_budget = 0
    
    def init_cache(self, compression_config, layer_idx):
        """Initialize streaming cache settings"""
        super().init_cache(compression_config, layer_idx)
        self.streaming_budget = compression_config.streaming_budget if compression_config.streaming_budget is not None else 0
        self.recent_budget -= self.streaming_budget
    
    def update(self, attn_scores):
        """Update cache using streaming method"""
        if not (self.seq_length > self.total_budget):
            return
            
        self.key_data = torch.cat((
            self.key_data[:,:,:self.streaming_budget,:],
            self.key_data[:,:,-self.recent_budget:,:]
        ), dim=self.seq_dim)
        
        self.value_data = torch.cat((
            self.value_data[:,:,:self.streaming_budget,:],
            self.value_data[:,:,-self.recent_budget:,:]
        ), dim=self.seq_dim) 