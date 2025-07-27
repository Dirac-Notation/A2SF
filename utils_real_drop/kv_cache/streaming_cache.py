import torch
from . import BaseCache

class StreamingCache(BaseCache):
    """Streaming LLM cache implementation"""
    
    def __init__(self, num_key_value_heads: int, seq_dim: int = 2):
        super().__init__(num_key_value_heads, seq_dim)
        self.use_compression = False
        self.total_budget = 0
        self.recent_budget = 0
        self.streaming_budget = 0
    
    def init_cache(self, compression_config, layer_idx):
        """Initialize streaming cache settings"""
        self.use_compression = True
        self.total_budget = max(round(compression_config.total_budget * compression_config.layerwise_ratio[layer_idx]), 2)
        self.recent_budget = round(self.total_budget * 0.5)
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