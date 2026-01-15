import torch
import math
from . import LayerCache

class FullCache(LayerCache):
    """Full cache implementation - no compression, stores all KV cache"""
    
    def __init__(self, num_key_value_heads: int, device: torch.device, seq_dim: int = 2):
        super().__init__(num_key_value_heads, seq_dim)
        self.device = device
    
    def init_cache(self, compression_config, layer_idx):
        """Initialize Full cache - no compression settings needed"""
        self.seq_length = 0
        # No budget limits for full cache
        self.total_budget = float('inf')
        self.recent_budget = 0
        self.select_budget = 0
    
    def select(self, scores):
        """Full cache doesn't select - keeps everything"""
        pass
    
    def flash_prepare_scores(self, attn_scores, q_start=None, q_end=None):
        """Full cache doesn't need score preparation"""
        pass

