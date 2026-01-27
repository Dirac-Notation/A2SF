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