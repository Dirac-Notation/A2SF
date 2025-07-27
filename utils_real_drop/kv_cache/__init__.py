import torch
from abc import ABC, abstractmethod

class BaseCache(ABC):
    """Abstract base class for cache implementations"""
    
    def __init__(self, num_key_value_heads: int, seq_dim: int = 2):
        self.seq_dim = seq_dim
        self.seq_length = 0
        self.key_data = None
        self.value_data = None
        self.num_key_value_heads = num_key_value_heads
        self.device = None
    
    def len(self):
        return self.seq_length
    
    def cat(self, key_tensor: torch.Tensor, value_tensor: torch.Tensor):
        """Concatenate new key-value pairs to existing cache"""
        self.device = key_tensor.device
        L = key_tensor.size(self.seq_dim)
        
        if self.seq_length == 0:
            self.key_data = key_tensor
            self.value_data = value_tensor
        else:
            self.key_data = torch.cat((self.key_data, key_tensor), dim=self.seq_dim)
            self.value_data = torch.cat((self.value_data, value_tensor), dim=self.seq_dim)

        self.seq_length += L
        return self.key_data, self.value_data
    
    @abstractmethod
    def init_cache(self, compression_config, layer_idx):
        """Initialize cache with compression settings"""
        pass
    
    @abstractmethod
    def update(self, attn_scores):
        """Update cache based on attention scores"""
        pass

class KVCache(BaseCache):
    """Main cache class that dispatches to specific implementations"""
    
    def __init__(self, num_key_value_heads: int, seq_dim: int = 2):
        super().__init__(num_key_value_heads, seq_dim)
        self.cache_impl = None
        self.cache_type = None
    
    def init_cache(self, compression_config, layer_idx):
        """Initialize cache with compression settings and create appropriate implementation"""
        self.seq_length = 0
        
        # Determine cache type based on config
        if compression_config.use_compression:
            if compression_config.compression_method == "streamingLLM":
                self.cache_type = 'streaming'
                from .streaming_cache import StreamingCache
                self.cache_impl = StreamingCache(self.num_key_value_heads, self.seq_dim)
            elif compression_config.compression_method == "average":
                self.cache_type = 'average'
                from .average_cache import AverageCache
                self.cache_impl = AverageCache(self.num_key_value_heads, self.seq_dim)
            elif compression_config.compression_method == "h2o" or compression_config.compression_method == "a2sf":
                # Check if it's H2O (forgetting_factor == 1) or A2SF
                if compression_config.forgetting_factors and compression_config.forgetting_factors[layer_idx] == 1:
                    self.cache_type = 'h2o'
                    from .h2o_cache import H2OCache
                    self.cache_impl = H2OCache(self.num_key_value_heads, self.seq_dim)
                else:
                    self.cache_type = 'a2sf'
                    from .a2sf_cache import A2SFCache
                    self.cache_impl = A2SFCache(self.num_key_value_heads, self.seq_dim)
            else:
                raise ValueError(f"Unsupported compression method: {compression_config.compression_method}")
        else:
            self.cache_type = 'none'
            self.cache_impl = None
        
        # Initialize the specific implementation
        if self.cache_impl:
            self.cache_impl.init_cache(compression_config, layer_idx)
    
    def update(self, attn_scores):
        """Update cache based on attention scores"""
        if self.cache_impl:
            self.cache_impl.update(attn_scores)
    
    def set_forget(self, forget, exponents):
        """Set forgetting parameters for A2SF/H2O"""
        if self.cache_impl and hasattr(self.cache_impl, 'set_forget'):
            self.cache_impl.set_forget(forget, exponents)
    
            # Delegate other methods to the implementation
        def __getattr__(self, name):
            if self.cache_impl and hasattr(self.cache_impl, name):
                return getattr(self.cache_impl, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

# Export the main KVCache class
__all__ = ["KVCache", "BaseCache"] 