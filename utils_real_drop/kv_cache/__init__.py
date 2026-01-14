import torch
import math
from typing import Optional, Tuple, Dict, Any, List
from transformers.cache_utils import Cache

class KVCache(Cache):
    """
    Unified KV Cache that manages all layers and supports compression algorithms.
    This completely replaces transformers' Cache structure.
    """
    
    def __init__(self, layer_caches: List[Cache]):
        """
        Initialize with a list of per-layer cache objects.
        
        Args:
            layer_caches: List of cache objects, one for each layer (H2OCache, A2SFCache, etc.)
        """
        super().__init__()
        self.layer_caches = layer_caches
        self._seen_tokens = 0
        
        # Initialize key_cache and value_cache lists for transformers compatibility
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
        # Initialize with empty lists for each layer
        for _ in range(len(layer_caches)):
            self.key_cache.append([])
            self.value_cache.append([])
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`."""
        if layer_idx >= len(self.layer_caches):
            raise ValueError(f"Layer index {layer_idx} out of range. Total layers: {len(self.layer_caches)}")
        
        # Update the number of seen tokens (only for first layer)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        
        # Update the per-layer cache
        layer_cache = self.layer_caches[layer_idx]
        updated_key, updated_value = layer_cache.update(key_states, value_states, layer_idx, cache_kwargs)
        
        # Update the transformers-compatible cache lists
        if len(self.key_cache) <= layer_idx:
            # Extend lists if needed
            for _ in range(len(self.key_cache), layer_idx + 1):
                self.key_cache.append([])
                self.value_cache.append([])
        
        # Store the updated tensors
        self.key_cache[layer_idx] = updated_key
        self.value_cache[layer_idx] = updated_value
        
        return updated_key, updated_value
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states for the given layer."""
        if layer_idx is None:
            layer_idx = 0
        if layer_idx >= len(self.layer_caches):
            return 0
        return self.layer_caches[layer_idx].get_seq_length()
    
    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. None means no limit."""
        return None
    
    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Support for backwards-compatible `past_key_value` indexing."""
        if layer_idx < len(self.layer_caches):
            layer_cache = self.layer_caches[layer_idx]
            if hasattr(layer_cache, 'key_data') and hasattr(layer_cache, 'value_data'):
                return (layer_cache.key_data, layer_cache.value_data)
            elif len(self.key_cache) > layer_idx and len(self.key_cache[layer_idx]) > 0:
                return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        raise KeyError(f"Cache only has {len(self.layer_caches)} layers, attempted to access layer with index {layer_idx}")
    
    def __iter__(self):
        """Support for backwards-compatible `past_key_value` iteration."""
        for layer_idx in range(len(self.layer_caches)):
            yield self[layer_idx]
    
    def __len__(self):
        """Support for backwards-compatible `past_key_value` length."""
        return len(self.layer_caches)
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search."""
        for layer_idx in range(len(self.layer_caches)):
            layer_cache = self.layer_caches[layer_idx]
            if hasattr(layer_cache, 'reorder_cache'):
                layer_cache.reorder_cache(beam_idx)
            
            # Also update the transformers-compatible cache lists
            if len(self.key_cache) > layer_idx and isinstance(self.key_cache[layer_idx], torch.Tensor):
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            if len(self.value_cache) > layer_idx and isinstance(self.value_cache[layer_idx], torch.Tensor):
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))
    
    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], ...]:
        """Converts the cache to the legacy format (tuple of tuples)."""
        legacy_cache = ()
        for layer_idx in range(len(self.layer_caches)):
            layer_cache = self.layer_caches[layer_idx]
            if hasattr(layer_cache, 'key_data') and hasattr(layer_cache, 'value_data'):
                if layer_cache.key_data is not None and layer_cache.value_data is not None:
                    legacy_cache += ((layer_cache.key_data, layer_cache.value_data),)
                else:
                    legacy_cache += ((torch.empty(0), torch.empty(0)),)
            elif len(self.key_cache) > layer_idx and len(self.key_cache[layer_idx]) > 0:
                legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
            else:
                legacy_cache += ((torch.empty(0), torch.empty(0)),)
        return legacy_cache
    
    def __getattr__(self, name):
        """
        Delegate attribute access to layer caches for methods like flash_attention.
        This allows KVCache to be used as if it were a single cache object
        when the method is called from LlamaAttention.
        """
        if len(self.layer_caches) > 0:
            return getattr(self.layer_caches[0], name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# Base class for per-layer cache implementations (compression algorithms)
class LayerCache(Cache):
    """Base class for per-layer cache implementations with compression support."""
    
    def __init__(self, num_key_value_heads: int, seq_dim: int = 2):
        super().__init__()
        self.seq_dim = seq_dim
        self.seq_length = 0
        self.key_data = None
        self.value_data = None
        self.num_key_value_heads = num_key_value_heads
        self.device = None
        self.total_budget = 0
        self.recent_budget = 0
        self.select_budget = 0
        self.key_cache: list = []
        self.value_cache: list = []
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states"""
        return self.seq_length
    
    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. None means no limit."""
        return None
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates the cache with the new `key_states` and `value_states` for the layer."""
        return self.cat(key_states, value_states)
    
    def cat(self, key_tensor: torch.Tensor, value_tensor: torch.Tensor):
        """Concatenate new key-value pairs to existing cache"""
        self.device = key_tensor.device
        L = key_tensor.size(self.seq_dim)
        
        if self.seq_length == 0:
            self.key_data = key_tensor
            self.value_data = value_tensor
            self.key_cache = [key_tensor]
            self.value_cache = [value_tensor]
        else:
            self.key_data = torch.cat((self.key_data, key_tensor), dim=self.seq_dim)
            self.value_data = torch.cat((self.value_data, value_tensor), dim=self.seq_dim)
            if len(self.key_cache) > 0:
                self.key_cache[0] = self.key_data
                self.value_cache[0] = self.value_data
            else:
                self.key_cache = [self.key_data]
                self.value_cache = [self.value_data]

        self.seq_length += L
        return self.key_data, self.value_data
    
    def init_cache(self, compression_config, layer_idx):
        """Initialize cache with compression settings - to be overridden by subclasses"""
        pass

    def select(self, scores):
        """Select tokens to keep based on scores - to be overridden by subclasses"""
        pass
    
    def flash_prepare_scores(self, attn_scores, q_start=None, q_end=None):
        """Prepare scores for flash attention - to be overridden by subclasses"""
        pass
    
    def flash_attention(self, query, key, value, attn_mask, head_dim, block_size=1024):
        """
        Flash Attention implementation with chunked processing.
        Supports score accumulation for compression algorithms.
        """
        batch_size, num_heads, seq_len_q, _ = query.shape
        _, _, seq_len_k, _ = key.shape
        
        sm_scale = torch.tensor(1.0 / math.sqrt(head_dim), device=query.device, dtype=query.dtype)
        
        output = torch.zeros_like(query)
        running_max = torch.full((batch_size, num_heads, seq_len_q, 1), float('-inf'), device=query.device, dtype=query.dtype)
        running_sum = torch.zeros((batch_size, num_heads, seq_len_q, 1), device=query.device, dtype=query.dtype)
        
        # Process key-value pairs in chunks
        for k_start in range(0, seq_len_k, block_size):
            k_end = min(k_start + block_size, seq_len_k)
            
            k_chunk = key[:, :, k_start:k_end, :]
            v_chunk = value[:, :, k_start:k_end, :]
            
            scores = torch.matmul(query, k_chunk.transpose(-2, -1)).mul_(sm_scale)
            
            if attn_mask is not None:
                scores.add_(attn_mask[:, :, :, k_start:k_end])
            
            new_max = torch.maximum(running_max, torch.max(scores, dim=-1, keepdim=True)[0])
            scores.sub_(new_max).exp_()
            running_sum.mul_(torch.exp(running_max - new_max)).add_(torch.sum(scores, dim=-1, keepdim=True))
            output.mul_(torch.exp(running_max - new_max)).add_(torch.matmul(scores, v_chunk))
            running_max.copy_(new_max)
        
        output = output / running_sum
        return output
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search"""
        if self.key_data is not None:
            self.key_data = self.key_data.index_select(0, beam_idx)
        if self.value_data is not None:
            self.value_data = self.value_data.index_select(0, beam_idx)
        for layer_idx in range(len(self.key_cache)):
            if len(self.key_cache) > layer_idx and self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            if len(self.value_cache) > layer_idx and self.value_cache[layer_idx] is not None:
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))


__all__ = ["KVCache", "LayerCache"]
