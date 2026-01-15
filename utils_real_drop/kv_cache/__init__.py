import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, List
from transformers.cache_utils import Cache

class KVCache(Cache):
    """
    Unified KV Cache that manages all layers and supports compression algorithms.
    This completely replaces transformers' Cache structure and is fully compatible with DynamicCache.
    """
    
    def __init__(self, layer_caches: Optional[List[Cache]] = None):
        """
        Initialize with a list of per-layer cache objects.
        
        Args:
            layer_caches: List of cache objects, one for each layer (H2OCache, A2SFCache, etc.)
                         If None, creates an empty cache that can be populated later.
        """
        super().__init__()
        self.layer_caches = layer_caches if layer_caches is not None else []
        self._seen_tokens = 0
        
        # Initialize key_cache and value_cache lists for transformers compatibility
        # These must be lists of tensors (one per layer), matching DynamicCache structure
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
        # Initialize with empty tensors for each layer (DynamicCache uses empty lists initially)
        for _ in range(len(self.layer_caches)):
            self.key_cache.append(torch.empty(0))
            self.value_cache.append(torch.empty(0))
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        This matches the DynamicCache.update() interface exactly.
        """
        # Update the number of seen tokens (only for first layer)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        
        # If we have layer caches (compression enabled), use them
        if len(self.layer_caches) > layer_idx:
            layer_cache = self.layer_caches[layer_idx]
            updated_key, updated_value = layer_cache.update(key_states, value_states, layer_idx, cache_kwargs)
        else:
            # Fallback to simple concatenation (like DynamicCache)
            if len(self.key_cache) <= layer_idx:
                # Extend lists if needed (for skipped layers)
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.empty(0))
                    self.value_cache.append(torch.empty(0))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif len(self.key_cache[layer_idx]) == 0 or self.key_cache[layer_idx].numel() == 0:
                # First time for this layer
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                # Concatenate with existing cache
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            updated_key = self.key_cache[layer_idx]
            updated_value = self.value_cache[layer_idx]
        
        # Always update the transformers-compatible cache lists
        if len(self.key_cache) <= layer_idx:
            # Extend lists if needed
            for _ in range(len(self.key_cache), layer_idx + 1):
                self.key_cache.append(torch.empty(0))
                self.value_cache.append(torch.empty(0))
        
        # Store the updated tensors (for compatibility with DynamicCache)
        self.key_cache[layer_idx] = updated_key
        self.value_cache[layer_idx] = updated_value
        
        return updated_key, updated_value
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states for the given layer."""
        if layer_idx is None:
            layer_idx = 0
        
        # Check if we have layer caches (compression enabled)
        if len(self.layer_caches) > layer_idx:
            return self.layer_caches[layer_idx].get_seq_length()
        
        # Fallback to checking key_cache directly (like DynamicCache)
        is_empty_layer = (
            len(self.key_cache) == 0
            or len(self.key_cache) <= layer_idx
            or self.key_cache[layer_idx].numel() == 0
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length
    
    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. None means no limit."""
        return None
    
    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing.
        Returns (key_cache[layer_idx], value_cache[layer_idx]) like DynamicCache.
        """
        if layer_idx < len(self.key_cache):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        raise KeyError(f"Cache only has {len(self.key_cache)} layers, attempted to access layer with index {layer_idx}")
    
    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration.
        Yields (key, value) tuples for each layer like DynamicCache.
        """
        for layer_idx in range(len(self.key_cache)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])
    
    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length.
        Returns the number of layers, matching DynamicCache behavior.
        """
        return len(self.key_cache)
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search."""
        # Reorder layer caches if they exist
        for layer_idx in range(len(self.layer_caches)):
            layer_cache = self.layer_caches[layer_idx]
            if hasattr(layer_cache, 'reorder_cache'):
                layer_cache.reorder_cache(beam_idx)
        
        # Reorder transformers-compatible cache lists
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel() > 0:
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            if self.value_cache[layer_idx].numel() > 0:
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))
    
    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the cache to the legacy format (tuple of tuples)."""
        legacy_key_cache = ()
        legacy_value_cache = ()
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel() > 0:
                legacy_key_cache += (self.key_cache[layer_idx],)
                legacy_value_cache += (self.value_cache[layer_idx],)
            else:
                legacy_key_cache += (torch.empty(0),)
                legacy_value_cache += (torch.empty(0),)
        return (legacy_key_cache, legacy_value_cache)
    
    @property
    def seen_tokens(self):
        """Returns the number of tokens seen by the cache."""
        return self._seen_tokens
    
    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Returns the usable length of the cache for the given layer."""
        return self.get_seq_length(layer_idx)


# Base class for per-layer cache implementations (compression algorithms)
class LayerCache(Cache):
    """Base class for per-layer cache implementations with compression support."""
    
    def __init__(self, num_key_value_heads: int, device: torch.device, seq_dim: int = 2):
        super().__init__()
        self.seq_dim = seq_dim
        self.seq_length = 0
        self.key_data = None
        self.value_data = None
        self.num_key_value_heads = num_key_value_heads
        self.device = device
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
        """
        Updates the cache with the new `key_states` and `value_states` for the layer.
        This matches the Cache.update() interface from transformers.
        """
        return self.cat(key_states, value_states)
    
    def cat(self, key_tensor: torch.Tensor, value_tensor: torch.Tensor):
        """Concatenate new key-value pairs to existing cache"""
        self.device = key_tensor.device
        L = key_tensor.size(self.seq_dim)
        
        if self.seq_length == 0:
            self.key_data = key_tensor
            self.value_data = value_tensor
            # Initialize cache lists (for compatibility with KVCache)
            self.key_cache = [key_tensor]
            self.value_cache = [value_tensor]
        else:
            self.key_data = torch.cat((self.key_data, key_tensor), dim=self.seq_dim)
            self.value_data = torch.cat((self.value_data, value_tensor), dim=self.seq_dim)
            # Update cache lists (for compatibility with KVCache)
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
        Matches original LlamaAttention computation exactly.
        Supports score accumulation for compression algorithms.
        """
        batch_size, num_heads, seq_len_q, _ = query.shape
        _, _, seq_len_k, _ = key.shape
        
        # Use float32 for softmax like original LlamaAttention
        sm_scale = 1.0 / math.sqrt(head_dim)
        
        # Keep output in float32 for numerical accuracy, convert at the end
        output = torch.zeros((batch_size, num_heads, seq_len_q, query.shape[-1]), device=query.device, dtype=torch.float32)
        running_max = torch.full((batch_size, num_heads, seq_len_q, 1), float('-inf'), device=query.device, dtype=torch.float32)
        running_sum = torch.zeros((batch_size, num_heads, seq_len_q, 1), device=query.device, dtype=torch.float32)
        
        # Process key-value pairs in chunks
        for k_start in range(0, seq_len_k, block_size):
            k_end = min(k_start + block_size, seq_len_k)
            
            k_chunk = key[:, :, k_start:k_end, :]
            v_chunk = value[:, :, k_start:k_end, :]
            
            # Compute scores in float32 for numerical stability (like original)
            scores = torch.matmul(query.to(torch.float32), k_chunk.transpose(-2, -1).to(torch.float32)) * sm_scale
            
            # Apply attention mask - use the full mask slice for this chunk
            if attn_mask is not None:
                # attn_mask is already sliced to key_states.shape[-2] in LlamaAttention.forward
                # So we can directly use the corresponding slice
                mask_slice = attn_mask[:, :, :, k_start:k_end]
                scores = scores + mask_slice.to(torch.float32)
            
            # Online softmax in float32
            new_max = torch.maximum(running_max, torch.max(scores, dim=-1, keepdim=True)[0])
            scores = scores - new_max
            scores_exp = torch.exp(scores)
            exp_scale = torch.exp(running_max - new_max)
            running_sum = running_sum * exp_scale + torch.sum(scores_exp, dim=-1, keepdim=True)
            # Keep output in float32 throughout for numerical accuracy
            output = output * exp_scale + torch.matmul(scores_exp, v_chunk.to(torch.float32))
            running_max = new_max
        
        # Normalize and convert back to original dtype
        output = (output / running_sum).to(query.dtype)
        
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
