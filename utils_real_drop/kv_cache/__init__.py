import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, List
from transformers.cache_utils import Cache

class KVCache(Cache):
    """
    Unified KV Cache that manages per-layer caches properly without data duplication.
    Acts as a proxy to specific LayerCache implementations.
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
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Delegates to the specific layer cache.
        """
        # Update seen tokens count (only once per step usually tracked by layer 0)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            
        # Delegate update to the specific layer cache
        if len(self.layer_caches) > layer_idx:
            return self.layer_caches[layer_idx].update(key_states, value_states, layer_idx, cache_kwargs)
        else:
            raise ValueError(f"Layer cache for layer {layer_idx} not found")
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states for the given layer."""
        if len(self.layer_caches) > layer_idx:
            return self.layer_caches[layer_idx].get_seq_length()
        return 0
    
    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. None means no limit."""
        return None
    
    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Support for legacy access (cache[i]). Returns views/copies from layer cache."""
        if layer_idx < len(self.layer_caches):
            return (self.layer_caches[layer_idx].key_data, self.layer_caches[layer_idx].value_data)
        raise KeyError(f"Layer index {layer_idx} out of bounds")

    def __iter__(self):
        for layer_cache in self.layer_caches:
            yield (layer_cache.key_data, layer_cache.value_data)

    def __len__(self):
        return len(self.layer_caches)
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search."""
        for layer_cache in self.layer_caches:
            if hasattr(layer_cache, 'reorder_cache'):
                layer_cache.reorder_cache(beam_idx)
    
# Base class for per-layer cache implementations (compression algorithms)
class LayerCache(Cache):
    """Base class for per-layer cache implementations with compression support."""
    
    def __init__(self, num_key_value_heads: int, device: torch.device, seq_dim: int = 2):
        super().__init__()
        self.seq_dim = seq_dim
        self.seq_length = 0
        self.num_key_value_heads = num_key_value_heads
        self.device = device
        
        # Initialize as empty tensors
        self.key_data: Optional[torch.Tensor] = None
        self.value_data: Optional[torch.Tensor] = None
    
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
        Standard update: concatenate new states to cache.
        """
        if self.key_data is None:
            self.key_data = key_states
            self.value_data = value_states
        else:
            self.key_data = torch.cat([self.key_data, key_states], dim=self.seq_dim)
            self.value_data = torch.cat([self.value_data, value_states], dim=self.seq_dim)
            
        self.seq_length = self.key_data.shape[self.seq_dim]
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


__all__ = ["KVCache", "LayerCache"]
