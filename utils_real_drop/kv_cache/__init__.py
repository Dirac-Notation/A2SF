import torch
import math
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
    
    def update(self, attn_scores):
        """Update cache based on attention scores"""
        # Let each implementation handle its own update logic
        pass
    
    def select(self):
        """Common selection logic for all cache implementations"""
        if not (self.use_compression and self.seq_length > self.total_budget):
            return
        
        # Select tokens to keep (common logic)
        selected_indices = self.score[:,:,:-self.recent_budget].topk(self.select_budget, dim=-1).indices.sort().values
        
        # Update scores
        self.score = torch.cat((
            self.score.gather(self.seq_dim, selected_indices),
            self.score[:,:,-self.recent_budget:]
        ), dim=self.seq_dim)
        
        if getattr(self, "cumulative_count", None) is not None:
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
    
    def prepare_scores(self, attn_scores):
        """Prepare scores for selection (implemented by each cache type)"""
        pass
    
    def flash_attention(self, query, key, value, attn_mask, head_dim, block_size=1024):
        """
        Real Flash Attention implementation with chunked processing
        Args:
            query: [batch_size, num_heads, seq_len_q, head_dim]
            key: [batch_size, num_heads, seq_len_k, head_dim]
            value: [batch_size, num_heads, seq_len_k, head_dim]
            attn_mask: [batch_size, 1, seq_len_q, seq_len_k] or None
            head_dim: dimension of each attention head
            block_size: size of chunks for memory-efficient processing
        Returns:
            output: [batch_size, num_heads, seq_len_q, head_dim]
        """
        # Use BFloat16 for all computations
        batch_size, num_heads, seq_len_q, _ = query.shape
        _, _, seq_len_k, _ = key.shape
        
        # Scale factor (convert to same dtype as query)
        sm_scale = torch.tensor(1.0 / math.sqrt(head_dim), device=query.device, dtype=query.dtype)
        
        # Initialize output and running statistics with BFloat16
        output = torch.zeros_like(query)
        running_max = torch.full((batch_size, num_heads, seq_len_q, 1), float('-inf'), device=query.device, dtype=query.dtype)
        running_sum = torch.zeros((batch_size, num_heads, seq_len_q, 1), device=query.device, dtype=query.dtype)
        
        self.score = torch.zeros((batch_size, num_heads, seq_len_k), dtype=query.dtype, device=query.device)
        
        # Process key-value pairs in chunks
        for k_start in range(0, seq_len_k, block_size):
            k_end = min(k_start + block_size, seq_len_k)
            
            # Extract current chunk of key and value
            k_chunk = key[:, :, k_start:k_end, :]  # [batch_size, num_heads, chunk_size, head_dim]
            v_chunk = value[:, :, k_start:k_end, :]  # [batch_size, num_heads, chunk_size, head_dim]
            
            # Compute attention scores for this chunk
            scores = torch.matmul(query, k_chunk.transpose(-2, -1)).mul_(sm_scale)  # [batch_size, num_heads, seq_len_q, chunk_size]
            
            # Apply attention mask if provided (includes causal masking)
            if attn_mask is not None:
                scores.add_(attn_mask[:, :, :, k_start:k_end])
            
            # Update running statistics using flash attention algorithm
            new_max = torch.maximum(running_max, torch.max(scores, dim=-1, keepdim=True)[0])
            
            # Compute exponential terms and update running sum
            scores.sub_(new_max).exp_()  # Subtract new_max and apply exp in-place
            running_sum.mul_(torch.exp(running_max - new_max)).add_(torch.sum(scores, dim=-1, keepdim=True))
            
            self.score.mul_(torch.exp(running_max - new_max).squeeze(-1))
            self.score[:,:,k_start:k_end].add_(self.flash_prepare_scores(scores))
            
            # Update output
            output.mul_(torch.exp(running_max - new_max)).add_(torch.matmul(scores, v_chunk))
            
            # Update running max
            running_max.copy_(new_max)
        
        self.score.div_(running_sum.squeeze(-1))
        # Final normalization
        output = output / running_sum
        
        return output

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
    
    def prepare_scores(self, attn_scores):
        """Delegate to cache implementation"""
        if self.cache_impl and hasattr(self.cache_impl, 'prepare_scores'):
            self.cache_impl.prepare_scores(attn_scores)
    
    def update(self, attn_scores):
        """Delegate to cache implementation"""
        if self.cache_impl:
            self.cache_impl.update(attn_scores)
    
    def select(self):
        """Delegate to cache implementation"""
        if self.cache_impl and hasattr(self.cache_impl, 'select'):
            self.cache_impl.select()
    
    def flash_attention(self, query, key, value, attn_mask, head_dim, block_size=1024):
        """Delegate to cache implementation"""
        if self.cache_impl and hasattr(self.cache_impl, 'flash_attention'):
            return self.cache_impl.flash_attention(query, key, value, attn_mask, head_dim, block_size)
        else:
            return super().flash_attention(query, key, value, attn_mask, head_dim, block_size)
    
    # Delegate other methods to the implementation
    def __getattr__(self, name):
        if self.cache_impl and hasattr(self.cache_impl, name):
            return getattr(self.cache_impl, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

# Export the main KVCache class
__all__ = ["KVCache", "BaseCache"] 