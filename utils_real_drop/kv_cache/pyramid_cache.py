import torch
import math
from . import KVCache

class PyramidCache(KVCache):
    """Pyramid cache implementation"""
    
    def __init__(self, num_key_value_heads: int, seq_dim: int = 2):
        super().__init__(num_key_value_heads, seq_dim)
        self.prompt = False
    
    def init_cache(self, compression_config, layer_idx):
        """Initialize Pyramid cache settings"""
        self.seq_length = 0
        default_budget = compression_config.total_budget
        beta = compression_config.beta
        self.recent_budget = compression_config.recent_budget
        
        min_budget = (default_budget - self.recent_budget) // beta
        max_budget = (default_budget - self.recent_budget) * 2 - min_budget
        
        step = (max_budget - min_budget) // 31
        self.total_budget = max_budget - step * layer_idx
        
        self.select_budget = self.total_budget - self.recent_budget
        self.score = None
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
    
    def select(self):
        if self.prompt:
            return
        
        if self.seq_length <= self.total_budget:
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
        
        acc_score = torch.zeros((batch_size, num_heads, seq_len_k), dtype=query.dtype, device=query.device)
        
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
            
            acc_score.mul_(torch.exp(running_max - new_max).squeeze(-1))
            acc_score[:,:,k_start:k_end].add_(self.flash_prepare_scores(scores))
            
            # Update output
            output.mul_(torch.exp(running_max - new_max)).add_(torch.matmul(scores, v_chunk))
            
            # Update running max
            running_max.copy_(new_max)
        
        # GQA Aware Accumulation
        acc_score.div_(running_sum.squeeze(-1))
        acc_score = acc_score.view(acc_score.shape[0], self.num_key_value_heads, -1, *acc_score.shape[2:]).sum(dim=2)
        if self.score is None:
            self.score = acc_score
        else:
            acc_score[:,:,:-1] += self.score
            self.score = acc_score
        
        # Final normalization
        output = output / running_sum
        
        return output