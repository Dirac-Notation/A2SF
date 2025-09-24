import torch
import math
from . import KVCache

class SnapCache(KVCache):
    """Snap cache implementation"""
    
    def __init__(self, num_key_value_heads: int, seq_dim: int = 2):
        super().__init__(num_key_value_heads, seq_dim)
        self.prompt = False
    
    def init_cache(self, compression_config, layer_idx):
        """Initialize Snap cache settings"""
        self.seq_length = 0
        self.total_budget = compression_config.total_budget
        self.recent_budget = int(self.total_budget * 0.125)
        self.select_budget = self.total_budget - self.recent_budget
        self.observation_window = compression_config.observation_window
        self.prompt = False

    def select(self, scores):
        if self.seq_length <= self.total_budget:
            return
        
        selected_indices = scores[:,:,:-self.recent_budget].topk(self.select_budget, dim=-1).indices.sort().values
        
        selected_indices = selected_indices.unsqueeze(-1).expand(-1,-1,-1,self.key_data.size(-1))
        
        self.key_data = torch.cat((
            self.key_data.gather(self.seq_dim, selected_indices),
            self.key_data[:,:,-self.recent_budget:,:]
        ), dim=self.seq_dim)
        
        self.value_data = torch.cat((
            self.value_data.gather(self.seq_dim, selected_indices),
            self.value_data[:,:,-self.recent_budget:,:]
        ), dim=self.seq_dim)
    
    def flash_prepare_scores(self, attn_scores):
        return attn_scores[:,:,-self.observation_window:].sum(self.seq_dim)

    def prompt_flash_attention(self, query, key, value, attn_mask, head_dim, block_size=1024):
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
        
        acc_score = torch.zeros((batch_size, num_heads, seq_len_k), dtype=query.dtype, device=query.device)
        observation_point = self.seq_length - self.observation_window
        
        # Process key-value pairs in chunks
        for q_start in range(0, seq_len_q, block_size):
            q_end = min(q_start + block_size, seq_len_k)
            
            # Extract current chunk of key and value
            q_chunk = query[:, :, q_start:q_end, :]  # [batch_size, num_heads, chunk_size, head_dim]
            
            # Compute attention scores for this chunk
            scores = torch.matmul(q_chunk, key.transpose(2, 3)).mul_(sm_scale)  # [batch_size, num_heads, seq_len_q, chunk_size]
            
            # Apply attention mask if provided (includes causal masking)
            if attn_mask is not None:
                scores.add_(attn_mask[:, :, q_start:q_end, :])
            
            scores = torch.softmax(scores, dim=-1)
            
            output[:,:,q_start:q_end] = torch.matmul(scores, value)

            if q_start <= observation_point and q_end > observation_point:
                acc_score.add_(self.flash_prepare_scores(scores[:,:,observation_point:q_end:]))
            elif q_start >= observation_point:
                acc_score.add_(self.flash_prepare_scores(scores))
        
        # GQA Aware Accumulation
        acc_score = acc_score.view(acc_score.shape[0], self.num_key_value_heads, -1, *acc_score.shape[2:]).sum(dim=2)
        
        self.select(acc_score)
        
        return output

    def flash_attention(self, query, key, value, attn_mask, head_dim, block_size=1024):
        if not self.prompt:
            self.prompt = True
            return self.prompt_flash_attention(query, key, value, attn_mask, head_dim, block_size)
        else:
            return super().flash_attention(query, key, value, attn_mask, head_dim, block_size)