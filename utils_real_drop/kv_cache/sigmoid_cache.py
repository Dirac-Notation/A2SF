import torch
import math
from . import LayerCache

class SigmoidCache(LayerCache):
    """Sigmoid cache implementation"""
    
    def __init__(self, num_key_value_heads: int, device: torch.device, seq_dim: int = 2):
        super().__init__(num_key_value_heads, device, seq_dim)
        self.a = None
        self.b = None
        self.exponents = None
        self.window = None
        
    def init_cache(self, compression_config, layer_idx):
        """Initialize Sigmoid cache settings"""
        self.total_budget = compression_config.total_budget
        self.recent_budget = int(self.total_budget * 0.125)
        self.select_budget = self.total_budget - self.recent_budget
        self.a = compression_config.a
        self.b = compression_config.b
        self.window = None
        
    def select(self, acc_scores):
        """
        Compresses key_data and value_data based on accumulated scores.
        Strategy: Keep 'Recent' tokens + Top-K 'History' tokens.
        """
        if self.key_data.size(self.seq_dim) <= self.total_budget:
            return

        # Update sequence length
        self.seq_length = self.key_data.shape[self.seq_dim]

        # Split into History and Recent
        seq_len = self.key_data.size(self.seq_dim)
        num_history = seq_len - self.recent_budget
        
        if num_history <= 0:
            return

        # acc_scores shape: (Batch, Num_KV_Heads, Seq_Len)
        history_scores = acc_scores[:, :, :num_history]
        
        # Select top-k indices from history
        k = min(self.select_budget, num_history)
        selected_indices = torch.topk(history_scores, k, dim=-1).indices # (B, H_kv, k)
        
        # Sort indices to maintain temporal order
        selected_indices = selected_indices.sort(dim=-1).values
        
        # Expand indices for gathering
        head_dim = self.key_data.size(-1)
        gather_indices = selected_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        
        # Gather selected history
        selected_keys = torch.gather(self.key_data[:, :, :num_history, :], self.seq_dim, gather_indices)
        selected_values = torch.gather(self.value_data[:, :, :num_history, :], self.seq_dim, gather_indices)
        
        # Get recent window
        recent_keys = self.key_data[:, :, -self.recent_budget:, :]
        recent_values = self.value_data[:, :, -self.recent_budget:, :]
        
        # Concatenate: [Selected History, Recent]
        self.key_data = torch.cat([selected_keys, recent_keys], dim=self.seq_dim)
        self.value_data = torch.cat([selected_values, recent_values], dim=self.seq_dim)

    def flash_prepare_scores(self, attn_scores, q_start, q_end):
        """
        Apply sigmoid window to attention scores and sum over query dimension.
        attn_scores: (B, H, Q_chunk, K)
        Returns: (B, H, K)
        """
        if self.window is None:
            # Fallback: no window
            return attn_scores.sum(dim=-2)
        
        # Apply sigmoid window
        # window shape: (B, 1, Q_total), we need (B, 1, Q_chunk)
        q_chunk_size = q_end - q_start
        window_chunk = self.window[:, :, q_start:q_end].to(attn_scores.device)  # (B, 1, Q_chunk)
        window_chunk = window_chunk.view(1, 1, q_chunk_size, 1)  # (1, 1, Q_chunk, 1)
        
        # Apply window and sum over query dimension
        weighted_scores = window_chunk * attn_scores  # (B, H, Q_chunk, K)
        return weighted_scores.sum(dim=-2)  # (B, H, K)

    def prompt_flash_attention(self, query, key, value, attn_mask, head_dim, block_size=1024):
        """
        Prompt Phase: Compute Output + Accumulate Attention Scores for Sigmoid.
        """
        batch_size, num_heads, seq_len_q, _ = query.shape
        _, _, seq_len_k, _ = key.shape
        
        # Note: key and value are already repeated via repeat_kv
        num_kv_heads = self.num_key_value_heads
        
        scale = 1.0 / math.sqrt(head_dim)
        output = torch.zeros_like(query)
        
        # Accumulator for scores: (Batch, Num_Heads, Seq_Len_K) -> Reduced later for GQA
        acc_score = torch.zeros((batch_size, num_heads, seq_len_k), dtype=torch.float32, device=query.device)
        
        # Compute sigmoid window if not already computed
        if self.window is None and self.exponents is not None:
            seq_len = query.size(2)
            self.window = 1/(1+torch.exp(-self.a * (self.exponents - (seq_len - self.b - 0.5))))
        
        # Chunking over Query
        for q_start in range(0, seq_len_q, block_size):
            q_end = min(q_start + block_size, seq_len_q)
            q_chunk = query[:, :, q_start:q_end, :]
            
            # Standard Q * K^T
            scores = torch.matmul(q_chunk.to(torch.float32), key.transpose(-2, -1).to(torch.float32)) * scale
            
            if attn_mask is not None:
                scores = scores + attn_mask[:, :, q_start:q_end, :].to(torch.float32)
            
            # Standard Softmax
            probs = torch.softmax(scores, dim=-1) # (B, H, Q_chunk, K_full)
            
            # Compute Output
            output[:, :, q_start:q_end, :] = torch.matmul(probs.to(query.dtype), value)
            
            # Apply sigmoid window and accumulate
            chunk_scores = self.flash_prepare_scores(probs, q_start, q_end)  # (B, H, K)
            acc_score += chunk_scores

        # Reduce scores for GQA (Grouped Query Attention)
        if num_heads != num_kv_heads:
            num_groups = num_heads // num_kv_heads
            acc_score = acc_score.view(batch_size, num_kv_heads, num_groups, seq_len_k).sum(dim=2)
            
        # Perform Compression
        self.select(acc_score)
        
        return output

    def flash_attention(self, query, key, value, attn_mask, head_dim, block_size=1024):
        """
        Dispatch based on phase (Prompt vs Generation).
        """
        seq_len_q = query.shape[-2]
        
        # Heuristic: If Query Length > 1, it's the prompt phase (prefill).
        if seq_len_q > 1:
            return self.prompt_flash_attention(query, key, value, attn_mask, head_dim, block_size)
        else:
            # Generation phase: Use standard implementation
            return super().flash_attention(query, key, value, attn_mask, head_dim, block_size)