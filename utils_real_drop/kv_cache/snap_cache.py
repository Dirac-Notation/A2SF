import torch
import math
import torch.nn.functional as F
from . import LayerCache

class SnapCache(LayerCache):
    """
    SnapKV Cache Implementation.
    Accurately accumulates attention scores during the prompt phase based on an 'observation window'
    and compresses the KV cache before generation starts.
    """
    
    def __init__(self, num_key_value_heads: int, device: torch.device, seq_dim: int = 2):
        super().__init__(num_key_value_heads, device, seq_dim)
        self.total_budget = 0
        self.recent_budget = 0
        self.select_budget = 0
        self.observation_window = 0
    
    def init_cache(self, compression_config, layer_idx):
        """Initialize Snap cache settings"""
        self.total_budget = compression_config.total_budget
        self.observation_window = compression_config.observation_window
        # SnapKV default split: 1/8 for recent, rest for significant history
        self.recent_budget = int(self.total_budget * 0.125) 
        self.select_budget = self.total_budget - self.recent_budget

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
        # We only prune from history. Recent tokens are always kept.
        seq_len = self.key_data.size(self.seq_dim)
        num_history = seq_len - self.recent_budget
        
        if num_history <= 0:
            return

        # acc_scores shape: (Batch, Num_KV_Heads, Seq_Len)
        # Consider only history scores
        history_scores = acc_scores[:, :, :num_history]
        
        # Select top-k indices from history
        k = min(self.select_budget, num_history)
        selected_indices = torch.topk(history_scores, k, dim=-1).indices # (B, H_kv, k)
        
        # Sort indices to maintain temporal order (optional but good for RoPE/Bias consistency)
        selected_indices = selected_indices.sort(dim=-1).values
        
        # Expand indices for gathering: (B, H_kv, k, Head_Dim)
        # Note: key_data is (B, H, Seq, Dim). We need to handle GQA if num_heads != num_kv_heads
        # But LayerCache.key_data is usually stored as (B, H_kv, Seq, Dim) in optimized backends, 
        # Here standard LlamaModel outputs (B, H_kv, Seq, Dim).
        
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
    
    def prompt_flash_attention(self, query, key, value, attn_mask, head_dim, block_size=1024):
        """
        Prompt Phase: Compute Output + Accumulate Attention Scores for SnapKV.
        """
        batch_size, num_heads, seq_len_q, _ = query.shape
        _, _, seq_len_k, _ = key.shape
        
        # Note: key and value are already repeated via repeat_kv, so key.shape[1] == num_heads
        # But we need the original num_key_value_heads for GQA reduction
        num_kv_heads = self.num_key_value_heads
        
        scale = 1.0 / math.sqrt(head_dim)
        output = torch.zeros_like(query)
        
        # Accumulator for scores: (Batch, Num_Heads, Seq_Len_K) -> Reduced later for GQA
        acc_score = torch.zeros((batch_size, num_heads, seq_len_k), dtype=torch.float32, device=query.device)
        
        # SnapKV looks at the LAST 'observation_window' tokens of the prompt as queries
        obs_start_idx = max(0, seq_len_q - self.observation_window)
        
        # Chunking over Query (to manage memory while getting full row scores)
        for q_start in range(0, seq_len_q, block_size):
            q_end = min(q_start + block_size, seq_len_q)
            q_chunk = query[:, :, q_start:q_end, :]
            
            # Standard Q * K^T
            # We compute full attention row for the chunk
            scores = torch.matmul(q_chunk.to(torch.float32), key.transpose(-2, -1).to(torch.float32)) * scale
            
            if attn_mask is not None:
                scores = scores + attn_mask[:, :, q_start:q_end, :].to(torch.float32)
            
            # Standard Softmax
            probs = torch.softmax(scores, dim=-1) # (B, H, Q_chunk, K_full)
            
            # Compute Output
            output[:, :, q_start:q_end, :] = torch.matmul(probs.to(query.dtype), value)
            
            # SnapKV Accumulation Logic
            # Only accumulate if the current query chunk overlaps with the observation window
            
            # Intersection of [q_start, q_end) and [obs_start_idx, seq_len_q)
            overlap_start = max(q_start, obs_start_idx)
            overlap_end = min(q_end, seq_len_q)
            
            if overlap_start < overlap_end:
                # Relative indices within the chunk
                rel_start = overlap_start - q_start
                rel_end = overlap_end - q_start
                
                # Extract relevant probabilities and sum over the Query dimension
                # probs: (B, H, Q_chunk, K) -> select rows -> (B, H, Q_overlap, K) -> sum(dim=-2) -> (B, H, K)
                chunk_scores = probs[:, :, rel_start:rel_end, :].sum(dim=-2)
                acc_score += chunk_scores

        # Reduce scores for GQA (Grouped Query Attention)
        # If num_heads (32) > num_kv_heads (8), sum scores of heads sharing the same KV
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
            # Generation phase: Use standard implementation (no accumulation needed)
            # The cache is already compressed at the end of prompt phase.
            return super().flash_attention(query, key, value, attn_mask, head_dim, block_size)