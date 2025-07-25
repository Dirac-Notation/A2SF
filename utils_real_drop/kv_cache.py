import torch
import math

class KVCache():
    def __init__(self, num_key_value_heads: int, seq_dim: int = 2):
        self.seq_dim = seq_dim
        self.seq_length = 0
        self.key_data = None
        self.value_data = None
        self.score = None
        self.num_key_value_heads = num_key_value_heads
        self.cache_type = None  # 'a2sf' or 'streaming'

    def init_cache(self, compression_config, layer_idx):
        """Initialize cache with compression settings"""
        self.seq_length = 0
        
        # Determine cache type based on config
        if compression_config.use_compression:
            if compression_config.compression_method == "streamingLLM":
                self.cache_type = 'streaming'
                self._init_streaming_cache(compression_config, layer_idx)
            else:
                self.cache_type = 'a2sf'
                self._init_a2sf_cache(compression_config, layer_idx)
        else:
            self.cache_type = 'none'
            self.use_compression = False

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
        # Cast to float32 for numerical stability
        q = query.to(torch.float32)
        k = key.to(torch.float32)
        v = value.to(torch.float32)
        
        batch_size, num_heads, seq_len_q, _ = q.shape
        _, _, seq_len_k, _ = k.shape
        
        # Scale factor
        sm_scale = 1.0 / math.sqrt(head_dim)
        
        # Initialize output and running statistics
        output = torch.zeros_like(q)
        running_max = torch.full((batch_size, num_heads, seq_len_q, 1), float('-inf'), device=q.device, dtype=torch.float32)
        running_sum = torch.zeros((batch_size, num_heads, seq_len_q, 1), device=q.device, dtype=torch.float32)
        
        # Process key-value pairs in chunks
        for k_start in range(0, seq_len_k, block_size):
            k_end = min(k_start + block_size, seq_len_k)
            
            # Extract current chunk of key and value
            k_chunk = k[:, :, k_start:k_end, :]  # [batch_size, num_heads, chunk_size, head_dim]
            v_chunk = v[:, :, k_start:k_end, :]  # [batch_size, num_heads, chunk_size, head_dim]
            
            # Compute attention scores for this chunk
            scores = torch.matmul(q, k_chunk.transpose(-2, -1))  # [batch_size, num_heads, seq_len_q, chunk_size]
            scores = sm_scale * scores
            
            # Apply causal mask for this chunk
            if k_start == 0:  # Only need to compute causal mask once
                row_idx = torch.arange(seq_len_q, device=q.device).unsqueeze(1)
                col_idx = torch.arange(seq_len_k, device=q.device).unsqueeze(0)
                col_offset = seq_len_q - seq_len_k
                causal_mask = row_idx >= (col_offset + col_idx)
            
            # Apply causal mask to current chunk
            chunk_causal_mask = causal_mask[:, k_start:k_end]  # [seq_len_q, chunk_size]
            scores = scores.masked_fill(
                torch.logical_not(chunk_causal_mask.unsqueeze(0).unsqueeze(0)), float('-inf')
            )
            
            # Apply attention mask if provided
            if attn_mask is not None:
                attn_mask_chunk = attn_mask[:, :, :, k_start:k_end]
                scores = scores + attn_mask_chunk
            
            # Update running statistics using flash attention algorithm
            chunk_max = torch.max(scores, dim=-1, keepdim=True)[0]  # [batch_size, num_heads, seq_len_q, 1]
            
            # Update running max
            new_max = torch.maximum(running_max, chunk_max)
            
            # Compute exponential terms
            exp_scores = torch.exp(scores - new_max)  # Subtract new_max for numerical stability
            exp_scores_old = torch.exp(running_max - new_max)  # Scale old running sum
            
            # Update running sum
            running_sum = exp_scores_old * running_sum + torch.sum(exp_scores, dim=-1, keepdim=True)
            
            # Update output
            output_chunk = torch.matmul(exp_scores, v_chunk)  # [batch_size, num_heads, seq_len_q, head_dim]
            output = output + exp_scores_old * output + output_chunk
            
            # Update running max
            running_max = new_max
        
        # Final normalization
        output = output / running_sum
        
        # Cast back to original dtype
        output = output.to(query.dtype)
        
        return output

    def _init_streaming_cache(self, compression_config, layer_idx):
        """Initialize streaming cache settings"""
        self.use_compression = True
        self.total_budget = max(round(compression_config.total_budget * compression_config.layerwise_ratio[layer_idx]), 2)
        self.recent_budget = round(self.total_budget * 0.5)
        self.streaming_budget = compression_config.streaming_budget if compression_config.streaming_budget is not None else 0
        self.recent_budget -= self.streaming_budget

    def _init_a2sf_cache(self, compression_config, layer_idx):
        """Initialize A2SF cache settings"""
        self.use_compression = True
        self.compression_method = compression_config.compression_method
        self.total_budget = max(round(compression_config.total_budget * compression_config.layerwise_ratio[layer_idx]), 2)
        self.recent_budget = round(self.total_budget * 0.5)
        self.select_budget = self.total_budget - self.recent_budget
        self.forgetting_factor = compression_config.forgetting_factors[layer_idx] if compression_config.forgetting_factors is not None else None
        
        # Determine cache type based on compression method
        if self.compression_method == "average":
            self.cache_type = 'average'
        elif self.forgetting_factor == 1:
            self.cache_type = 'h2o'
        else:
            self.cache_type = 'a2sf'
            
        self.score = None
        self.input_ids = None
        
        # Initialize average-specific variables
        if self.cache_type == 'average':
            self.cumulative_count = None  # 누적 횟수 벡터
            self.is_prefill = True  # prefill phase 여부

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

    def update(self, attn_scores):
        """Update cache based on attention scores"""
        if self.cache_type == 'streaming':
            self._update_streaming()
        elif self.cache_type == 'h2o':
            self._update_h2o(attn_scores)
        elif self.cache_type == 'a2sf':
            self._update_a2sf(attn_scores)
        elif self.cache_type == 'average':
            self._update_average(attn_scores)

    def _update_streaming(self):
        """Update cache using streaming method"""
        if not (self.seq_length > self.total_budget):
            return
            
        self.key_data = torch.cat((
            self.key_data[:,:,:self.streaming_budget,:],
            self.key_data[:,:,-self.recent_budget:,:]
        ), dim=self.seq_dim)
        
        self.value_data = torch.cat((
            self.value_data[:,:,:self.streaming_budget,:],
            self.value_data[:,:,-self.recent_budget:,:]
        ), dim=self.seq_dim)

    def _update_h2o(self, attn_scores):
        """Update cache using H2O method (forgetting_factor == 1)"""
        if not (self.use_compression and self.seq_length > self.total_budget):
            return
        
        attn_scores_shape = attn_scores.shape
        
        attn_scores = attn_scores.view(attn_scores_shape[0], self.num_key_value_heads, -1, *attn_scores_shape[2:]).sum(dim=2)
        
        # For H2O, simply sum attention scores without forgetting factor
        current_score = attn_scores.sum(self.seq_dim)
        if self.score is not None:
            current_score[:,:,:-1] += self.score
        self.score = current_score
        
        # Select tokens to keep
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

    def _update_a2sf(self, attn_scores):
        """Update cache using A2SF method (forgetting_factor != 1)"""
        if not (self.use_compression and self.seq_length > self.total_budget):
            return
        
        attn_scores_shape = attn_scores.shape
        
        attn_scores = attn_scores.view(attn_scores_shape[0], self.num_key_value_heads, -1, *attn_scores_shape[2:]).sum(dim=2)
        
        seq_len = attn_scores.size(2)
        
        # Calculate weighted attention scores with forgetting factor
        if self.exponents is not None:
            forgetting = (self.forgetting_factor ** self.exponents.to(attn_scores.device)).view(1, 1, seq_len, 1)
            current_score = (forgetting * attn_scores).sum(dim=self.seq_dim)
            self.score = current_score
        else:
            current_score = attn_scores.sum(self.seq_dim)
            if self.score is not None:
                current_score[:,:,:-1] += self.score
            self.score = current_score
            if self.forget:
                self.score *= self.forgetting_factor
        
        # Select tokens to keep
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

    def _update_average(self, attn_scores):
        """Update cache using average-based selection method"""
        if not (self.use_compression and self.seq_length > self.total_budget):
            return
        
        attn_scores_shape = attn_scores.shape
        
        attn_scores = attn_scores.view(attn_scores_shape[0], self.num_key_value_heads, -1, *attn_scores_shape[2:]).sum(dim=2)
        
        # Calculate current score
        current_score = attn_scores.sum(self.seq_dim)
        current_count = (attn_scores > 0).sum(self.seq_dim).to(current_score.dtype)
        
        if self.score is None:
            self.score = current_score
            self.cumulative_count = current_count
        else:
            current_score[:,:,:-1] += self.score
            current_count[:,:,:-1] += self.cumulative_count
        self.score = current_score
        self.cumulative_count = current_count
        
        self.score = self.score.to(self.device)
        self.cumulative_count = self.cumulative_count.to(self.device)
        
        # Calculate average scores
        average_score = self.score / self.cumulative_count
        
        # Select tokens to keep based on average scores
        selected_indices = average_score[:,:,:-self.recent_budget].topk(self.select_budget, dim=-1).indices.sort().values
        
        # Update scores and counts
        self.score = torch.cat((
            self.score.gather(self.seq_dim, selected_indices),
            self.score[:,:,-self.recent_budget:]
        ), dim=self.seq_dim)
        
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
    
    def set_forget(self, forget, exponents):
        """Set forgetting parameters for A2SF/H2O"""
        if self.cache_type in ['a2sf', 'h2o']:
            self.forget = forget
            self.exponents = exponents