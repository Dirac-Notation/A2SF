import torch

class A2SFKVCache():
    def __init__(self, num_key_value_heads: int, seq_dim: int = 2):
        self.seq_dim = seq_dim
        self.seq_length = 0
        self.key_data = None
        self.value_data = None
        self.score = None
        self.use_compression = False
        self.num_key_value_heads = num_key_value_heads

    def init_cache(self, compression_config, layer_idx):
        """Initialize cache with compression settings"""
        self.seq_length = 0
        self.use_compression = compression_config.use_compression
        
        if self.use_compression:
            self.compression_method = compression_config.compression_method
            self.total_budget = compression_config.total_budget
            self.recent_budget = round(compression_config.compression_ratio[layer_idx] * self.total_budget)
            self.select_budget = self.total_budget - self.recent_budget
            self.forgetting_factor = compression_config.forgetting_factors[layer_idx] if compression_config.forgetting_factors is not None else None
            if compression_config.streaming_budget is not None:
                self.streaming_budget = compression_config.streaming_budget
                self.recent_budget -= self.streaming_budget
            else:
                self.streaming_budget = 0
            self.score = None
            self.input_ids = None

    def len(self):
        return self.seq_length

    def cat(self, key_tensor: torch.Tensor, value_tensor: torch.Tensor):
        """Concatenate new key-value pairs to existing cache"""
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
        if not (self.use_compression and self.seq_length > self.total_budget):
            return
        elif self.compression_method == "streamingLLM":
            self.key_data = torch.cat((
                self.key_data[:,:,:self.streaming_budget,:],
                self.key_data[:,:,-self.recent_budget:,:]
            ), dim=self.seq_dim)
            
            self.value_data = torch.cat((
                self.value_data[:,:,:self.streaming_budget,:],
                self.value_data[:,:,-self.recent_budget:,:]
            ), dim=self.seq_dim)
        
        attn_scores_shape = attn_scores.shape
        
        attn_scores = attn_scores.view(attn_scores_shape[0], self.num_key_value_heads, -1, *attn_scores_shape[2:]).sum(dim=2)
        
        seq_len = attn_scores.size(2)
        
        # Calculate weighted attention scores
        if self.exponents is not None:
            forgetting = (self.forgetting_factor ** self.exponents).view(1, 1, seq_len, 1)
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
    
    def set_forget(self, forget, exponents):
        self.forget = forget
        self.exponents = exponents