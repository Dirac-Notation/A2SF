import torch
from . import KVCache

class A2SFCache(KVCache):
    """A2SF cache implementation (forgetting_factor != 1)"""
    
    def __init__(self, num_key_value_heads: int, seq_dim: int = 2):
        super().__init__(num_key_value_heads, seq_dim)
        self.forgetting_factor = None
        self.forget = None
        self.exponents = None
        self.input_ids = None
        self.prompt = False
    
    def init_cache(self, compression_config, layer_idx):
        """Initialize A2SF cache settings"""
        super().init_cache(compression_config, layer_idx)
        self.forgetting_factor = compression_config.forgetting_factors[layer_idx] if compression_config.forgetting_factors is not None else None
        self.input_ids = None
        self.prompt = False

    def update(self, attn_scores):
        """Update cache using A2SF method (forgetting_factor != 1)"""
        if not self.prompt:
            self.prepare_scores(attn_scores)
            self.select()
            self.prompt = True
        else:
            return
    
    def prepare_scores(self, attn_scores):
        """Prepare scores for A2SF method (with forgetting factor)"""
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
    
    def flash_prepare_scores(self, attn_scores):
        seq_len = attn_scores.size(2)
        
        if self.exponents is not None:
            forgetting = (self.forgetting_factor ** self.exponents.to(attn_scores.device)).view(1, 1, seq_len, 1)
            return (forgetting * attn_scores).sum(dim=self.seq_dim)
        else:
            current_score = attn_scores.sum(self.seq_dim)
            if self.forget:
                self.score *= self.forgetting_factor
                return current_score*self.forgetting_factor
            return current_score
    
    def set_forget(self, forget, exponents):
        """Set forgetting parameters for A2SF"""
        self.forget = forget
        self.exponents = exponents