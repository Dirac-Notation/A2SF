import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from .main import A2SFRLConfig

# Import RoPE functions from transformers
try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
except ImportError:
    # Fallback: try to import from utils_real_drop if transformers import fails
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils_real_drop.kv_llama import apply_rotary_pos_emb

class AttentionEncoder(nn.Module):
    """
    Attention-based encoder using cloned parameters from target model's first layer, first head
    - Clones target model's embedding layer weights
    - Clones first layer's first head Q, K projections from target model
    - Query: last 16 tokens, Key: all tokens
    - Creates independent RoPE from first layer's config
    - Performs attention: softmax(Q*K^T/sqrt(head_dim))
    - Sums over query dimension to get (1, seq_len) vector
    - Pads to 8192 dimensions
    - All parameters are frozen (no training)
    """
    
    def __init__(self, target_model, target_tokenizer, device: str = "cpu", output_dim: int = 8192, num_query_tokens: int = 16):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.target_tokenizer = target_tokenizer
        self.output_dim = output_dim
        self.num_query_tokens = num_query_tokens
        
        # Get config from target model
        if hasattr(target_model, 'config'):
            config = target_model.config
        elif hasattr(target_model, 'model') and hasattr(target_model.model, 'config'):
            config = target_model.model.config
        else:
            raise ValueError("Cannot determine config from target model")
        
        self.embedding_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embedding_dim // self.num_heads
        
        # Get first layer's attention module to clone parameters
        first_layer_attn = self._get_first_layer_attention(target_model)
        
        # Clone embedding layer weights
        embed_layer = self._get_embedding_layer(target_model)
        with torch.no_grad():
            # Clone embedding weights: (vocab_size, hidden_size)
            embed_weight = embed_layer.weight.clone().to(dtype=torch.float32)
        
        # Register embedding as buffer
        self.register_buffer('embed_weight', embed_weight)
        
        # Extract only first head's parameters from q_proj and k_proj
        # q_proj.weight shape: (num_heads * head_dim, hidden_size)
        # k_proj.weight shape: (num_key_value_heads * head_dim, hidden_size)
        # First head: weight[0:head_dim, :]
        with torch.no_grad():
            q_proj_first_head_weight = first_layer_attn.q_proj.weight[0:self.head_dim, :].clone().to(dtype=torch.float32)  # (head_dim, hidden_size)
            k_proj_first_head_weight = first_layer_attn.k_proj.weight[0:self.head_dim, :].clone().to(dtype=torch.float32)  # (head_dim, hidden_size)
            
            # Extract bias if exists (first head only)
            if first_layer_attn.q_proj.bias is not None:
                q_proj_first_head_bias = first_layer_attn.q_proj.bias[0:self.head_dim].clone().to(dtype=torch.float32)  # (head_dim,)
            else:
                q_proj_first_head_bias = None
                
            if first_layer_attn.k_proj.bias is not None:
                k_proj_first_head_bias = first_layer_attn.k_proj.bias[0:self.head_dim].clone().to(dtype=torch.float32)  # (head_dim,)
            else:
                k_proj_first_head_bias = None
        
        # Register as buffers (not parameters, so they won't be trained)
        self.register_buffer('q_proj_first_head', q_proj_first_head_weight)
        self.register_buffer('k_proj_first_head', k_proj_first_head_weight)
        if q_proj_first_head_bias is not None:
            self.register_buffer('q_proj_first_head_bias', q_proj_first_head_bias)
        else:
            self.register_buffer('q_proj_first_head_bias', None)
        if k_proj_first_head_bias is not None:
            self.register_buffer('k_proj_first_head_bias', k_proj_first_head_bias)
        else:
            self.register_buffer('k_proj_first_head_bias', None)
        
        # Create independent RoPE from first layer's config
        # Clone rotary embedding parameters
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        with torch.no_grad():
            # Get rotary embedding config from first layer
            rope_head_dim = first_layer_attn.head_dim
            rope_max_position_embeddings = first_layer_attn.max_position_embeddings
            rope_theta = first_layer_attn.rope_theta
            
            # Create independent rotary embedding
            self.rotary_emb = LlamaRotaryEmbedding(
                rope_head_dim,
                max_position_embeddings=rope_max_position_embeddings,
                base=rope_theta,
            )
            # Clone inv_freq if it exists
            if hasattr(first_layer_attn.rotary_emb, 'inv_freq'):
                self.rotary_emb.inv_freq = first_layer_attn.rotary_emb.inv_freq.clone()
        
        # Scale factor for attention (using head_dim instead of query_dim)
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        # Move all buffers to specified device
        self.to(self.device)
        
        # Freeze all parameters (encoder should not be trained)
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def _get_first_layer_attention(self, target_model):
        """Get the first layer's attention module from target model"""
        # Try different model structures
        if hasattr(target_model, 'model') and hasattr(target_model.model, 'layers'):
            # Standard structure: model.model.layers[0].self_attn
            if len(target_model.model.layers) == 0:
                raise ValueError("Target model has no layers")
            return target_model.model.layers[0].self_attn
        elif hasattr(target_model, 'layers'):
            # Alternative structure: model.layers[0].self_attn
            if len(target_model.layers) == 0:
                raise ValueError("Target model has no layers")
            return target_model.layers[0].self_attn
        else:
            raise ValueError("Cannot find layers in target model")
    
    def _get_embedding_layer(self, target_model):
        """Get the embedding layer from target model"""
        if hasattr(target_model, 'embed_tokens'):
            return target_model.embed_tokens
        elif hasattr(target_model, 'model') and hasattr(target_model.model, 'embed_tokens'):
            return target_model.model.embed_tokens
        else:
            raise ValueError("Cannot find embedding layer in target model")
    
    def _init_weights(self):
        """No initialization needed - using cloned parameters"""
        pass
    
    def trainable_parameters(self):
        """
        Return empty list - encoder is frozen and not trained.
        """
        return []
    
    def encode_context(self, text: str, generation_length: int, token_budget: int) -> torch.Tensor:
        """
        Encode text using attention mechanism with target model's first layer, first head
        
        Args:
            text: Input text string
            generation_length: Generation length
            token_budget: Token budget for compression
        Returns:
            torch.Tensor: Encoded vector of shape (output_dim + 2,)
        """  
        # Tokenize text
        tokenized = self.target_tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False
        )
        input_ids = tokenized.input_ids.to(self.device)  # (1, seq_len)
        seq_len = input_ids.size(1)
        
        # Get embeddings using cloned embedding weights
        with torch.no_grad():
            # Use cloned embedding weights: (1, seq_len, embedding_dim)
            # F.embedding(input_ids, embed_weight) is equivalent to embed_layer(input_ids)
            token_embeddings = F.embedding(input_ids, self.embed_weight).to(dtype=torch.float32)  # (1, seq_len, embedding_dim)
        
        # Extract query (last num_query_tokens) and key (all tokens)
        if seq_len <= self.num_query_tokens:
            # If sequence is shorter than num_query_tokens, use all tokens for query
            query_embeddings = token_embeddings  # (1, seq_len, embedding_dim)
            key_embeddings = token_embeddings  # (1, seq_len, embedding_dim)
            actual_query_len = seq_len
        else:
            # Query: last num_query_tokens
            query_embeddings = token_embeddings[:, -self.num_query_tokens:, :]  # (1, num_query_tokens, embedding_dim)
            # Key: all tokens
            key_embeddings = token_embeddings  # (1, seq_len, embedding_dim)
            actual_query_len = self.num_query_tokens
        
        # Use first head's parameters directly (already extracted in __init__)
        with torch.no_grad():
            # Project using first head's q_proj and k_proj parameters
            # query_embeddings: (1, actual_query_len, embedding_dim)
            # key_embeddings: (1, seq_len, embedding_dim)
            # q_proj_first_head: (head_dim, hidden_size)
            # k_proj_first_head: (head_dim, hidden_size)
            # Ensure both are float32 for matmul
            query_embeddings_f32 = query_embeddings.to(dtype=torch.float32)
            key_embeddings_f32 = key_embeddings.to(dtype=torch.float32)
            query_states = torch.matmul(query_embeddings_f32, self.q_proj_first_head.t())  # (1, actual_query_len, head_dim)
            key_states = torch.matmul(key_embeddings_f32, self.k_proj_first_head.t())  # (1, seq_len, head_dim)
            
            # Add bias if exists
            if self.q_proj_first_head_bias is not None:
                query_states = query_states + self.q_proj_first_head_bias.unsqueeze(0)  # (1, actual_query_len, head_dim)
            if self.k_proj_first_head_bias is not None:
                key_states = key_states + self.k_proj_first_head_bias.unsqueeze(0)  # (1, seq_len, head_dim)
            
            # Reshape for RoPE: need (batch, num_heads, seq_len, head_dim) format
            # Add head dimension: (1, 1, seq_len, head_dim)
            query_states = query_states.unsqueeze(1)  # (1, 1, actual_query_len, head_dim)
            key_states = key_states.unsqueeze(1)  # (1, 1, seq_len, head_dim)
            
            # Apply RoPE from first layer
            # Create position_ids for query and key
            if actual_query_len == seq_len:
                # Same sequence, use same position_ids
                position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, seq_len)
                query_position_ids = position_ids
                key_position_ids = position_ids
            else:
                # Query is last num_query_tokens, key is all tokens
                query_position_ids = torch.arange(seq_len - actual_query_len, seq_len, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, actual_query_len)
                key_position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, seq_len)
            
            # Get RoPE embeddings using cloned rotary_emb
            # rotary_emb expects (tensor, position_ids) and returns (cos, sin)
            cos_query, sin_query = self.rotary_emb(query_states, query_position_ids)
            cos_key, sin_key = self.rotary_emb(key_states, key_position_ids)
            
            # Apply RoPE - apply_rotary_pos_emb(query, key, cos, sin) returns (rotated_query, rotated_key)
            # Since query and key have different position_ids, we apply RoPE separately
            # For query: use query's cos/sin
            query_states_rotated, _ = apply_rotary_pos_emb(query_states, query_states, cos_query, sin_query)
            query_states = query_states_rotated
            # For key: use key's cos/sin
            _, key_states_rotated = apply_rotary_pos_emb(key_states, key_states, cos_key, sin_key)
            key_states = key_states_rotated
            
            # Now query_states: (1, 1, actual_query_len, head_dim)
            # key_states: (1, 1, seq_len, head_dim)
            # Squeeze head dimension for attention computation
            query_states = query_states.squeeze(1)  # (1, actual_query_len, head_dim)
            key_states = key_states.squeeze(1)  # (1, seq_len, head_dim)
        
        # Compute attention scores: Q * K^T
        # Q: (1, actual_query_len, head_dim)
        # K: (1, seq_len, head_dim)
        # Q * K^T: (1, actual_query_len, seq_len)
        attention_scores = torch.bmm(query_states, key_states.transpose(1, 2))  # (1, actual_query_len, seq_len)
        
        # Scale by sqrt(head_dim)
        attention_scores = attention_scores * self.scale
        
        # Apply softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (1, actual_query_len, seq_len)
        
        # Sum over query dimension (axis 1) to get (1, seq_len)
        attention_output = attention_weights.sum(dim=1)  # (1, seq_len)
        
        # Squeeze batch dimension
        attention_output = attention_output.squeeze(0)  # (seq_len,)
        
        # Pad to output_dim (8192) from the left with zeros
        if seq_len < self.output_dim:
            padding = torch.zeros(self.output_dim - seq_len, device=self.device)
            attention_output = torch.cat([padding, attention_output], dim=0)  # (output_dim,)
        elif seq_len > self.output_dim:
            # If longer, truncate from the left (keep rightmost output_dim tokens)
            attention_output = attention_output[-self.output_dim:]  # (output_dim,)
        
        # Add generation_length and token_budget features
        generation_feature = torch.zeros(1, device=self.device) + (generation_length/512)
        token_budget_feature = torch.zeros(1, device=self.device) + (token_budget/2048)
        attention_output = torch.cat([attention_output, generation_feature, token_budget_feature], dim=0)  # (output_dim + 2,)
        
        return attention_output

@dataclass
class EpisodeResult:
    """Result of an episode"""
    accuracy_score: float
    forgetting_factor: float
    total_reward: float
    metrics: Dict[str, Any]

class A2SFEnv:
    """RL Environment for A2SF model (single-step / bandit)"""
    
    def __init__(self, runner, config: A2SFRLConfig):
        self.runner = runner
        self.config = config
        self.device = torch.device(config.device)
        
        # Attention encoder using target model's first layer, first head parameters
        self.context_encoder = AttentionEncoder(
            target_model=runner.model,
            target_tokenizer=runner.tokenizer,
            device=config.device,
            output_dim=8192,
            num_query_tokens=16
        )
        
        # Current episode cache
        self.current_prompt = None
        self.current_dataset = None
        self.current_answer = None
        self.current_generation_length = None
        self.current_token_budget = None
    
    def encode_to_state(self, prompt: str, generation_length: int, answer: str, token_budget: int, dataset: str = None) -> torch.Tensor:
        self.current_prompt = prompt
        self.current_dataset = dataset
        self.current_answer = answer
        self.current_generation_length = generation_length
        self.current_token_budget = token_budget
        
        return self.context_encoder.encode_context(prompt, generation_length, token_budget).to(self.device, dtype=torch.float32)
    
    def step(self, action: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            action: tuple of (a, b) tensors for sigmoid cache parameters
        Returns:
            reward, info
        """
        a_val, b_val = action
        a_val = float(a_val.item() if isinstance(a_val, torch.Tensor) else a_val)
        b_val = float(b_val.item() if isinstance(b_val, torch.Tensor) else b_val)

        with torch.no_grad():
            result = self.runner.run_with_compression(
                prompt=self.current_prompt,
                a=a_val,
                b=b_val,
                generation_length=self.current_generation_length,
                token_budget=self.current_token_budget,
                answer=self.current_answer,
                dataset=self.current_dataset,
            )
        
        reward = torch.tensor(float(result.reward), device=self.device)
        
        info = {
            "a": a_val,
            "b": b_val,
            "reward": result.reward,
            "generated_text": result.generated_text,
        }
        
        return reward, info
