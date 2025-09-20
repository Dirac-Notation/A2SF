import torch
import torch.nn.functional as F
from typing import List, Optional
import numpy as np

from sentence_transformers import SentenceTransformer

class ContextEncoder:
    """Encodes recent context tokens using sentence transformer"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.device = device
        self.sentence_transformer = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.sentence_transformer.get_sentence_embedding_dimension()
    
    def encode_context(self, tokens: List[str], max_tokens: int = 64) -> torch.Tensor:
        """
        Encode recent tokens using sentence transformer
        
        Args:
            tokens: List of token strings
            max_tokens: Maximum number of tokens to consider
            
        Returns:
            torch.Tensor: Encoded context vector
        """
        # Take the most recent tokens
        streaming_len = max_tokens // 4
        recent_len = max_tokens - streaming_len
        context_tokens = tokens[:streaming_len] + tokens[-recent_len:] if len(tokens) > max_tokens else tokens
        
        # Join tokens into a sentence
        context_text = " ".join(context_tokens)
        
        # Encode using sentence transformer
        with torch.no_grad():
            embedding = self.sentence_transformer.encode(
                context_text, 
                convert_to_tensor=True, 
                device=self.device
            )
        
        return embedding

def build_state_from_context(
    context_embedding: torch.Tensor
) -> torch.Tensor:
    """
    Build state vector from context embedding only
    
    Args:
        context_embedding: Context embedding from sentence transformer
        
    Returns:
        torch.Tensor: State vector (same as context embedding)
    """
    return context_embedding

