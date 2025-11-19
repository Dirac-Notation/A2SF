import torch
import torch.nn.functional as F
from typing import List, Optional
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

class ContextEncoder:
    """Encodes text using sentence transformer with chunking and similarity calculation"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu", context_window: int = 512, max_context: int = 32):
        self.device = device
        self.sentence_transformer = SentenceTransformer(model_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/"+model_name)
        self.embedding_dim = self.sentence_transformer.get_sentence_embedding_dimension()
        self.context_window = context_window
        self.max_context = max_context
    
    def encode_context(self, text: str) -> torch.Tensor:
        """
        Encode text by chunking into 512-token segments and computing similarity with last segment
        
        Args:
            text: Input text string
            
        Returns:
            torch.Tensor: Fixed-length similarity vector (32 dimensions)
        """
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Create chunks of context_window
        batched_tokens = []
        for i in range(len(tokens), 0, -self.context_window):
            if i < self.context_window:
                batched_tokens.append(tokens[:i])
            else:    
                batched_tokens.append(tokens[i-self.context_window:i])
        batched_tokens.reverse()
        
        # Decode tokens back to text
        batched_text = []
        for tokens in batched_tokens:
            batched_text.append(self.tokenizer.decode(tokens))
        
        # Encode using sentence transformer
        with torch.no_grad():
            embedding = self.sentence_transformer.encode(
                batched_text,
                batch_size=len(batched_text),
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        
        # Compute similarity with last segment
        similarity_vector = self._compute_similarity(embedding)
        
        # Pad or truncate to fixed length of 32
        fixed_similarity = self._fix_length(similarity_vector, target_length=self.max_context)

        return fixed_similarity
    
    def _compute_similarity(self, embedding: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between all segments and the last segment"""
        if embedding.size(0) <= 1:
            # If only one segment, return a single similarity value
            return torch.tensor([1.0], device=self.device)
        
        # Get the last segment as reference
        last_segment = embedding[-1:]  # Shape: [1, embedding_dim]
        
        # Compute cosine similarity with all segments
        # embedding: [num_segments, embedding_dim]
        # last_segment: [1, embedding_dim]
        similarities = F.cosine_similarity(embedding, last_segment, dim=1)
        
        return similarities
    
    def _fix_length(self, vector: torch.Tensor, target_length: int = 64) -> torch.Tensor:
        """Pad or truncate vector to target length"""
        current_length = vector.size(0)
        
        if current_length == target_length:
            return vector
        elif current_length > target_length:
            # Truncate from the end (keep the most recent similarities)
            return vector[-target_length:]
        else:
            # Pad with zeros on the left
            padding_size = target_length - current_length
            padding = torch.zeros(padding_size, device=self.device, dtype=vector.dtype)
            return torch.cat([padding, vector], dim=0)