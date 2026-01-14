import torch
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModel
from .main import A2SFRLConfig

class ContextEncoder:
    """Encodes text using jina-embeddings model with CLS token"""
    
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-small-en", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
    
    def encode_context(self, text: str) -> torch.Tensor:
        """
        Encode entire text and extract CLS token
        
        Args:
            text: Input text string
            
        Returns:
            torch.Tensor: CLS token embedding (embedding_dim dimensions)
        """
        if not text or not text.strip():
            # Empty text, return zero vector
            return torch.zeros(self.embedding_dim, device=self.device)
        
        # Encode entire text at once
        with torch.no_grad():
            # Tokenize entire text
            input_ids = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8192
            ).input_ids.to(self.device)
            
            outputs = self.model(input_ids)
            # Extract CLS token (first token)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]
            cls_embedding = cls_embedding.squeeze(0)  # [hidden_size]
        
        return cls_embedding

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
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Context encoder
        self.context_encoder = ContextEncoder(
            model_name=config.context_encoder_model,
            device=config.device
        )
        
        # Current episode cache
        self.current_prompt = None
        self.current_dataset = None
        self.current_generated_text_full = None
    
    def encode_to_state(self, prompt: str, generated_text_full: str, dataset: str = None) -> torch.Tensor:
        self.current_prompt = prompt
        self.current_dataset = dataset
        self.current_generated_text_full = generated_text_full
        
        context_embedding = self.context_encoder.encode_context(prompt).to(self.device, dtype=torch.float32)
        
        return context_embedding
    
    def step(self, action: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            action: tuple of (a, b) where a is the a parameter value, b is in [0, 1]
        Returns:
            reward, info
        """
        a, b = action
        a_val = float(a.item() if isinstance(a, torch.Tensor) else a)
        b_val = float(b.item() if isinstance(b, torch.Tensor) else b)

        with torch.no_grad():
            result = self.runner.run_with_compression(
                prompt=self.current_prompt,
                a=a_val,
                b=b_val,
                generated_text_full=self.current_generated_text_full,
                dataset=self.current_dataset
            )
        
        reward = torch.tensor(float(result.reward), device=self.device)
        
        info = {
            "a": a_val,
            "b": b_val,
            "reward": result.reward,
            "generated_text": result.generated_text
        }
        
        return reward, info
