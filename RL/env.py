import torch
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from .features import ContextEncoder
from .config import A2SFRLConfig

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
            model_name=config.sentence_transformer_model,
            device=config.device,
            context_window=config.context_window,
            max_context=config.max_context
        )
        
        # Current episode cache
        self.current_prompt = None
        self.current_dataset = None
        self.current_selected_indices = None
    
    def encode_to_state(self, prompt: str, selected_indices: list, dataset: str = None) -> torch.Tensor:
        self.current_prompt = prompt
        self.current_dataset = dataset
        self.current_selected_indices = selected_indices
        
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
                selected_indices=self.current_selected_indices,
                dataset=self.current_dataset
            )
        
        reward = torch.tensor(float(result.reward), device=self.device)
        
        info = {
            "a": a_val,
            "b": b_val,
            "reward": result.reward
        }
        
        return reward, info
