import torch
import json
import os
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from .features import ContextEncoder, build_state_from_context
from .config import A2SFRLConfig

@dataclass
class EpisodeResult:
    """Result of an episode"""
    accuracy_score: float
    forgetting_factor: float
    total_reward: float
    metrics: Dict[str, Any]

class A2SFEnv:
    """RL Environment for A2SF model"""
    
    def __init__(self, runner, config: A2SFRLConfig):
        self.runner = runner
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Context encoder
        self.context_encoder = ContextEncoder(
            model_name=config.sentence_transformer_model,
            device=config.device
        )
        
        # Current episode state
        self.current_prompt = None
        self.current_task = None
        self.current_tokens = None
        self.current_answers = None
    
    def reset(self, prompt: str, task: str, tokens: list, answers: list) -> torch.Tensor:
        """
        Reset environment for new episode
        
        Args:
            prompt: Input prompt text
            task: Task type
            tokens: List of token strings
            answers: Ground truth answers (required)
            
        Returns:
            Initial state tensor
        """
        self.current_prompt = prompt
        self.current_task = task
        self.current_tokens = tokens
        self.current_answers = answers
        
        # Encode context
        context_embedding = self.context_encoder.encode_context(
            tokens, max_tokens=self.config.context_window
        )
        
        # Build state
        state = build_state_from_context(context_embedding)
        
        return state
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool, Dict[str, Any]]:
        """
        Execute action and return next state, reward, done, info
        
        Args:
            action: Compression ratio action (0.0 to 1.0)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Clamp action to valid range
        forgetting_factor = torch.clamp(action, self.config.action_min, self.config.action_max).item()
        
        # Run model with forgetting factor
        result = self.runner.run_with_compression(
            prompt=self.current_prompt,
            task=self.current_task,
            forgetting_factor=forgetting_factor,
            answers=self.current_answers
        )
        
        # Compute reward
        reward = result.accuracy_score
        
        # Build next state (same as current since episode is done after one step)
        context_embedding = self.context_encoder.encode_context(
            self.current_tokens, max_tokens=self.config.context_window
        )
        
        next_state = build_state_from_context(context_embedding)
        
        # Episode is done after one step
        done = True
        
        # Info dictionary
        info = {
            "accuracy_score": result.accuracy_score,
            "forgetting_factor": forgetting_factor,
            "metrics": result.metrics
        }
        
        return next_state, torch.tensor(reward, device=self.device), done, info
