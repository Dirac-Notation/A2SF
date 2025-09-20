import torch
from typing import List, Tuple
import numpy as np

class RolloutBuffer:
    """Experience buffer for PPO training"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.clear()
    
    def clear(self):
        """Clear the buffer"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        log_prob: torch.Tensor, 
        reward: torch.Tensor, 
        value: torch.Tensor, 
        done: torch.Tensor
    ):
        """
        Add experience to buffer
        
        Args:
            state: State tensor
            action: Action tensor
            log_prob: Log probability of action
            reward: Reward scalar
            value: Value estimate
            done: Done flag
        """
        self.states.append(state.to(self.device))
        self.actions.append(action.to(self.device))
        self.log_probs.append(log_prob.to(self.device))
        self.rewards.append(reward.to(self.device))
        self.values.append(value.to(self.device))
        self.dones.append(done.to(self.device))
    
    def compute_gae(self, gamma: float, gae_lambda: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Tuple of (advantages, returns)
        """
        if not self.states:
            return torch.tensor([]), torch.tensor([])
        
        # Convert to tensors
        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values)
        dones = torch.stack(self.dones)
        
        # Add bootstrap value for last state
        if not dones[-1]:
            # If episode is not done, we need a bootstrap value
            # For simplicity, we'll use the last value
            bootstrap_value = values[-1]
        else:
            bootstrap_value = 0.0
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        last_advantage = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = bootstrap_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t].float()) - values[t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * (1 - dones[t].float()) * last_advantage
        
        # Compute returns
        returns = advantages + values
        
        return advantages, returns
    
    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all experiences as tensors
        
        Returns:
            Tuple of (states, actions, log_probs, returns, advantages)
        """
        if not self.states:
            return (torch.tensor([]), torch.tensor([]), torch.tensor([]), 
                   torch.tensor([]), torch.tensor([]))
        
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        log_probs = torch.stack(self.log_probs)
        
        # Compute GAE
        advantages, returns = self.compute_gae(0.99, 0.95)  # Default values, will be overridden
        
        return states, actions, log_probs, returns, advantages
    
    def size(self) -> int:
        """Get buffer size"""
        return len(self.states)
