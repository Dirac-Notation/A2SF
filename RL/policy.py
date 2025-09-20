import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math
import numpy as np

class A2SFPolicy(nn.Module):
    """Policy network for A2SF RL agent"""
    
    def __init__(self, state_dim: int, action_min: float = 0.0, action_max: float = 1.0):
        super().__init__()
        self.action_min = action_min
        self.action_max = action_max
        
        # Policy network
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )
        
        # Policy head (outputs alpha and beta for beta distribution)
        self.policy_alpha = nn.Linear(512, 1)
        self.policy_beta = nn.Linear(512, 1)
        
        # Value head
        self.value_head = nn.Linear(512, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier uniform initialization for better stability
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the policy network
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Dict containing policy outputs and value
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        features = self.backbone(state)
        
        # Policy outputs (alpha and beta for beta distribution)
        alpha_raw = self.policy_alpha(features)
        beta_raw = self.policy_beta(features)
        alpha = F.relu(alpha_raw) + 1.0  # Ensure alpha > 1
        beta = F.relu(beta_raw) + 1.0    # Ensure beta > 1
        
        # Value output
        value = self.value_head(features).squeeze(-1)
        
        return {
            "alpha": alpha.squeeze(-1),
            "beta": beta.squeeze(-1),
            "value": value
        }
    
    @torch.no_grad()
    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        outputs = self.forward(state)
        alpha = outputs["alpha"]
        beta = outputs["beta"]
        value = outputs["value"]
        
        # Create beta distribution
        dist = torch.distributions.Beta(alpha, beta)
        
        # Sample action
        action = dist.sample()
        
        # Clamp action to valid range
        action = torch.clamp(action, self.action_min, self.action_max)
        
        # Compute log probability
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def log_prob_value(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probability and value for given state-action pairs
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size,)
            
        Returns:
            Tuple of (log_prob, value)
        """
        outputs = self.forward(state)
        alpha = outputs["alpha"]
        beta = outputs["beta"]
        value = outputs["value"]
        
        # Create beta distribution
        dist = torch.distributions.Beta(alpha, beta)
        
        # Compute log probability
        log_prob = dist.log_prob(action)
        
        return log_prob, value

def ppo_update(
    policy: A2SFPolicy, 
    buffer, 
    config,
    optimizer: torch.optim.Optimizer
) -> Dict[str, float]:
    """
    Perform PPO update
    
    Args:
        policy: Policy network
        buffer: Experience buffer
        config: Configuration object
        optimizer: Optimizer
        
    Returns:
        Dict containing loss statistics
    """
    policy.train()
    
    # Compute GAE returns
    _, _ = buffer.compute_gae(config.gamma, config.gae_lambda)
    states, actions, old_log_probs, returns, advantages = buffer.get()
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Shuffle data
    batch_size = states.size(0)
    indices = torch.randperm(batch_size, device=states.device)
    states = states[indices]
    actions = actions[indices]
    old_log_probs = old_log_probs[indices]
    returns = returns[indices]
    advantages = advantages[indices]
    
    # Training statistics
    policy_losses = []
    value_losses = []
    
    # PPO updates
    for epoch in range(config.update_epochs):
        for start in range(0, batch_size, config.minibatch_size):
            end = start + config.minibatch_size
            batch_states = states[start:end]
            batch_actions = actions[start:end]
            batch_old_log_probs = old_log_probs[start:end]
            batch_returns = returns[start:end]
            batch_advantages = advantages[start:end]
            
            # Forward pass
            log_probs, values = policy.log_prob_value(batch_states, batch_actions)
            
            # Compute ratios
            ratios = torch.exp(log_probs - batch_old_log_probs)
            
            # Policy loss
            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, batch_returns)
            
            # Total loss
            total_loss = policy_loss + config.value_coef * value_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()
            
            # Store losses
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
    
    return {
        "policy_loss": np.mean(policy_losses),
        "value_loss": np.mean(value_losses)
    }
