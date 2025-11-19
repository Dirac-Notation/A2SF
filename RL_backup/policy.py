# policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Dict, Tuple

EPS = 1e-6

class A2SFPolicy(nn.Module):
    """
    Policy network for A2SF RL agent (single-step / bandit)
    - Continuous action in [0, 1]
    - Beta policy: network outputs (mu, kappa) -> (alpha, beta)
    - Value head as baseline
    """

    def __init__(self, state_dim: int, action_min: float = 0.0, action_max: float = 1.0):
        super().__init__()
        self.action_min = action_min
        self.action_max = action_max

        # Backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )

        # Policy heads (μ in (0,1), κ>0)
        self.mu_head = nn.Linear(512, 1)       # -> sigmoid
        self.kappa_head = nn.Linear(512, 1)    # -> softplus + offset

        # Value head
        self.value_head = nn.Linear(512, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def _alpha_beta(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stable (mu, kappa) parameterization -> (alpha, beta).
        mu in (0,1), kappa > 0
        """
        mu = torch.sigmoid(self.mu_head(h))  # (B,1) in (0,1)
        kappa = F.softplus(self.kappa_head(h)) + 1.0  # (>1 keeps distribution well-behaved)
        alpha = (mu * kappa).clamp_min(EPS)
        beta = ((1.0 - mu) * kappa).clamp_min(EPS)
        return alpha.squeeze(-1), beta.squeeze(-1)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: (B, state_dim) or (state_dim,)
        Returns:
            dict with alpha, beta, value
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        h = self.backbone(state)
        alpha, beta = self._alpha_beta(h)
        value = self.value_head(h).squeeze(-1)
        return {"alpha": alpha, "beta": beta, "value": value}

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action (single-step)
        Returns:
            action (B,), log_prob (B,), value (B,)
        """
        out = self.forward(state)
        alpha, beta, value = out["alpha"], out["beta"], out["value"]

        dist = torch.distributions.Beta(alpha, beta)
        action = dist.sample()
        action = action.clamp(EPS, 1 - EPS)  # numerical safety in log_prob

        log_prob = dist.log_prob(action)  # log prob in (0,1) space
        return action.squeeze(-1), log_prob.squeeze(-1), value.squeeze(-1)

    def log_prob_value(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        out = self.forward(state)
        alpha, beta, value = out["alpha"], out["beta"], out["value"]

        dist = torch.distributions.Beta(alpha, beta)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob.squeeze(-1), value.squeeze(-1), entropy.squeeze(-1)

def ppo_update(
    policy: A2SFPolicy,
    buffer,
    config,
    optimizer: torch.optim.Optimizer
) -> Dict[str, float]:
    """
    PPO update for single-step episodes.
    - advantages = rewards - values (baseline)
    - returns    = rewards
    """
    policy.train()

    states, actions, old_log_probs, rewards, old_values = buffer.get()
    
    # Single-step advantages/returns
    advantages = (rewards - old_values).detach()

    # Shuffle
    batch_size = states.size(0)
    idx = torch.randperm(config.episodes_per_update, device=states.device)
    states, actions, old_log_probs, rewards, advantages = states[idx], actions[idx], old_log_probs[idx], rewards[idx], advantages[idx]

    policy_losses, value_losses, entropies = [], [], []

    for _ in range(config.update_epochs):
        for start in range(0, batch_size, config.minibatch_size):
            end = start + config.minibatch_size
            
            state = states[start:end]
            action = actions[start:end]
            old_log_prob = old_log_probs[start:end]
            reward = rewards[start:end]
            advantage = advantages[start:end]

            # Recompute under current policy
            log_prob, value, entropy = policy.log_prob_value(state, action)

            # PPO ratio
            ratio = torch.exp(log_prob - old_log_prob)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (MSE to returns)
            value_loss = F.mse_loss(value, reward)

            # Entropy bonus
            entropy_bonus = entropy.mean()

            total_loss = (policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_bonus)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy_bonus.item())

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
    }
