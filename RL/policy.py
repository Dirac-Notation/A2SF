# policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Dict, Tuple

EPS = 1e-6

# a values: [0.0, 0.0001, 0.0005, 0.001, 0.01, 0.1]
A_VALUES = torch.tensor([0.0, 0.0001, 0.0005, 0.001, 0.01, 0.1])

class A2SFPolicy(nn.Module):
    """
    Policy network for A2SF RL agent (single-step / bandit) with sigmoid cache
    - a: discrete action from [0.0, 0.0001, 0.0005, 0.001, 0.01, 0.1]
    - b: continuous action in [0, 1] using Beta distribution
    - Value head as baseline
    """

    def __init__(self, state_dim: int, a_values: torch.Tensor = None):
        super().__init__()
        if a_values is None:
            a_values = A_VALUES
        self.register_buffer('a_values', a_values)
        self.num_a_values = len(a_values)

        # Backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )

        # Policy heads
        # a: discrete (Categorical over 6 values)
        self.a_head = nn.Linear(512, self.num_a_values)
        
        # b: continuous (Beta distribution)
        self.b_mu_head = nn.Linear(512, 1)       # -> sigmoid
        self.b_kappa_head = nn.Linear(512, 1)    # -> softplus + offset

        # Value head
        self.value_head = nn.Linear(512, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def _b_alpha_beta(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stable (mu, kappa) parameterization -> (alpha, beta) for b.
        mu in (0,1), kappa > 0
        """
        mu = torch.sigmoid(self.b_mu_head(h))  # (B,1) in (0,1)
        kappa = F.softplus(self.b_kappa_head(h)) + 1.0  # (>1 keeps distribution well-behaved)
        alpha = (mu * kappa).clamp_min(EPS)
        beta = ((1.0 - mu) * kappa).clamp_min(EPS)
        return alpha.squeeze(-1), beta.squeeze(-1)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: (B, state_dim) or (state_dim,)
        Returns:
            dict with a_logits, b_alpha, b_beta, value
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        h = self.backbone(state)
        a_logits = self.a_head(h)  # (B, num_a_values)
        b_alpha, b_beta = self._b_alpha_beta(h)
        value = self.value_head(h).squeeze(-1)
        return {"a_logits": a_logits, "b_alpha": b_alpha, "b_beta": b_beta, "value": value}

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Sample action (single-step)
        Returns:
            action: tuple of (a, b) where a is discrete index, b is continuous [0,1]
            log_prob: combined log probability
            value: state value
        """
        out = self.forward(state)
        a_logits = out["a_logits"]
        b_alpha, b_beta = out["b_alpha"], out["b_beta"]
        value = out["value"]

        # Sample a (discrete)
        a_dist = torch.distributions.Categorical(logits=a_logits)
        a_idx = a_dist.sample()  # (B,)
        a = self.a_values[a_idx]  # (B,)
        a_log_prob = a_dist.log_prob(a_idx)  # (B,)

        # Sample b (continuous)
        b_dist = torch.distributions.Beta(b_alpha, b_beta)
        b = b_dist.sample()  # (B,)
        b = b.clamp(EPS, 1 - EPS)  # numerical safety
        b_log_prob = b_dist.log_prob(b)  # (B,)

        # Combined log probability
        log_prob = a_log_prob + b_log_prob

        return (a, b), log_prob, value

    def log_prob_value(
        self, state: torch.Tensor, action: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (B, state_dim) or (state_dim,)
            action: tuple of (a, b) where a is the actual a value, b is continuous [0,1]
        Returns:
            log_prob, value, entropy
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        a, b = action
        if a.dim() == 0:
            a = a.unsqueeze(0)
        if b.dim() == 0:
            b = b.unsqueeze(0)

        out = self.forward(state)
        a_logits = out["a_logits"]
        b_alpha, b_beta = out["b_alpha"], out["b_beta"]
        value = out["value"]

        # Find closest a value index
        a_expanded = a.unsqueeze(-1)  # (B, 1)
        a_values_expanded = self.a_values.unsqueeze(0)  # (1, num_a_values)
        a_idx = torch.argmin(torch.abs(a_expanded - a_values_expanded), dim=-1)  # (B,)
        
        # Compute log prob for a
        a_dist = torch.distributions.Categorical(logits=a_logits)
        a_log_prob = a_dist.log_prob(a_idx)
        a_entropy = a_dist.entropy()

        # Compute log prob for b
        b_dist = torch.distributions.Beta(b_alpha, b_beta)
        b_log_prob = b_dist.log_prob(b)
        b_entropy = b_dist.entropy()

        # Combined
        log_prob = a_log_prob + b_log_prob
        entropy = a_entropy + b_entropy

        return log_prob, value, entropy

def ppo_update(
    policy: A2SFPolicy,
    buffer,
    config,
    optimizer: torch.optim.Optimizer
) -> Dict[str, float]:
    """
    PPO update for single-step episodes with (a, b) actions.
    - advantages = rewards - values (baseline)
    - returns    = rewards
    """
    policy.train()

    states, actions, old_log_probs, rewards, old_values = buffer.get()
    
    # Single-step advantages/returns
    advantages = (rewards - old_values).detach()

    # Shuffle
    batch_size = states.size(0)
    idx = torch.randperm(batch_size, device=states.device)
    states = states[idx]
    actions = (actions[0][idx], actions[1][idx])  # Unpack tuple, shuffle, repack
    old_log_probs = old_log_probs[idx]
    rewards = rewards[idx]
    advantages = advantages[idx]

    policy_losses, value_losses, entropies = [], [], []

    for _ in range(config.update_epochs):
        for start in range(0, batch_size, config.minibatch_size):
            end = start + config.minibatch_size
            
            state = states[start:end]
            action = (actions[0][start:end], actions[1][start:end])  # Slice tuple
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
