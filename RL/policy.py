# policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Dict, Tuple

EPS = 1e-6

# a values: [0.0, 0.0001, 0.0005, 0.001, 0.01, 0.1]
A_VALUES = torch.tensor([0.0, 0.0001, 0.0005, 0.001, 0.01, 0.1])
# b values: [8192, 4096, 1024, 256, 64, 32, 16, 8, 4, 2, 1]
B_VALUES = torch.tensor([8192, 4096, 1024, 256, 64, 32, 16, 8, 4, 2, 1])

class A2SFPolicy(nn.Module):
    """
    Policy network for A2SF RL agent (single-step / bandit) with sigmoid cache
    - a: discrete action from [0.0, 0.0001, 0.0005, 0.001, 0.01, 0.1]
    - b: discrete action from [8192, 4096, 1024, 256, 64, 32, 16, 8, 4, 2, 1]
    - Value head as baseline
    """

    def __init__(self, state_dim: int, a_values: torch.Tensor = None, b_values: torch.Tensor = None):
        super().__init__()
        if a_values is None:
            a_values = A_VALUES
        if b_values is None:
            b_values = B_VALUES
        self.register_buffer('a_values', a_values)
        self.register_buffer('b_values', b_values)
        self.num_a_values = len(a_values)
        self.num_b_values = len(b_values)

        # Backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )

        # Policy heads
        # a: discrete (Categorical over 6 values)
        self.a_head = nn.Linear(512, self.num_a_values)
        
        # b: discrete (Categorical over 11 values)
        self.b_head = nn.Linear(512, self.num_b_values)

        # Value head
        self.value_head = nn.Linear(512, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: (B, state_dim) or (state_dim,)
        Returns:
            dict with a_logits, b_logits, value
        """
        h = self.backbone(state)
        a_logits = self.a_head(h)  # (B, num_a_values)
        b_logits = self.b_head(h)  # (B, num_b_values)
        value = self.value_head(h)
        return {"a_logits": a_logits, "b_logits": b_logits, "value": value}

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Sample action (single-step)
        Returns:
            action: tuple of (a, b) where both are discrete values
            log_prob: combined log probability
            value: state value
        """
        out = self.forward(state)
        a_logits = out["a_logits"]
        b_logits = out["b_logits"]
        value = out["value"]

        # Sample a (discrete)
        a_dist = torch.distributions.Categorical(logits=a_logits)
        a_idx = a_dist.sample()  # (B,)
        a = self.a_values[a_idx]  # (B,)
        a_log_prob = a_dist.log_prob(a_idx)  # (B,)

        # Sample b (discrete)
        b_dist = torch.distributions.Categorical(logits=b_logits)
        b_idx = b_dist.sample()  # (B,)
        b = self.b_values[b_idx]  # (B,)
        b_log_prob = b_dist.log_prob(b_idx)  # (B,)

        # Combined log probability
        log_prob = a_log_prob + b_log_prob
        
        return (a, b), log_prob, value

    def log_prob_value(
        self, state: torch.Tensor, action: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (B, state_dim) or (state_dim,)
            action: tuple of (a, b) where both are discrete values
        Returns:
            log_prob, value, entropy
        """
        a, b = action

        out = self.forward(state)
        a_logits = out["a_logits"]
        b_logits = out["b_logits"]
        value = out["value"]
        
        # Find closest a value index
        a_values_expanded = self.a_values.unsqueeze(0)  # (1, num_a_values)
        a_idx = torch.argmin(torch.abs(a.unsqueeze(-1) - a_values_expanded), dim=-1)  # (B,)
        
        # Find closest b value index
        b_values_expanded = self.b_values.unsqueeze(0)  # (1, num_b_values)
        b_idx = torch.argmin(torch.abs(b.unsqueeze(-1) - b_values_expanded), dim=-1)  # (B,)
        
        # Compute log prob for a
        a_dist = torch.distributions.Categorical(logits=a_logits)
        a_log_prob = a_dist.log_prob(a_idx)
        a_entropy = a_dist.entropy()

        # Compute log prob for b
        b_dist = torch.distributions.Categorical(logits=b_logits)
        b_log_prob = b_dist.log_prob(b_idx)
        b_entropy = b_dist.entropy()
        
        # Combined
        log_prob = a_log_prob + b_log_prob
        entropy = a_entropy + b_entropy

        return log_prob, value, entropy

    def ppo_update(
        self,
        buffer,
        config,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        PPO update for single-step episodes with (a, b) actions.
        - advantages = rewards - values (baseline)
        - returns    = rewards
        """
        self.train()

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
                log_prob, value, entropy = self.log_prob_value(state, action)

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
                torch.nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_bonus.item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }
