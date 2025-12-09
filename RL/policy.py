# policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Dict, Tuple

EPS = 1e-6

# b values: discrete set for b parameter
B_VALUES = torch.tensor([1, 2, 4, 6, 8, 12, 16, 20, 32, 48, 64, 96, 128, 512, 1024, 2048, 4096, 8192])

class A2SFPolicy(nn.Module):
    """
    Policy network for A2SF RL agent (single-step / bandit) with sigmoid cache
    - a: continuous action in [0, 10] using Beta distribution (0~1 * 10)
    - b: discrete action from predefined set using Categorical distribution
    - Value head as baseline
    """

    def __init__(self, state_dim: int, a_values: torch.Tensor = None, b_values: torch.Tensor = None):
        super().__init__()
        if b_values is None:
            b_values = B_VALUES
        self.register_buffer('b_values', b_values)
        self.num_b_values = len(b_values)

        # Backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )

        # Policy heads
        # a: Beta(alpha_a, beta_a) -> outputs in [0, 1] then * 10 -> [0, 10]
        self.a_head = nn.Linear(512, 2)
        
        # b: discrete (Categorical over b_values)
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
            dict with a_params, b_logits, value
            a_params: (B, 2) - [alpha_a, beta_a]
            b_logits: (B, num_b_values) - logits for Categorical distribution
        """
        h = self.backbone(state)
        a_params_raw = self.a_head(h)  # (B, 2)
        b_logits = self.b_head(h)  # (B, num_b_values)
        
        # Transform to positive values for Beta distribution parameters
        # Using softplus + 1 to ensure alpha, beta >= 1 (for numerical stability)
        a_params = F.relu(a_params_raw) + 1.0  # (B, 2): [alpha_a, beta_a]
        
        value = self.value_head(h)
        value = nn.functional.sigmoid(value)
        return {"a_params": a_params, "b_logits": b_logits, "value": value}

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Sample action (single-step)
        Returns:
            action: tuple of (a, b) where a is in [0, 10] and b is from discrete set
            log_prob: tuple of (a_log_prob, b_log_prob)
            value: state value
        """
        out = self.forward(state)
        a_params = out["a_params"]  # (B, 2): [alpha_a, beta_a]
        b_logits = out["b_logits"]  # (B, num_b_values)
        value = out["value"]

        if a_params.ndim == 1:
            a_params = a_params.unsqueeze(0)
        if b_logits.ndim == 1:
            b_logits = b_logits.unsqueeze(0)

        # Sample a from Beta distribution, then scale to [0, 10]
        alpha_a = a_params[:, 0]  # (B,)
        beta_a = a_params[:, 1]    # (B,)
        a_dist = torch.distributions.Beta(alpha_a, beta_a)
        a_normalized = a_dist.sample()  # (B,) in [0, 1]
        a = a_normalized * 10.0  # (B,) in [0, 10]
        a_log_prob = a_dist.log_prob(a_normalized)  # (B,)
        
        # Sample b from Categorical distribution
        b_dist = torch.distributions.Categorical(logits=b_logits)
        b_idx = b_dist.sample()  # (B,)
        b = self.b_values[b_idx]  # (B,)
        b_log_prob = b_dist.log_prob(b_idx)  # (B,)
        
        return (a, b), (a_log_prob, b_log_prob), value

    def log_prob_value(
        self, state: torch.Tensor, action: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (B, state_dim) or (state_dim,)
            action: tuple of (a, b) where a is in [0, 10] and b is from discrete set
        Returns:
            log_prob: tuple of (a_log_prob, b_log_prob)
            value, entropy
        """
        a, b = action

        out = self.forward(state)
        a_params = out["a_params"]  # (B, 2): [alpha_a, beta_a]
        b_logits = out["b_logits"]  # (B, num_b_values)
        value = out["value"]
        
        # Normalize a from [0, 10] to [0, 1] for Beta distribution
        a_normalized = a / 10.0
        a_normalized = torch.clamp(a_normalized, EPS, 1.0 - EPS)
        
        # Compute log prob for a using Beta distribution
        alpha_a = a_params[:, 0]  # (B,)
        beta_a = a_params[:, 1]    # (B,)
        a_dist = torch.distributions.Beta(alpha_a, beta_a)
        a_log_prob = a_dist.log_prob(a_normalized)  # (B,)
        a_entropy = a_dist.entropy()     # (B,)
        
        # Find closest b value index
        b_values_expanded = self.b_values.unsqueeze(0)  # (1, num_b_values)
        b_idx = torch.argmin(torch.abs(b.unsqueeze(-1) - b_values_expanded), dim=-1)  # (B,)
        
        # Compute log prob for b using Categorical distribution
        b_dist = torch.distributions.Categorical(logits=b_logits)
        b_log_prob = b_dist.log_prob(b_idx)  # (B,)
        b_entropy = b_dist.entropy()     # (B,)
        
        # Combined entropy
        entropy = (a_entropy + b_entropy) / 2

        return (a_log_prob, b_log_prob), value, entropy

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
        old_a_log_probs, old_b_log_probs = old_log_probs
        
        # Single-step: returns = rewards (no discounting)
        returns = rewards.squeeze(-1)  # (N,)
        
        # Single-step advantages
        advantages = (returns - old_values.squeeze(-1)).detach()
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

        # Shuffle
        batch_size = states.size(0)
        idx = torch.randperm(batch_size, device=states.device)
        states = states[idx]
        actions = (actions[0][idx], actions[1][idx])  # Unpack tuple, shuffle, repack
        old_a_log_probs = old_a_log_probs[idx]
        old_b_log_probs = old_b_log_probs[idx]
        returns = returns[idx]
        advantages = advantages[idx]

        policy_losses, policy_losses_a, policy_losses_b, value_losses, entropies = [], [], [], [], []

        for _ in range(config.update_epochs):
            for start in range(0, batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                
                state = states[start:end]
                action = (actions[0][start:end], actions[1][start:end])  # Slice tuple
                old_a_log_prob = old_a_log_probs[start:end]
                old_b_log_prob = old_b_log_probs[start:end]
                return_batch = returns[start:end]
                advantage = advantages[start:end]

                # Recompute under current policy
                (a_log_prob, b_log_prob), value, entropy = self.log_prob_value(state, action)
                value = value.squeeze(-1)  # (B,)

                # PPO ratios for a and b separately
                ratio_a = torch.exp(a_log_prob - old_a_log_prob)
                ratio_b = torch.exp(b_log_prob - old_b_log_prob)
                
                # PPO clipped objective: min(ratio * advantage, clip(ratio) * advantage)
                surr_a = ratio_a * advantage
                clipped_surr_a = torch.clamp(ratio_a, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip) * advantage
                policy_loss_a = -torch.min(surr_a, clipped_surr_a).mean()
                
                surr_b = ratio_b * advantage
                clipped_surr_b = torch.clamp(ratio_b, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip) * advantage
                policy_loss_b = -torch.min(surr_b, clipped_surr_b).mean()
                
                # Policy loss is sum of losses for a and b
                policy_loss = policy_loss_a + policy_loss_b

                # Value loss (MSE to returns)
                value_loss = F.mse_loss(value, return_batch)

                # Entropy bonus
                entropy_bonus = entropy.mean()

                total_loss = (policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_bonus)

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
                optimizer.step()

                policy_losses.append(policy_loss.item())
                policy_losses_a.append(policy_loss_a.item())
                policy_losses_b.append(policy_loss_b.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_bonus.item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "policy_loss_a": float(np.mean(policy_losses_a)),
            "policy_loss_b": float(np.mean(policy_losses_b)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }
