import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

EPS = 1e-6

class A2SFPolicy(nn.Module):
    def __init__(self, state_dim: int, action_min: float = 0.0, action_max: float = 1.0):
        super().__init__()
        self.action_min = action_min
        self.action_max = action_max

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )

        self.mu_head = nn.Linear(512, 1)
        self.kappa_head = nn.Linear(512, 1)
        self.value_head = nn.Linear(512, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def _alpha_beta(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = torch.sigmoid(self.mu_head(h))
        kappa = F.softplus(self.kappa_head(h)) + 2.0
        alpha = (mu * kappa).clamp_min(EPS).clamp_max(100.0)
        beta = ((1.0 - mu) * kappa).clamp_min(EPS).clamp_max(100.0)
        return alpha.squeeze(-1), beta.squeeze(-1)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        h = self.backbone(state)
        alpha, beta = self._alpha_beta(h)
        value = self.value_head(h).squeeze(-1)
        return {"alpha": alpha, "beta": beta, "value": value}

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.forward(state)
        alpha, beta, value = out["alpha"], out["beta"], out["value"]

        dist = torch.distributions.Beta(alpha, beta)
        a = dist.sample().clamp(EPS, 1 - EPS)
        action = self.action_min + a * (self.action_max - self.action_min)
        log_prob = dist.log_prob(a)
        return action.squeeze(-1), log_prob.squeeze(-1), value.squeeze(-1)

    def log_prob_value(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        out = self.forward(state)
        alpha, beta, value = out["alpha"], out["beta"], out["value"]

        a01 = (action - self.action_min) / (self.action_max - self.action_min)
        a01 = a01.clamp(EPS, 1 - EPS)

        dist = torch.distributions.Beta(alpha, beta)
        log_prob = dist.log_prob(a01)
        entropy = dist.entropy()
        return log_prob.squeeze(-1), value.squeeze(-1), entropy.squeeze(-1)

@torch.no_grad()
def _normalize_adv(adv: torch.Tensor) -> torch.Tensor:
    if adv.numel() <= 1:
        return adv
    
    adv_mean = adv.mean()
    adv_std = adv.std()
    
    if adv_std < 1e-6:
        return adv - adv_mean
    
    return (adv - adv_mean) / (adv_std + 1e-8)

def ppo_update(policy: A2SFPolicy, buffer, config, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    policy.train()

    states, actions, old_log_probs, rewards, old_values = buffer.get()
    if states.numel() == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    advantages = (rewards - old_values).detach()
    returns = rewards.detach()
    advantages = _normalize_adv(advantages)

    batch_size = states.size(0)
    idx = torch.randperm(batch_size, device=states.device)
    states, actions = states[idx], actions[idx]
    old_log_probs, returns, advantages = old_log_probs[idx], returns[idx], advantages[idx]

    policy_losses, value_losses, entropies = [], [], []
    effective_minibatch_size = min(config.minibatch_size, batch_size)
    
    for _ in range(config.update_epochs):
        for start in range(0, batch_size, effective_minibatch_size):
            end = start + effective_minibatch_size
            bs = states[start:end]
            ba = actions[start:end]
            bold_lp = old_log_probs[start:end]
            bret = returns[start:end]
            badv = advantages[start:end]

            lp, v, ent = policy.log_prob_value(bs, ba)

            ratio = torch.exp(lp - bold_lp).clamp(1e-8, 1e8)
            surr1 = ratio * badv
            surr2 = torch.clamp(ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip) * badv
            policy_loss = -torch.min(surr1, surr2).mean()

            if torch.isnan(policy_loss):
                policy_loss = torch.tensor(0.0, device=policy_loss.device)

            value_loss = F.mse_loss(v, bret)
            if torch.isnan(value_loss):
                value_loss = torch.tensor(0.0, device=value_loss.device)

            entropy_bonus = ent.mean()
            if torch.isnan(entropy_bonus):
                entropy_bonus = torch.tensor(0.0, device=entropy_bonus.device)

            total_loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_bonus

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