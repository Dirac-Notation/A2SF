# buffer.py
import torch
from typing import Tuple

class RolloutBuffer:
    """
    Single-step (bandit) buffer for PPO/REINFORCE.
    Stores independent (s, a, logp, r, v).
    Actions are now tuples of (a, b) for sigmoid cache.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.clear()

    def clear(self):
        self.states = []
        self.actions_a = []  # a values
        self.actions_b = []  # b values
        self.log_probs = []
        self.rewards = []
        self.values = []

    def add(
        self,
        state: torch.Tensor,
        action: Tuple[torch.Tensor, torch.Tensor],
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
    ):
        a, b = action
        self.states.append(state.to(self.device))
        self.actions_a.append(a.to(self.device))
        self.actions_b.append(b.to(self.device))
        self.log_probs.append(log_prob.to(self.device))
        self.rewards.append(reward.to(self.device))
        self.values.append(value.to(self.device))

    def get(self) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            states:   (N, state_dim)
            actions:  tuple of (a, b) where each is (N,)
            log_probs:(N,)
            rewards:  (N,)
            values:   (N,)
        """
        states = torch.stack(self.states)
        actions = (torch.stack(self.actions_a), torch.stack(self.actions_b))
        log_probs = torch.stack(self.log_probs)
        rewards = torch.stack(self.rewards).unsqueeze(-1)
        values = torch.stack(self.values)
        return states, actions, log_probs, rewards, values

    def size(self) -> int:
        return len(self.states)