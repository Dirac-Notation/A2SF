# buffer.py
import torch
from typing import Tuple

class NeuralUCBBuffer:
    """
    Buffer for NeuralUCB algorithm.
    Stores independent (state, action, reward) tuples.
    Actions are tuples of (a, b) for sigmoid cache.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.clear()

    def clear(self):
        self.states = []
        self.actions_a = []  # a values
        self.actions_b = []  # b values
        self.rewards = []

    def add(
        self,
        state: torch.Tensor,
        action: Tuple[torch.Tensor, torch.Tensor],
        reward: torch.Tensor,
    ):
        a, b = action
        self.states.append(state.to(self.device))
        self.actions_a.append(a.to(self.device))
        self.actions_b.append(b.to(self.device))
        self.rewards.append(reward.to(self.device))

    def get(self) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns:
            states:   (N, state_dim)
            actions:  tuple of (a, b) where each is (N,)
            rewards:  (N, 1)
        """
        states = torch.stack(self.states)
        actions = (torch.stack(self.actions_a), torch.stack(self.actions_b))
        rewards = torch.stack(self.rewards).unsqueeze(-1)
        return states, actions, rewards

    def size(self) -> int:
        return len(self.states)