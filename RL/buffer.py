import torch
from typing import Tuple

class RolloutBuffer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []

    def add(self, state: torch.Tensor, action: torch.Tensor, log_prob: torch.Tensor, reward: torch.Tensor, value: torch.Tensor):
        self.states.append(state.to(self.device))
        self.actions.append(action.to(self.device))
        self.log_probs.append(log_prob.to(self.device))
        self.rewards.append(reward.to(self.device))
        self.values.append(value.to(self.device))

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.states) == 0:
            return (
                torch.tensor([], device=self.device),
                torch.tensor([], device=self.device),
                torch.tensor([], device=self.device),
                torch.tensor([], device=self.device),
                torch.tensor([], device=self.device),
            )

        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        log_probs = torch.stack(self.log_probs)
        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values)
        return states, actions, log_probs, rewards, values

    def size(self) -> int:
        return len(self.states)