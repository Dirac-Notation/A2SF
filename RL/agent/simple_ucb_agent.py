"""SimpleUCBAgent — minimal RL agent (per-task linear heads).

Architecture:
  state(D) -> Linear(D->H) + ReLU [+ Dropout] -> Linear(H->H) + ReLU [+ Dropout]   [trunk]
              -> per-task Linear(H->A)                                              [head]
              -> sigmoid                                                            [output]

Per-task UCB inverse covariance buffer (T tasks * A actions * H * H).
No head_merge / no LayerNorm / no per-metric heads / no task_cond residual.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

EPS = 1e-6


class SimpleUCBAgent(nn.Module):
    def __init__(
        self,
        state_dim: int,
        a_values: torch.Tensor,
        b_values: torch.Tensor,
        num_metric_types: int = 0,
        num_task_types: int = 0,
        hidden: int = 256,
        dropout: float = 0.0,
        paired_actions: bool = False,
        include_seq_len: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.feature_dim = int(hidden)
        self.num_metric_types = int(num_metric_types)
        self.num_task_types = int(num_task_types)
        self.include_seq_len = bool(include_seq_len)
        self.paired_actions = bool(paired_actions)

        self.register_buffer("a_values", a_values)
        self.register_buffer("b_values", b_values)
        self.num_a_values = int(len(a_values))
        self.num_b_values = int(len(b_values))
        self.num_actions = self.num_a_values if self.paired_actions else (self.num_a_values * self.num_b_values)

        layers: List[nn.Module] = [nn.Linear(self.state_dim, self.feature_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)

        if self.num_task_types <= 0:
            raise ValueError("SimpleUCBAgent requires num_task_types > 0 (per-task head)")
        self.task_heads = nn.ModuleList([
            nn.Linear(self.feature_dim, self.num_actions) for _ in range(self.num_task_types)
        ])

        self.lambda_reg = 5 * self.num_actions
        inv0 = torch.eye(self.feature_dim) / self.lambda_reg
        self.register_buffer(
            "inverse_lambdas",
            inv0.unsqueeze(0).unsqueeze(0).repeat(self.num_task_types, self.num_actions, 1, 1).contiguous(),
        )
        self.register_buffer(
            "action_counts",
            torch.zeros(self.num_task_types, self.num_actions, dtype=torch.long),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def _task_idx_from_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        task_oh_start = (1 if self.include_seq_len else 0) + self.num_metric_types
        task_oh = state[:, task_oh_start:task_oh_start + self.num_task_types]
        return task_oh.argmax(dim=-1).long()

    def forward(
        self,
        state: torch.Tensor,
        metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
    ) -> Dict[str, torch.Tensor]:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.size(-1) != self.state_dim:
            raise ValueError(f"Expected state last dim {self.state_dim}, got {state.size(-1)}")

        task_idx = self._task_idx_from_state(state)
        h = self.trunk(state.to(dtype=torch.float32))

        logits = torch.empty(state.size(0), self.num_actions, device=h.device, dtype=h.dtype)
        for t in task_idx.unique().tolist():
            mask = (task_idx == t).nonzero(as_tuple=True)[0]
            logits[mask] = self.task_heads[t](h[mask])

        reward_pred = torch.sigmoid(logits)
        return {"reward_pred": reward_pred, "feature_vector": h, "task_idx": task_idx}

    @torch.no_grad()
    def act(
        self,
        state: torch.Tensor,
        metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        out = self.forward(state, metric_type=metric_type)
        reward_pred = out["reward_pred"]
        action_idx = reward_pred.argmax(dim=-1)
        a_val = self.a_values[action_idx]
        b_val = self.b_values[action_idx]
        batch_indices = torch.arange(action_idx.size(0), device=reward_pred.device)
        return (a_val, b_val), reward_pred[batch_indices, action_idx]
