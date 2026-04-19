"""Heuristic agent — fixed per-(task, metric) action lookup.

Purpose: a baseline that ignores attention features entirely. For every
(task_type, metric_type) pair, it picks the action that maximizes the MEAN
reward across training samples with that combination. This gives the
"metadata-only upper bound" — the best achievable without state-level signals.

The lookup table is precomputed from offline training data and stored as a
buffer `action_table[task_idx, metric_idx] -> action_idx` on the agent.

At inference, we extract (task_idx, metric_idx) from the state's meta section
(one-hot encoded) and emit a = a_values[table[task_idx, metric_idx]].

Interface matches RWRAgent / RWRFlatAgent so the same dispatcher can load it.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class HeuristicAgent(nn.Module):
    def __init__(
        self,
        state_dim: int,
        a_values: torch.Tensor,
        b_values: torch.Tensor,
        action_table: torch.Tensor,          # (num_task_types, num_metric_types) int64
        metric_heads: Optional[List[str]] = None,
        num_heads: int = 32,
        num_metric_types: int = 10,
        side_dim: int = 65536,
        num_task_types: int = 0,
        hidden: int = 256,                   # kept for signature compat; unused
        # legacy kwargs accepted
        decoding_temperature: float = 1.0,
        decoding_mode: str = "argmax",
        backbone_depth: int = 2,
        dropout: float = 0.0,
        budget_slot: bool = False,
    ):
        super().__init__()
        del hidden, decoding_temperature, decoding_mode, backbone_depth, dropout, budget_slot

        self.register_buffer("a_values", a_values)
        self.register_buffer("b_values", b_values)
        self.register_buffer("action_table", action_table.long())  # (T, M)
        self.num_a_values = len(a_values)
        self.num_b_values = len(b_values)
        self.num_actions = self.num_a_values * self.num_b_values

        self.state_dim = int(state_dim)
        self.num_heads = int(num_heads)
        self.num_metric_types = int(num_metric_types)
        self.num_task_types = int(num_task_types)
        self.side_dim = int(side_dim)
        self.metric_heads = metric_heads or ["qa_f1_score"]

        # Legacy buffer for format compat
        self.register_buffer(
            "action_counts",
            torch.zeros(len(self.metric_heads), self.num_actions, dtype=torch.long),
        )

    def _extract_task_metric_idx(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """State layout: [seq_len(1) | metric_one_hot(M) | task_one_hot(T) | ...]"""
        state = state.to(dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        metric_oh = state[:, 1:1 + self.num_metric_types]
        task_oh = state[:, 1 + self.num_metric_types:1 + self.num_metric_types + self.num_task_types]
        metric_idx = metric_oh.argmax(dim=-1)
        task_idx = task_oh.argmax(dim=-1)
        return task_idx, metric_idx

    def forward(self, state: torch.Tensor,
                metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
                ) -> Dict[str, torch.Tensor]:
        task_idx, metric_idx = self._extract_task_metric_idx(state)
        action_idx = self.action_table[task_idx, metric_idx]              # (B,)
        B = action_idx.size(0)
        # Emit a one-hot "policy" matching the selected action
        probs = torch.zeros(B, self.num_actions, device=state.device)
        probs.scatter_(1, action_idx.unsqueeze(1), 1.0)
        return {
            "reward_pred": probs,
            "logits": probs.log().clamp_min(-1e6),
            "policy": probs,
            "feature_vector": torch.zeros(B, 1, device=state.device),
        }

    @torch.no_grad()
    def act(self, state: torch.Tensor,
            metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
            ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        task_idx, metric_idx = self._extract_task_metric_idx(state)
        action_idx = self.action_table[task_idx, metric_idx].clamp(0, self.num_actions - 1)
        B = action_idx.size(0)
        a_i = action_idx // self.num_b_values
        b_i = action_idx % self.num_b_values
        a_out = self.a_values[a_i]
        b_out = self.b_values[b_i]
        conf = torch.ones(B, device=state.device)
        return (a_out, b_out), conf

    # ── Legacy compat stubs ──
    def _get_action_indices(self, action):
        a, b = action
        if a.ndim == 0: a = a.unsqueeze(0)
        if b.ndim == 0: b = b.unsqueeze(0)
        ai = torch.argmin(torch.abs(a.unsqueeze(-1) - self.a_values.unsqueeze(0)), dim=-1)
        bi = torch.argmin(torch.abs(b.unsqueeze(-1) - self.b_values.unsqueeze(0)), dim=-1)
        return ai * self.num_b_values + bi

    def predict_reward(self, state, action, metric_type=None):
        out = self.forward(state, metric_type=metric_type)
        idx = self._get_action_indices(action)
        return out["reward_pred"].gather(1, idx.unsqueeze(1)).squeeze(1)

    def _update_inverse_covariances(self, *a, **k):
        return

    def _compute_ucb_scores(self, state, beta, metric_type=None):
        out = self.forward(state, metric_type=metric_type)
        rp = out["reward_pred"]
        return rp, rp


__all__ = ["HeuristicAgent"]
