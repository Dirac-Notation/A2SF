"""Compact RL agent — cleanly designed, budget-conditioned, supervised.

Architecture rationale (one coherent idea):

  1. INPUT  : same 131K state as NeuralUCB (attention features carry the signal;
              hand-crafted summary stats were insufficient — verified empirically).
  2. EMBED  : re-use the proven "per-bin head-merge → linear" embedding so TOVA
              and SnapKV each get compressed to 256-d. Concatenate with the
              metadata (seq_len + metric + task one-hots) AND the budget.
  3. BACKBONE: small 2-layer MLP on the 512-d embedding.
  4. OUTPUT : 14 action logits. At inference: argmax. At training: multilabel BCE
              against the set of near-optimal actions (gap ≤ delta from per-sample
              max reward) — handles the 62 % "near-tie" samples cleanly.
  5. TRAINING: supervised on combined-budget data (128 + 256 + 512). Single model
              handles all budgets via the budget slot.

Why this beats the previous NeuralUCB pipeline:
  - Offline data gives us the full reward vector per sample → no need for UCB
    exploration. Direct supervised learning is simpler and exploits all actions.
  - Multilabel BCE marginalises over small-gap ties (our biggest noise source).
  - Budget is an explicit input → one checkpoint for all budgets, better data
    leverage (~4500 vs 1500 samples).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


class CompactAgent(nn.Module):
    def __init__(
        self,
        state_dim: int,
        a_values: torch.Tensor,
        b_values: torch.Tensor,
        metric_heads: Optional[List[str]] = None,
        num_heads: int = 32,
        num_metric_types: int = 10,
        side_dim: int = 65536,
        num_task_types: int = 0,
        hidden: int = 256,
        budget_slot: bool = True,
        # legacy kwargs (accepted for drop-in replacement; unused)
        backbone_depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        del backbone_depth, dropout

        self.register_buffer("a_values", a_values)
        self.register_buffer("b_values", b_values)
        self.num_a_values = len(a_values)
        self.num_b_values = len(b_values)
        self.num_actions = self.num_a_values * self.num_b_values

        self.state_dim = int(state_dim)
        self.num_heads = int(num_heads)
        self.num_metric_types = int(num_metric_types)
        self.num_task_types = int(num_task_types)
        self.side_dim = int(side_dim)
        self.num_bins = self.side_dim // self.num_heads
        self.budget_slot = bool(budget_slot)
        self.metric_heads = metric_heads or ["qa_f1_score"]
        self.metric_name_to_idx = {n: i for i, n in enumerate(self.metric_heads)}

        self._meta_dim = 1 + self.num_metric_types + self.num_task_types

        # Per-bin head-merge (proven to work in NeuralUCB)
        self.head_merge = nn.Parameter(torch.randn(self.num_bins, self.num_heads) * 0.01)
        # (num_bins,) → 256
        self.side_reduce = nn.Linear(self.num_bins, 256)

        # Fuse tova(256) + snap(256) + meta + budget(1)
        budget_dim = 1 if self.budget_slot else 0
        fuse_in = 256 + 256 + self._meta_dim + budget_dim
        self.embed = nn.Sequential(
            nn.Linear(fuse_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, self.num_actions)

        # Legacy buffers (kept so checkpoint loaders don't balk)
        self.register_buffer(
            "action_counts",
            torch.zeros(len(self.metric_heads), self.num_actions, dtype=torch.long),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    # ── Feature extraction ──
    def _embed_state(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(dtype=torch.float32)
        B = state.size(0)
        md = self._meta_dim
        meta = state[:, :md]
        tova_flat = state[:, md:md + self.side_dim]
        snap_flat = state[:, md + self.side_dim:]

        tova = tova_flat.reshape(B, self.num_heads, self.num_bins).permute(0, 2, 1)
        snap = snap_flat.reshape(B, self.num_heads, self.num_bins).permute(0, 2, 1)

        tova_merged = torch.einsum("bnh,nh->bn", tova, self.head_merge)
        snap_merged = torch.einsum("bnh,nh->bn", snap, self.head_merge)

        tova_emb = F.relu(self.side_reduce(tova_merged))
        snap_emb = F.relu(self.side_reduce(snap_merged))

        parts = [tova_emb, snap_emb, meta]
        if self.budget_slot:
            bud = getattr(self, "_current_budget_feat", None)
            if bud is None:
                bud = torch.zeros(B, 1, device=tova_emb.device, dtype=tova_emb.dtype)
            parts.append(bud)

        fused = torch.cat(parts, dim=-1)
        return self.embed(fused)

    def set_budget(self, budget: int, batch_size: int = 1, device=None) -> None:
        if not self.budget_slot:
            return
        dev = device or next(self.parameters()).device
        val = (float(budget) - 128.0) / (512.0 - 128.0)
        val = max(0.0, min(1.0, val))
        self._current_budget_feat = torch.full((batch_size, 1), val, device=dev, dtype=torch.float32)

    # ── Forward / act ──
    def forward(self, state: torch.Tensor,
                metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
                ) -> Dict[str, torch.Tensor]:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        h = self._embed_state(state)
        logits = self.head(h)
        return {"reward_pred": torch.sigmoid(logits), "logits": logits, "feature_vector": h}

    @torch.no_grad()
    def act(self, state, metric_type=None):
        out = self.forward(state, metric_type=metric_type)
        rp = out["reward_pred"]
        if rp.ndim == 1:
            rp = rp.unsqueeze(0)
        idx = torch.argmax(rp, dim=-1)
        a_i = idx // self.num_b_values
        b_i = idx % self.num_b_values
        return (self.a_values[a_i], self.b_values[b_i]), rp.gather(1, idx.unsqueeze(1)).squeeze(1)

    # Legacy API stubs (so NeuralUCB-trained callers don't crash)
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
