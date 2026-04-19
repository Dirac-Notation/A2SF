"""RWR (Reward-Weighted Regression) agent — policy-based RL for sigmoid KV cache.

Design rationale (observation-driven):

  1. Phase A-O2: rewards are smooth across action indices (nb-corr 0.83-0.87).
     → Expected-action decoding naturally exploits this smoothness.
  2. Phase A-O3: 63% of samples have best↔2nd-best reward gap < 0.02.
     → Hard classification overfits noise.  RWR's soft target  p* = softmax(r/T)
       degrades gracefully to uniform when rewards are close.
  3. Phase A-O6': Ranking of actions reshuffles drastically across budgets
     (ρ ≈ 0.11-0.21).  → Per-budget models.
  4. Phase B-O4: feature-to-best-a correlation *varies by task in sign*.
     → Keep full state (attention features + task + metric) so the MLP can
       learn task-specific interactions.

Architecture (shared encoder with NeuralUCB — proven to carry signal):

  state (131K)
    → head-merge + side-reduce to 256-d tova + 256-d snap
    → concat meta (seq_len + metric + task one-hot)
    → 2-layer MLP → 14 logits
  logits → softmax → π(a_k | s)

Training: reward-weighted cross-entropy (RWR)
     target p*_k = softmax(r_k / T)
     L = -Σ_k p*_k log π(a_k | s)

Inference: expected-action decoding (single forward pass)
     α = Σ_k π(a_k | s) · a_k        [continuous scalar, possibly outside 14-grid]
     b = 0  (fixed)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class RWRAgent(nn.Module):
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
        decoding_temperature: float = 1.0,
        decoding_mode: str = "expected",        # "expected" | "argmax"
        # legacy kwargs accepted for drop-in replacement
        backbone_depth: int = 2,
        dropout: float = 0.0,
        budget_slot: bool = False,
    ):
        super().__init__()
        del backbone_depth, dropout, budget_slot
        assert decoding_mode in ("expected", "argmax")

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
        self.metric_heads = metric_heads or ["qa_f1_score"]
        self.metric_name_to_idx = {n: i for i, n in enumerate(self.metric_heads)}
        self.decoding_temperature = float(decoding_temperature)
        self.decoding_mode = str(decoding_mode)

        self._meta_dim = 1 + self.num_metric_types + self.num_task_types

        # NeuralUCB-style attention embedding (proven to work)
        self.head_merge = nn.Parameter(torch.randn(self.num_bins, self.num_heads) * 0.01)
        self.side_reduce = nn.Linear(self.num_bins, 256)

        fuse_in = 256 + 256 + self._meta_dim
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
        fused = torch.cat([tova_emb, snap_emb, meta], dim=-1)
        return self.embed(fused)

    def forward(self, state: torch.Tensor,
                metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
                ) -> Dict[str, torch.Tensor]:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        h = self._embed_state(state)
        logits = self.head(h)                                # (B, num_actions)
        # Policy distribution
        probs = torch.softmax(logits / self.decoding_temperature, dim=-1)
        # Expose `reward_pred` for legacy interfaces — here it equals π(a|s)
        return {
            "reward_pred": probs,
            "logits": logits,
            "policy": probs,
            "feature_vector": h,
        }

    @torch.no_grad()
    def act(self, state: torch.Tensor,
            metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
            ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Expected-action decoding: α = Σ_k π(a_k|s) · a_k.

        Because a_values can live on a different device (agent's buffer) from the
        attention features computed by the first-layer encoder (on layer 0's GPU),
        keep everything on the policy's device.
        """
        out = self.forward(state, metric_type=metric_type)
        probs = out["policy"]
        if probs.ndim == 1:
            probs = probs.unsqueeze(0)
        B = probs.size(0)

        if self.decoding_mode == "argmax":
            idx = probs.argmax(dim=-1)                                  # (B,)
            a_out = self.a_values[idx]
        else:
            a_out = (probs * self.a_values.view(1, -1)).sum(dim=-1)     # (B,)

        b_val = torch.zeros(B, device=probs.device, dtype=probs.dtype)
        conf = probs.max(dim=-1).values
        return (a_out, b_val), conf

    # ── Legacy compatibility ──
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
