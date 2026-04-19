"""RWRv3 agent — v2 + per-head feature preservation (no head_merge bottleneck).

Key change over v2:
  Instead of `head_merge` (num_bins × num_heads param → 1 scalar per bin),
  we apply a shared `bin_reduce` (Linear num_bins → hidden_per_head) to each
  head independently, then flatten. This preserves per-head structure.

  v2 pipeline:
    tova (B, 2048, 32) → einsum head_merge → (B, 2048) → Linear(2048, 256) → 256
  v3 pipeline:
    tova (B, 32, 2048) → Linear(2048, hidden_per_head=8) per head → (B, 32, 8) → flatten → 256

Same final dim (256) but per-head info preserved.  Phase C analysis showed
heads have distinct contributions (Gini ~0.3+); head_merge was erasing that.

Keeps all v2 features: task/metric embeddings, task-conditional head.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class RWRv3Agent(nn.Module):
    def __init__(
        self,
        state_dim: int,
        a_values: torch.Tensor,
        b_values: torch.Tensor,
        metric_heads: Optional[List[str]] = None,
        num_heads: int = 32,
        num_metric_types: int = 10,
        side_dim: int = 65536,
        num_task_types: int = 7,
        hidden: int = 256,
        hidden_per_head: int = 8,
        task_emb_dim: int = 32,
        metric_emb_dim: int = 32,
        decoding_temperature: float = 1.0,
        decoding_mode: str = "argmax",
        dropout: float = 0.0,
        backbone_depth: int = 2,
        budget_slot: bool = False,
    ):
        super().__init__()
        del backbone_depth, budget_slot
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
        self.decoding_temperature = float(decoding_temperature)
        self.decoding_mode = str(decoding_mode)
        self.task_emb_dim = int(task_emb_dim)
        self.metric_emb_dim = int(metric_emb_dim)
        self.hidden_per_head = int(hidden_per_head)

        self._meta_dim_state = 1 + self.num_metric_types + self.num_task_types

        # Learned embeddings (v2 inherited)
        self.task_embed = nn.Embedding(max(1, self.num_task_types), self.task_emb_dim)
        self.metric_embed = nn.Embedding(max(1, self.num_metric_types), self.metric_emb_dim)
        nn.init.normal_(self.task_embed.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.metric_embed.weight, mean=0.0, std=0.1)

        # ── NEW: per-head bin reduce (shared weights, independent application) ──
        self.bin_reduce = nn.Linear(self.num_bins, self.hidden_per_head)

        per_view_dim = self.num_heads * self.hidden_per_head     # 32 × 8 = 256
        # Fused input dim: 2 views + seq_len + task_emb + metric_emb
        fuse_in = 2 * per_view_dim + 1 + self.task_emb_dim + self.metric_emb_dim
        self.embed = nn.Sequential(
            nn.Linear(fuse_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Task-conditional head
        self.task_head = nn.Linear(hidden, self.num_task_types * self.num_actions)

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

    def _split_state(self, state: torch.Tensor):
        state = state.to(dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        B = state.size(0)
        seq_len_feat = state[:, 0:1]
        metric_oh = state[:, 1:1 + self.num_metric_types]
        task_oh = state[:, 1 + self.num_metric_types:
                         1 + self.num_metric_types + self.num_task_types]
        atten_start = 1 + self.num_metric_types + self.num_task_types
        tova_flat = state[:, atten_start:atten_start + self.side_dim]
        snap_flat = state[:, atten_start + self.side_dim:]
        return seq_len_feat, metric_oh, task_oh, tova_flat, snap_flat, B

    def _encode_per_head(self, view_flat, B):
        """view_flat: (B, H * num_bins) → (B, H, num_bins) → Linear → flatten (B, H*hidden_per_head)."""
        x = view_flat.reshape(B, self.num_heads, self.num_bins)
        x = self.bin_reduce(x)              # (B, H, hidden_per_head)
        x = F.relu(x)
        return x.reshape(B, -1)              # (B, H*hidden_per_head)

    def _embed_state(self, state):
        seq_len_feat, metric_oh, task_oh, tova_flat, snap_flat, B = self._split_state(state)
        tova_emb = self._encode_per_head(tova_flat, B)
        snap_emb = self._encode_per_head(snap_flat, B)
        task_idx = task_oh.argmax(dim=-1)
        metric_idx = metric_oh.argmax(dim=-1)
        task_emb = self.task_embed(task_idx)
        metric_emb = self.metric_embed(metric_idx)
        fused = torch.cat([tova_emb, snap_emb, seq_len_feat, task_emb, metric_emb], dim=-1)
        h = self.embed(fused)
        return h, task_idx

    def forward(self, state, metric_type=None):
        h, task_idx = self._embed_state(state)
        all_logits = self.task_head(h).reshape(-1, self.num_task_types, self.num_actions)
        logits = all_logits.gather(1, task_idx.view(-1, 1, 1).expand(-1, 1, self.num_actions)).squeeze(1)
        probs = torch.softmax(logits / self.decoding_temperature, dim=-1)
        return {"reward_pred": probs, "logits": logits, "policy": probs, "feature_vector": h}

    @torch.no_grad()
    def act(self, state, metric_type=None):
        out = self.forward(state, metric_type=metric_type)
        probs = out["policy"]
        if probs.ndim == 1:
            probs = probs.unsqueeze(0)
        B = probs.size(0)
        if self.decoding_mode == "argmax":
            idx = probs.argmax(dim=-1)
            a_out = self.a_values[idx]
        else:
            a_out = (probs * self.a_values.view(1, -1)).sum(dim=-1)
        b_val = torch.zeros(B, device=probs.device, dtype=probs.dtype)
        conf = probs.max(dim=-1).values
        return (a_out, b_val), conf

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


__all__ = ["RWRv3Agent"]
