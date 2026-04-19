"""RWRv2 agent — learned task/metric embeddings + task-conditional heads.

Changes over RWRAgent:
  (A) Task and metric one-hots are replaced by learned embeddings
      (Embedding(num_task_types, task_emb_dim) etc.). Signal strength
      scales with embedding dimension, not attention-feature dim ratio.
  (B) Output head is conditioned on task: a set of per-task heads
      { head_t: Linear(hidden, num_actions) }_{t ∈ tasks}, selected by
      the input's task index at forward time.

State layout is unchanged (encoder still emits one-hot); we derive indices
inside the agent by argmax of the one-hot slots.

Training loss can pair with a per-sample adaptive target temperature (computed
outside in rwr_v2_train.py).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class RWRv2Agent(nn.Module):
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
        task_emb_dim: int = 32,
        metric_emb_dim: int = 32,
        decoding_temperature: float = 1.0,
        decoding_mode: str = "argmax",
        dropout: float = 0.0,
        # legacy kwargs accepted
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

        # State meta layout (from encoder):
        #   [seq_len(1), metric_one_hot(M), task_one_hot(T), tova, snap]
        self._meta_dim_state = 1 + self.num_metric_types + self.num_task_types

        # Learned embeddings
        self.task_embed = nn.Embedding(max(1, self.num_task_types), self.task_emb_dim)
        self.metric_embed = nn.Embedding(max(1, self.num_metric_types), self.metric_emb_dim)
        nn.init.normal_(self.task_embed.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.metric_embed.weight, mean=0.0, std=0.1)

        # Attention feature reduction (same as RWRAgent)
        self.head_merge = nn.Parameter(torch.randn(self.num_bins, self.num_heads) * 0.01)
        self.side_reduce = nn.Linear(self.num_bins, 256)

        # Fused input dim: 256 (tova) + 256 (snap) + 1 (seq_len) + task_emb + metric_emb
        fuse_in = 256 + 256 + 1 + self.task_emb_dim + self.metric_emb_dim
        self.embed = nn.Sequential(
            nn.Linear(fuse_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Per-task output head.  Represented as a single big Linear with
        # num_task_types * num_actions outputs, then we gather per sample.
        self.task_head = nn.Linear(hidden, self.num_task_types * self.num_actions)

        # Legacy buffers
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
        seq_len_feat = state[:, 0:1]                                                     # (B, 1)
        metric_oh = state[:, 1:1 + self.num_metric_types]                                # (B, M)
        task_oh = state[:, 1 + self.num_metric_types:
                         1 + self.num_metric_types + self.num_task_types]                # (B, T)
        atten_start = 1 + self.num_metric_types + self.num_task_types
        tova_flat = state[:, atten_start:atten_start + self.side_dim]
        snap_flat = state[:, atten_start + self.side_dim:]
        return seq_len_feat, metric_oh, task_oh, tova_flat, snap_flat, B

    def _encode_attention(self, tova_flat, snap_flat, B):
        tova = tova_flat.reshape(B, self.num_heads, self.num_bins).permute(0, 2, 1)
        snap = snap_flat.reshape(B, self.num_heads, self.num_bins).permute(0, 2, 1)
        tova_merged = torch.einsum("bnh,nh->bn", tova, self.head_merge)
        snap_merged = torch.einsum("bnh,nh->bn", snap, self.head_merge)
        tova_emb = F.relu(self.side_reduce(tova_merged))
        snap_emb = F.relu(self.side_reduce(snap_merged))
        return tova_emb, snap_emb

    def _embed_state(self, state: torch.Tensor):
        seq_len_feat, metric_oh, task_oh, tova_flat, snap_flat, B = self._split_state(state)
        tova_emb, snap_emb = self._encode_attention(tova_flat, snap_flat, B)

        task_idx = task_oh.argmax(dim=-1)                                                # (B,)
        metric_idx = metric_oh.argmax(dim=-1)                                            # (B,)
        task_emb = self.task_embed(task_idx)                                             # (B, Et)
        metric_emb = self.metric_embed(metric_idx)                                       # (B, Em)

        fused = torch.cat([tova_emb, snap_emb, seq_len_feat, task_emb, metric_emb], dim=-1)
        h = self.embed(fused)                                                            # (B, hidden)
        return h, task_idx

    def forward(self, state: torch.Tensor,
                metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
                ) -> Dict[str, torch.Tensor]:
        h, task_idx = self._embed_state(state)
        # All-task output: (B, num_task_types * num_actions) → (B, T, A)
        all_logits = self.task_head(h).reshape(-1, self.num_task_types, self.num_actions)
        logits = all_logits.gather(1, task_idx.view(-1, 1, 1).expand(-1, 1, self.num_actions)).squeeze(1)
        probs = torch.softmax(logits / self.decoding_temperature, dim=-1)
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


__all__ = ["RWRv2Agent"]
