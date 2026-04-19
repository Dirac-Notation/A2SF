"""Binary-decision agent for sigmoid-cache RL.

Observation-driven design (see Phase A/B/C observations):

  * Reward vs action-index is NOT unimodal (only 5-30% of samples unimodal).
  * 63-68% of samples have best↔2nd-best gap < 0.02 → 14-way classification
    overfits noise.
  * Action 0 ("no forgetting") is (near-)best for 20-50% of samples; when it's
    not, a fixed "moderate forgetting" backup captures 75-81% of oracle reward.

Design:
  INPUT  : same 131K state as NeuralUCB (proven to carry signal)
  OUTPUT : a single sigmoid probability  P("action 0 is (near-)best")
  INFER  : if P > 0.5 → action 0 (a=0); else → budget-specific fixed backup
  LOSS   : BCE against y = 1 if reward[0] ≥ max(rewards) - delta else 0
  TRAIN  : offline supervised (we have the full reward vector per sample)

Backup action per budget (from Phase C observation):
  b=128 → action 8 (a=0.075)
  b=256 → action 10 (a=0.250)
  b=512 → action 9 (a=0.100)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# Budget → backup action index (in the 14-action grid)
DEFAULT_BACKUP = {128: 8, 256: 10, 512: 9}


class BinaryAgent(nn.Module):
    """Binary sigmoid-cache agent: 'should we forget?' + fixed backup."""

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
        backup_idx: int = 8,
        # legacy kwargs accepted (ignored)
        backbone_depth: int = 2,
        dropout: float = 0.0,
        budget_slot: bool = False,
    ):
        super().__init__()
        del backbone_depth, dropout, budget_slot

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
        self.backup_idx = int(backup_idx)
        self.metric_heads = metric_heads or ["qa_f1_score"]
        self._meta_dim = 1 + self.num_metric_types + self.num_task_types

        # NeuralUCB-style embedding (proven)
        self.head_merge = nn.Parameter(torch.randn(self.num_bins, self.num_heads) * 0.01)
        self.side_reduce = nn.Linear(self.num_bins, 256)

        fuse_in = 256 + 256 + self._meta_dim
        self.embed = nn.Sequential(
            nn.Linear(fuse_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)   # single logit

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
        logit = self.head(h).squeeze(-1)        # (B,)
        p0 = torch.sigmoid(logit)
        # Build a fake 14-dim reward_pred so downstream inference code works
        # action 0 gets p0, backup gets (1 - p0), others 0.
        reward_pred = torch.zeros(state.size(0), self.num_actions, device=state.device)
        reward_pred[:, 0] = p0
        reward_pred[:, self.backup_idx] = 1.0 - p0
        return {
            "reward_pred": reward_pred,
            "logits": logit,
            "feature_vector": h,
            "p_zero": p0,
        }

    @torch.no_grad()
    def act(self, state, metric_type=None):
        out = self.forward(state, metric_type=metric_type)
        p0 = out["p_zero"]
        if p0.ndim == 0:
            p0 = p0.unsqueeze(0)
        # If P(action 0) > 0.5, pick action 0; else pick backup
        idx = torch.where(p0 >= 0.5,
                          torch.tensor(0, device=p0.device),
                          torch.tensor(self.backup_idx, device=p0.device))
        a_i = idx // self.num_b_values
        b_i = idx % self.num_b_values
        return (self.a_values[a_i], self.b_values[b_i]), torch.where(p0 >= 0.5, p0, 1 - p0)

    # ── Legacy stubs ──
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
