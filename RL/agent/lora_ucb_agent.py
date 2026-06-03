# lora_ucb_agent.py
"""NeuralUCB-style agent with task-wise LoRA backbone and per-metric heads.

Architecture (per user spec):
  state(D) → Linear(D → 256)                             [Shared Embedding]
  → LoRAResidualBlock × 2 (GeLU, fc1+fc2 each with shared FC + per-task LoRA)
  → per-metric Linear(256 → num_actions)                  [Head, NO task residual]
  → sigmoid                                                [output]

LoRA: each FC = W_shared(x) + B_t @ (A_t @ x) where (A_t, B_t) is per-task low-rank.
Init: A ~ randn × 0.01, B = 0 → initial Δ = 0 (warm start = shared backbone only).
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

EPS = 1e-6


class LoRALinear(nn.Module):
    """Linear with shared W,b and per-task low-rank delta.

    Forward: y = shared(x) + B[t] @ (A[t] @ x)
      A: (T, rank, in), B: (T, out, rank)
    """

    def __init__(self, in_dim: int, out_dim: int, num_tasks: int, rank: int = 4):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_tasks = int(num_tasks)
        self.rank = int(rank)
        self.shared = nn.Linear(in_dim, out_dim)
        self.A = nn.Parameter(torch.randn(num_tasks, rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(num_tasks, out_dim, rank))

    def forward(self, x: torch.Tensor, task_idx: torch.Tensor) -> torch.Tensor:
        out = self.shared(x)
        # x: (B, in), task_idx: (B,) long
        A_t = self.A[task_idx]   # (B, r, in)
        B_t = self.B[task_idx]   # (B, out, r)
        h = torch.einsum("bri,bi->br", A_t, x)
        delta = torch.einsum("bor,br->bo", B_t, h)
        return out + delta


class LoRAResidualBlock(nn.Module):
    """MLPResidualBlock with LoRA-augmented FCs and GeLU."""

    def __init__(self, dim: int, hidden_dim: int, num_tasks: int,
                 rank: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = LoRALinear(dim, hidden_dim, num_tasks, rank)
        self.fc2 = LoRALinear(hidden_dim, dim, num_tasks, rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, task_idx: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.act(self.fc1(x, task_idx))
        h = self.dropout(h)
        h = self.fc2(h, task_idx)
        h = self.dropout(h)
        return residual + h


class LoRAUCBAgent(nn.Module):
    """NeuralUCB-style agent with per-task LoRA backbone and per-metric heads.

    No task residual on output (only per-metric Linear head + sigmoid).
    Compatible with NeuralUCB API (act, act_with_ucb, _update_inverse_covariances).
    """

    def __init__(
        self,
        state_dim: int,
        a_values: torch.Tensor,
        b_values: torch.Tensor,
        num_metric_types: int,
        num_task_types: int,
        metric_heads: List[str] = None,
        hidden: int = 256,
        block_hidden: int = 512,
        lora_rank: int = 4,
        dropout: float = 0.0,
        paired_actions: bool = True,
        include_seq_len: bool = True,
        backbone_route_by: str = "task",     # "task" | "metric" — what indexes per-route LoRA
        head_route_by: str = "metric",        # "task" | "metric" — what selects per-route head
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden = int(hidden)
        self.block_hidden = int(block_hidden)
        self.lora_rank = int(lora_rank)
        self.num_metric_types = int(num_metric_types)
        self.num_task_types = int(num_task_types)
        self.include_seq_len = bool(include_seq_len)
        self.paired_actions = bool(paired_actions)
        self.dropout_p = float(dropout)
        if backbone_route_by not in ("task", "metric"):
            raise ValueError(f"backbone_route_by must be 'task'|'metric', got {backbone_route_by}")
        if head_route_by not in ("task", "metric"):
            raise ValueError(f"head_route_by must be 'task'|'metric', got {head_route_by}")
        self.backbone_route_by = backbone_route_by
        self.head_route_by = head_route_by

        if num_task_types <= 0:
            raise ValueError("LoRAUCBAgent requires num_task_types > 0 for LoRA routing.")

        self.register_buffer("a_values", a_values)
        self.register_buffer("b_values", b_values)
        self.num_a_values = int(len(a_values))
        self.num_b_values = int(len(b_values))
        if self.paired_actions:
            assert self.num_a_values == self.num_b_values
            self.num_actions = self.num_a_values
        else:
            self.num_actions = self.num_a_values * self.num_b_values

        self.lambda_reg = 5 * self.num_actions

        # metric_heads list is informational (used to map metric_name → idx for inverse_lambdas
        # routing when head_route_by == "metric"). When head_route_by == "task", heads are
        # indexed by task_idx so this list is just stored for compatibility.
        if metric_heads is None or len(metric_heads) == 0:
            metric_heads = ["qa_f1_score"]
        self.metric_heads = list(metric_heads)
        self.default_metric_head = self.metric_heads[0]
        self.metric_name_to_idx = {n: i for i, n in enumerate(self.metric_heads)}

        # Cardinalities
        backbone_n = self.num_task_types if backbone_route_by == "task" else self.num_metric_types
        head_n = self.num_task_types if head_route_by == "task" else len(self.metric_heads)
        self._backbone_n = int(backbone_n)
        self._head_n = int(head_n)

        # Shared Embedding: single Linear state_dim → hidden
        self.embed = nn.Linear(self.state_dim, self.hidden)

        # Backbone: 2 × LoRAResidualBlock with GeLU
        # LoRA num_tasks param actually means "num_routes" — keep the kwarg name for compat.
        self.blocks = nn.ModuleList([
            LoRAResidualBlock(
                dim=self.hidden, hidden_dim=self.block_hidden,
                num_tasks=backbone_n, rank=self.lora_rank,
                dropout=self.dropout_p,
            ) for _ in range(2)
        ])

        # Heads: ModuleDict keyed by metric name (when route_by=metric) or
        # ModuleList by task idx (when route_by=task). To keep ckpt format
        # consistent, always use ModuleDict with keys ["m_<name>"] or ["t_<idx>"].
        if head_route_by == "metric":
            head_keys = list(self.metric_heads)
        else:
            head_keys = [f"t{i}" for i in range(self.num_task_types)]
        self._head_keys = head_keys
        self.reward_heads = nn.ModuleDict({
            k: nn.Linear(self.hidden, self.num_actions) for k in head_keys
        })

        # UCB Σ⁻¹ buffers: shape (num_head_routes, num_actions, hidden, hidden)
        eye = torch.eye(self.hidden, device=a_values.device)
        inv0 = eye.unsqueeze(0).repeat(self.num_actions, 1, 1) / self.lambda_reg
        self.register_buffer(
            "inverse_lambdas",
            inv0.unsqueeze(0).repeat(head_n, 1, 1, 1).contiguous(),
        )
        self.register_buffer(
            "action_counts",
            torch.zeros(head_n, self.num_actions, dtype=torch.long),
        )

        self._init_weights()

    @property
    def backbone_out_dim(self) -> int:
        return self.hidden

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        # LoRA: A small randn, B zero (already done in LoRALinear.__init__)

    def _task_idx_from_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        task_oh_start = (1 if self.include_seq_len else 0) + self.num_metric_types
        task_oh = state[:, task_oh_start:task_oh_start + self.num_task_types]
        return task_oh.argmax(dim=-1).long()

    def _metric_idx_from_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        metric_oh_start = (1 if self.include_seq_len else 0)
        metric_oh = state[:, metric_oh_start:metric_oh_start + self.num_metric_types]
        return metric_oh.argmax(dim=-1).long()

    def _route_idx_from_state(self, state: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "task":
            return self._task_idx_from_state(state)
        return self._metric_idx_from_state(state)

    def _resolve_metric_keys(
        self,
        metric_type: Union[str, List[str], Tuple[str, ...], None],
        batch_size: int,
    ) -> List[str]:
        if metric_type is None:
            return [self.default_metric_head] * batch_size
        if isinstance(metric_type, str):
            mt = [metric_type] * batch_size
        else:
            if len(metric_type) != batch_size:
                raise ValueError(f"metric_type len {len(metric_type)} != batch {batch_size}")
            mt = [str(m) for m in metric_type]
        return [m if m in self.metric_name_to_idx else self.default_metric_head for m in mt]

    def _metric_idx(self, metric_name: str) -> int:
        """Used by train.py and inference for inverse_lambdas indexing.
        When head_route_by == "metric", returns metric idx in metric_heads list.
        When head_route_by == "task", DEPRECATED — caller should use _head_idx_for_state instead."""
        key = metric_name if metric_name in self.metric_name_to_idx else self.default_metric_head
        return int(self.metric_name_to_idx[key])

    def _head_idx_for_state(self, state: torch.Tensor, metric_type=None) -> torch.Tensor:
        """Per-sample head index. Returns (B,) long tensor for inverse_lambdas indexing."""
        if self.head_route_by == "task":
            return self._task_idx_from_state(state)
        # metric routing: prefer explicit metric_type arg; else read from state's metric_oh.
        keys = self._resolve_metric_keys(metric_type, state.size(0))
        return torch.tensor([self.metric_name_to_idx[k] for k in keys],
                            device=state.device, dtype=torch.long)

    def _embed_and_backbone(self, state: torch.Tensor, route_idx: torch.Tensor) -> torch.Tensor:
        z = self.embed(state.to(dtype=torch.float32))
        for blk in self.blocks:
            z = blk(z, route_idx)
        return z

    def forward(
        self,
        state: torch.Tensor,
        metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
    ) -> Dict[str, torch.Tensor]:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.size(-1) != self.state_dim:
            raise ValueError(f"Expected state last dim {self.state_dim}, got {state.size(-1)}")

        backbone_idx = self._route_idx_from_state(state, self.backbone_route_by)
        h = self._embed_and_backbone(state, backbone_idx)   # (B, hidden)

        # Heads dispatch
        if self.head_route_by == "metric":
            keys = self._resolve_metric_keys(metric_type, state.size(0))
        else:  # task
            task_idx = self._task_idx_from_state(state)
            keys = [f"t{int(i.item())}" for i in task_idx]

        logits = torch.empty(state.size(0), self.num_actions, device=h.device, dtype=h.dtype)
        for k in set(keys):
            mask = [i for i, kk in enumerate(keys) if kk == k]
            logits[mask] = self.reward_heads[k](h[mask])

        reward_pred = torch.sigmoid(logits)
        # Return both for inverse_lambdas indexing (train.py uses this)
        return {
            "reward_pred": reward_pred,
            "feature_vector": h,
            "backbone_idx": backbone_idx,
            "head_keys": keys,
        }

    # ───────────── action selection ─────────────
    def _select_action_from_scores(
        self, scores: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        action_idx = torch.argmax(scores, dim=-1)
        if self.paired_actions:
            a_idx = action_idx
            b_idx = action_idx
        else:
            a_idx = action_idx // self.num_b_values
            b_idx = action_idx % self.num_b_values
        a_val = self.a_values[a_idx]
        b_val = self.b_values[b_idx]
        batch_indices = torch.arange(scores.size(0), device=scores.device)
        selected_score = scores[batch_indices, action_idx]
        return (a_val, b_val), selected_score

    def _compute_ucb_scores(
        self,
        state: torch.Tensor,
        beta: float,
        metric_type: Union[str, List[str], Tuple[str, ...], None],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        out = self.forward(state, metric_type=metric_type)
        reward_pred = out["reward_pred"]
        feature_vector = out["feature_vector"]
        if reward_pred.ndim == 1:
            reward_pred = reward_pred.unsqueeze(0)
            feature_vector = feature_vector.unsqueeze(0)

        head_idx = self._head_idx_for_state(state, metric_type)
        invs = self.inverse_lambdas[head_idx]   # (B, num_actions, feat, feat)
        uncertainty = torch.einsum("bi,baij,bj->ba", feature_vector, invs, feature_vector)
        ucb = reward_pred + beta * torch.sqrt(uncertainty + EPS)
        return reward_pred, ucb

    @torch.no_grad()
    def act(
        self, state: torch.Tensor,
        metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        out = self.forward(state, metric_type=metric_type)
        return self._select_action_from_scores(out["reward_pred"])

    @torch.no_grad()
    def act_with_ucb(
        self, state: torch.Tensor, beta: float = 1.0,
        metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        _, ucb = self._compute_ucb_scores(state, beta, metric_type)
        return self._select_action_from_scores(ucb)

    def predict_reward(
        self, state: torch.Tensor,
        action: Tuple[torch.Tensor, torch.Tensor],
        metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
    ) -> torch.Tensor:
        out = self.forward(state, metric_type=metric_type)
        reward_pred = out["reward_pred"]
        action_idx = self._get_action_indices(action)
        batch_indices = torch.arange(reward_pred.size(0), device=reward_pred.device)
        return reward_pred[batch_indices, action_idx]

    def _get_action_indices(self, action: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        a_val, b_val = action
        if a_val.ndim == 0: a_val = a_val.unsqueeze(0)
        if b_val.ndim == 0: b_val = b_val.unsqueeze(0)
        a_val = a_val.view(-1); b_val = b_val.view(-1)
        if self.paired_actions:
            da = (a_val.unsqueeze(-1) - self.a_values.unsqueeze(0)).abs()
            db = (b_val.unsqueeze(-1) - self.b_values.unsqueeze(0)).abs()
            return torch.argmin(da + db, dim=-1)
        a_idx = torch.argmin((a_val.unsqueeze(-1) - self.a_values.unsqueeze(0)).abs(), dim=-1)
        b_idx = torch.argmin((b_val.unsqueeze(-1) - self.b_values.unsqueeze(0)).abs(), dim=-1)
        return a_idx * self.num_b_values + b_idx

    def _update_inverse_covariances(
        self, feature_vectors: torch.Tensor, action_idx: torch.Tensor, metric_type,
    ):
        """Sherman-Morrison rank-1 update of inverse covariance.

        metric_type can be:
          - str (single metric name): all samples use this metric → resolved to single head_idx
            (only when head_route_by=='metric')
          - int: a single head_idx applied to all samples
          - LongTensor (B,): per-sample head_idx (preferred for general routing)
        """
        if action_idx.ndim == 0:
            action_idx = action_idx.unsqueeze(0)
        action_indices = action_idx.flatten()

        # Resolve to per-sample head index tensor.
        B = action_indices.size(0)
        if isinstance(metric_type, str):
            m = self._metric_idx(metric_type)
            head_idx_tensor = torch.full((B,), m, dtype=torch.long,
                                          device=action_indices.device)
        elif isinstance(metric_type, int):
            head_idx_tensor = torch.full((B,), int(metric_type), dtype=torch.long,
                                          device=action_indices.device)
        else:
            # tensor
            head_idx_tensor = metric_type.to(device=action_indices.device, dtype=torch.long).flatten()
            if head_idx_tensor.numel() != B:
                raise ValueError(f"head_idx length {head_idx_tensor.numel()} != {B}")

        # Group by (head_idx, action_idx) pairs.
        for h_val in torch.unique(head_idx_tensor).tolist():
            sel_h = head_idx_tensor == h_val
            for a_val in torch.unique(action_indices[sel_h]).tolist():
                mask = sel_h & (action_indices == a_val)
                z_batch = feature_vectors[mask]
                lambda_inv = self.inverse_lambdas[h_val, a_val].clone()
                for z in z_batch:
                    z = z.unsqueeze(0)
                    z_lambda_z = torch.mm(torch.mm(z, lambda_inv), z.t()).item()
                    lambda_inv_z = torch.mm(lambda_inv, z.t())
                    z_lambda_inv = torch.mm(z, lambda_inv)
                    update = torch.mm(lambda_inv_z, z_lambda_inv) / (1.0 + z_lambda_z + EPS)
                    lambda_inv = lambda_inv - update
                    self.action_counts[h_val, a_val] += 1
                self.inverse_lambdas[h_val, a_val] = lambda_inv


__all__ = ["LoRAUCBAgent", "LoRALinear", "LoRAResidualBlock"]
