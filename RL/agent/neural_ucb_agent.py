# neural_ucb_agent.py
import torch
import torch.nn as nn

from typing import Dict, List, Tuple, Union

EPS = 1e-6


class MLPResidualBlock(nn.Module):
    """Residual MLP block with ReLU activations."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class NeuralUCBAgent(nn.Module):
    """
    NeuralUCB agent network for Sigmoid Cache RL (single-step / bandit).

    State vector layout: [seq_len(1), metric_one_hot(M),
                          tova_binned(side_dim), snap_binned(side_dim)]

    Embedding (2-stage):
      1. tova_proj: Linear(side_dim, 256)
         snap_proj: Linear(side_dim, 256)
      2. cat(tova_256, snap_256, seq_len, metric_one_hot) -> Linear(513, 512)

    -> backbone -> per-metric reward heads.
    """

    def __init__(
        self,
        state_dim: int,
        a_values: torch.Tensor,
        b_values: torch.Tensor,
        metric_heads: List[str] = None,
        num_heads: int = 32,
        num_metric_types: int = 10,
        side_dim: int = 65536,
    ):
        super().__init__()
        # Discrete action space: (a, b) pairs for sigmoid cache
        self.register_buffer("a_values", a_values)
        self.register_buffer("b_values", b_values)
        self.num_a_values = len(a_values)
        self.num_b_values = len(b_values)
        self.num_actions = self.num_a_values * self.num_b_values

        self.lambda_reg = 5 * self.num_actions
        self.state_dim = int(state_dim)
        self.num_heads = int(num_heads)
        self.num_metric_types = int(num_metric_types)
        self.side_dim = int(side_dim)
        if self.state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {self.state_dim}")

        if metric_heads is None:
            metric_heads = ["qa_f1_score"]
        self.metric_heads = list(metric_heads)
        self.default_metric_head = self.metric_heads[0]
        self.metric_name_to_idx = {name: i for i, name in enumerate(self.metric_heads)}

        self.feature_dim = 256
        self.num_bins = self.side_dim // self.num_heads
        meta_dim = 1 + self.num_metric_types  # seq_len + metric_one_hot

        # Stage 1: bin별 독립 weight로 H heads → 1 합침
        # weight shape: (num_bins, num_heads) — 각 bin 위치마다 고유한 head 결합 가중치
        self.head_merge = nn.Parameter(torch.randn(self.num_bins, self.num_heads) * 0.01)

        # Stage 2: (num_bins,) → Linear(num_bins, 256) → (256,)
        self.side_reduce = nn.Linear(self.num_bins, self.feature_dim)

        # Stage 3: cat(tova_256, snap_256, meta) → Linear → 512
        concat_dim = self.feature_dim * 2 + meta_dim
        self.embed_proj = nn.Sequential(
            nn.Linear(concat_dim, self.feature_dim * 2),
            nn.ReLU(),
        )

        # Backbone: 2x (linear + activation)
        self.backbone = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.ReLU(),
        )

        # Backbone output dim
        self.backbone_out_dim = self.feature_dim * 2  # 512

        # Per-metric reward heads: linear 1개씩
        self.reward_heads = nn.ModuleDict(
            {
                metric_name: nn.Linear(self.backbone_out_dim, self.num_actions)
                for metric_name in self.metric_heads
            }
        )

        # Per-metric, per-action inverse covariance (Neural-Linear uncertainty).
        eye = torch.eye(self.backbone_out_dim, device=a_values.device)
        inv0 = eye.unsqueeze(0).repeat(self.num_actions, 1, 1) / self.lambda_reg
        self.register_buffer(
            "inverse_lambdas",
            inv0.unsqueeze(0).repeat(len(self.metric_heads), 1, 1, 1).contiguous(),
        )

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
        meta_dim = 1 + self.num_metric_types
        meta = state[:, :meta_dim]                                # (B, 1+M)
        tova_flat = state[:, meta_dim:meta_dim + self.side_dim]   # (B, H*num_bins)
        snap_flat = state[:, meta_dim + self.side_dim:]           # (B, H*num_bins)

        # [B, H*num_bins] → [B, num_bins, H]
        tova = tova_flat.reshape(B, self.num_heads, self.num_bins).permute(0, 2, 1)
        snap = snap_flat.reshape(B, self.num_heads, self.num_bins).permute(0, 2, 1)

        # bin별 독립 weight로 H heads 합침: einsum("bnh,nh->bn", ...)
        tova_merged = torch.einsum("bnh,nh->bn", tova, self.head_merge)  # (B, num_bins)
        snap_merged = torch.einsum("bnh,nh->bn", snap, self.head_merge)  # (B, num_bins)

        # (num_bins,) → Linear → (256,)
        tova_emb = torch.relu(self.side_reduce(tova_merged))  # (B, 256)
        snap_emb = torch.relu(self.side_reduce(snap_merged))  # (B, 256)

        combined = torch.cat([tova_emb, snap_emb, meta], dim=-1)  # (B, 256+256+meta_dim)
        return self.embed_proj(combined)  # (B, 512)

    def _resolve_metric_type_for_batch(
        self,
        metric_type: Union[str, List[str], Tuple[str, ...], None],
        batch_size: int,
    ) -> List[str]:
        if metric_type is None:
            return [self.default_metric_head for _ in range(batch_size)]
        if isinstance(metric_type, str):
            return [metric_type for _ in range(batch_size)]
        if isinstance(metric_type, (list, tuple)):
            if len(metric_type) != batch_size:
                raise ValueError(
                    f"metric_type length must equal batch size ({batch_size}), got {len(metric_type)}"
                )
            return [str(m) for m in metric_type]
        raise TypeError(f"Unsupported metric_type type: {type(metric_type)}")

    def _head_key(self, metric_name: str) -> str:
        return metric_name if metric_name in self.reward_heads else self.default_metric_head

    def _metric_idx(self, metric_name: str) -> int:
        key = self._head_key(metric_name)
        return int(self.metric_name_to_idx[key])

    def forward(
        self,
        state: torch.Tensor,
        metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
    ) -> Dict[str, torch.Tensor]:
        if state.ndim == 1:
            state = state.unsqueeze(0)

        if state.size(-1) != self.state_dim:
            raise ValueError(f"Expected state last dim {self.state_dim}, got {state.size(-1)}")

        # Shared embedding + backbone
        embedded = self._embed_state(state)   # (B, feature_dim)
        h = self.backbone(embedded)           # (B, feature_dim)

        # Per-metric reward heads — 같은 metric끼리 묶어서 벡터화 (B개 sample → num_unique_metrics번 matmul)
        metric_types = self._resolve_metric_type_for_batch(metric_type, state.size(0))
        resolved_keys = [self._head_key(m) for m in metric_types]

        reward_pred = torch.empty(
            state.size(0), self.num_actions, device=h.device, dtype=h.dtype
        )
        for mname in set(resolved_keys):
            mask = [i for i, k in enumerate(resolved_keys) if k == mname]
            head = self.reward_heads[mname]
            logits = head(h[mask])  # (len(mask), num_actions)
            reward_pred[mask] = torch.sigmoid(logits)

        return {"reward_pred": reward_pred, "feature_vector": h}

    def _select_action_from_scores(
        self, scores: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Select discrete (a, b) action from per-action scores."""
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)

        action_idx = torch.argmax(scores, dim=-1)  # (B,)
        a_idx = action_idx // self.num_b_values
        b_idx = action_idx % self.num_b_values

        a_val = self.a_values[a_idx]
        b_val = self.b_values[b_idx]

        batch_indices = torch.arange(scores.size(0), device=scores.device)
        selected_score = scores[batch_indices, action_idx]  # (B,)
        return (a_val, b_val), selected_score

    def _compute_ucb_scores(
        self,
        state: torch.Tensor,
        beta: float,
        metric_type: Union[str, List[str], Tuple[str, ...], None],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.ndim == 1:
            state = state.unsqueeze(0)

        metric_types = self._resolve_metric_type_for_batch(metric_type, state.size(0))
        out = self.forward(state, metric_type=metric_type)
        reward_pred = out["reward_pred"]
        feature_vector = out["feature_vector"]

        if reward_pred.ndim == 1:
            reward_pred = reward_pred.unsqueeze(0)
            feature_vector = feature_vector.unsqueeze(0)

        # 배치 단위 einsum으로 uncertainty 계산 (B개 sample 한 번에).
        metric_idx_tensor = torch.tensor(
            [self._metric_idx(m) for m in metric_types],
            device=feature_vector.device,
            dtype=torch.long,
        )
        invs = self.inverse_lambdas[metric_idx_tensor]  # (B, num_actions, feat, feat)
        uncertainty = torch.einsum(
            "bi,baij,bj->ba", feature_vector, invs, feature_vector
        )

        ucb = reward_pred + beta * torch.sqrt(uncertainty + EPS)
        return reward_pred, ucb

    @torch.no_grad()
    def act(
        self,
        state: torch.Tensor,
        metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Inference action selection (pure exploitation, beta=0)."""
        out = self.forward(state, metric_type=metric_type)
        reward_pred = out["reward_pred"]
        return self._select_action_from_scores(reward_pred)

    @torch.no_grad()
    def act_with_ucb(
        self,
        state: torch.Tensor,
        beta: float = 1.0,
        metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Training action selection with UCB exploration."""
        _, ucb = self._compute_ucb_scores(state=state, beta=beta, metric_type=metric_type)
        return self._select_action_from_scores(ucb)

    def predict_reward(
        self,
        state: torch.Tensor,
        action: Tuple[torch.Tensor, torch.Tensor],
        metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
    ) -> torch.Tensor:
        """Predict reward for given state and action."""
        out = self.forward(state, metric_type=metric_type)
        reward_pred = out["reward_pred"]

        action_idx = self._get_action_indices(action)
        batch_indices = torch.arange(reward_pred.size(0), device=reward_pred.device)
        predicted_reward = reward_pred[batch_indices, action_idx]
        return predicted_reward

    def _get_action_indices(self, action: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Convert (a, b) action values to flat index in action space."""
        a_val, b_val = action

        if a_val.ndim == 0:
            a_val = a_val.unsqueeze(0)
        if b_val.ndim == 0:
            b_val = b_val.unsqueeze(0)

        a_val = a_val.view(-1)
        b_val = b_val.view(-1)

        a_expanded = self.a_values.unsqueeze(0)
        a_idx = torch.argmin(torch.abs(a_val.unsqueeze(-1) - a_expanded), dim=-1)

        b_expanded = self.b_values.unsqueeze(0)
        b_idx = torch.argmin(torch.abs(b_val.unsqueeze(-1) - b_expanded), dim=-1)

        return a_idx * self.num_b_values + b_idx

    def _update_inverse_covariances(
        self,
        feature_vectors: torch.Tensor,
        action_idx: torch.Tensor,
        metric_type: str,
    ):
        m = self._metric_idx(metric_type)

        if action_idx.ndim == 0:
            action_idx = action_idx.unsqueeze(0)
        action_indices = action_idx.flatten()

        unique_actions = torch.unique(action_indices)
        for act_idx in unique_actions:
            act_idx_val = act_idx.item() if isinstance(act_idx, torch.Tensor) else act_idx
            mask = action_indices == act_idx_val
            z_batch = feature_vectors[mask]

            lambda_inv = self.inverse_lambdas[m, act_idx].clone()
            for z in z_batch:
                z = z.unsqueeze(0)
                z_lambda_z = torch.mm(torch.mm(z, lambda_inv), z.t()).item()

                lambda_inv_z = torch.mm(lambda_inv, z.t())
                z_lambda_inv = torch.mm(z, lambda_inv)

                update_term = torch.mm(lambda_inv_z, z_lambda_inv) / (1.0 + z_lambda_z + EPS)
                lambda_inv = lambda_inv - update_term
                self.action_counts[m, act_idx] += 1

            self.inverse_lambdas[m, act_idx] = lambda_inv


__all__ = ["NeuralUCBAgent"]
