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

    State vector layout: [seq_len, token_budget,
                          tova_top1(H)..tova_top4(H),
                          snap_top1(H)..snap_top4(H)]

    Three embedding channels:
      1. meta_embed: (seq_len, token_budget) -> embed_dim
      2. tova_embed: (tova_top1..top4 per head, 4H) -> embed_dim
      3. snap_embed: (snap_top1..top4 per head, 4H) -> embed_dim

    Averaged -> shared backbone -> per-metric reward heads.
    """

    def __init__(
        self,
        state_dim: int,
        a_values: torch.Tensor,
        b_values: torch.Tensor,
        metric_heads: List[str] = None,
        num_heads: int = 32,
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
        if self.state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {self.state_dim}")

        if metric_heads is None:
            metric_heads = ["qa_f1_score"]
        self.metric_heads = list(metric_heads)
        self.default_metric_head = self.metric_heads[0]
        self.metric_name_to_idx = {name: i for i, name in enumerate(self.metric_heads)}

        # Embedding dimensions
        self.feature_dim = 512
        self.topk = 4
        meta_dim = 2                       # (seq_len, token_budget)
        tova_dim = self.topk * self.num_heads  # (top1..top4 per head)
        snap_dim = self.topk * self.num_heads  # (top1..top4 per head)

        # Three separate embedding projections
        self.meta_embed = nn.Sequential(
            nn.Linear(meta_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim),
            nn.ReLU(),
        )
        self.tova_embed = nn.Sequential(
            nn.Linear(tova_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim),
            nn.ReLU(),
        )
        self.snap_embed = nn.Sequential(
            nn.Linear(snap_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim),
            nn.ReLU(),
        )

        # Shared backbone
        self.backbone = nn.Sequential(
            MLPResidualBlock(dim=self.feature_dim, hidden_dim=1024, dropout=0.1),
            MLPResidualBlock(dim=self.feature_dim, hidden_dim=1024, dropout=0.1),
        )

        # Per-metric reward heads
        self.reward_heads = nn.ModuleDict(
            {
                metric_name: nn.Linear(self.feature_dim, self.num_actions)
                for metric_name in self.metric_heads
            }
        )

        # Per-metric, per-action inverse covariance (Neural-Linear uncertainty).
        eye = torch.eye(self.feature_dim, device=a_values.device)
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

    def _split_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split flat state vector into (meta, tova_topk, snap_topk) components."""
        kH = self.topk * self.num_heads
        meta = state[:, :2]                    # (B, 2)
        tova = state[:, 2:2 + kH]             # (B, 4H)
        snap = state[:, 2 + kH:2 + 2 * kH]   # (B, 4H)
        return meta, tova, snap

    def _embed_state(self, state: torch.Tensor) -> torch.Tensor:
        """Embed and average three channels."""
        state = state.to(dtype=torch.float32)
        meta, tova, snap = self._split_state(state)

        e_meta = self.meta_embed(meta)     # (B, feature_dim)
        e_tova = self.tova_embed(tova)     # (B, feature_dim)
        e_snap = self.snap_embed(snap)     # (B, feature_dim)

        # Channel-wise average
        return (e_meta + e_tova + e_snap) / 3.0

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

        # Per-metric reward heads
        metric_types = self._resolve_metric_type_for_batch(metric_type, state.size(0))
        reward_rows: List[torch.Tensor] = []
        for row_idx, mname in enumerate(metric_types):
            head = self.reward_heads[self._head_key(mname)]
            logits = head(h[row_idx:row_idx + 1])
            reward_rows.append(torch.sigmoid(logits))
        reward_pred = torch.cat(reward_rows, dim=0)

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

        batch_size = reward_pred.size(0)
        uncertainty = torch.zeros(
            batch_size,
            self.num_actions,
            device=reward_pred.device,
            dtype=reward_pred.dtype,
        )
        for b in range(batch_size):
            m = self._metric_idx(metric_types[b])
            fv = feature_vector[b : b + 1]
            inv = self.inverse_lambdas[m]
            uncertainty[b : b + 1] = torch.einsum("bi,aij,bj->ba", fv, inv, fv)

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
