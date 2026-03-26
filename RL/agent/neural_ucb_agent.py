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

    - Predicts expected reward for each discrete (a, b) action pair
    - Uses UCB (Upper Confidence Bound) for action selection
    - Uncertainty is computed using covariance matrix (Neural-Linear approach)
    """

    def __init__(
        self,
        state_dim: int,
        a_values: torch.Tensor,
        b_values: torch.Tensor,
        metric_heads: List[str] = None,
    ):
        super().__init__()
        # Discrete action space: (a, b) pairs for sigmoid cache
        self.register_buffer("a_values", a_values)
        self.register_buffer("b_values", b_values)
        self.num_a_values = len(a_values)
        self.num_b_values = len(b_values)
        self.num_actions = self.num_a_values * self.num_b_values  # Total action combinations

        # Regularization parameter for covariance matrix
        # Auto-set lambda_reg based on action space size.
        # (Do not accept user input; keep scale stable across different action spaces.)
        self.lambda_reg = 5 * self.num_actions
        self.state_dim = int(state_dim)
        if self.state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {self.state_dim}")

        # Projection + single normalization + residual ReLU MLP stack.
        self.input_proj = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.backbone = nn.Sequential(
            MLPResidualBlock(dim=512, hidden_dim=1024, dropout=0.1),
            MLPResidualBlock(dim=512, hidden_dim=1024, dropout=0.1),
        )
        self.feature_dim = 512

        if metric_heads is None:
            metric_heads = ["qa_f1_score"]
        self.metric_heads = list(metric_heads)
        self.default_metric_head = self.metric_heads[0]
        self.reward_heads = nn.ModuleDict(
            {metric_name: nn.Linear(512, self.num_actions) for metric_name in self.metric_heads}
        )

        # Initialize inverse covariance matrices for each action.
        self.register_buffer(
            "inverse_lambdas",
            torch.eye(self.feature_dim, device=a_values.device).unsqueeze(0).repeat(self.num_actions, 1, 1)
            / self.lambda_reg,
        )

        # Action counts for tracking
        self.register_buffer("action_counts", torch.zeros(self.num_actions, dtype=torch.long))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

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

    def _compute_reward_pred(self, features: torch.Tensor, metric_types: List[str]) -> torch.Tensor:
        out = []
        for row_idx, metric_name in enumerate(metric_types):
            head_name = metric_name if metric_name in self.reward_heads else self.default_metric_head
            logits = self.reward_heads[head_name](features[row_idx : row_idx + 1])
            pred_row = torch.sigmoid(logits)
            out.append(pred_row)
        return torch.cat(out, dim=0)

    def forward(
        self,
        state: torch.Tensor,
        metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
          - reward_pred: (B, num_actions)
          - feature_vector: (B, feature_dim)
        """
        # Ensure state is 2D
        if state.ndim == 1:
            state = state.unsqueeze(0)

        if state.size(-1) != self.state_dim:
            raise ValueError(f"Expected state last dim {self.state_dim}, got {state.size(-1)}")

        x = self.input_proj(state.to(dtype=torch.float32))
        h = self.backbone(x)

        metric_types = self._resolve_metric_type_for_batch(metric_type, h.size(0))
        reward_pred = self._compute_reward_pred(h, metric_types)

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

    def _select_action_from_scores_minmax(
        self, scores: torch.Tensor, mode: str
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Select (a, b) action from per-action scores by argmax/argmin.

        mode:
          - "best": argmax
          - "worst": argmin
        """
        if mode not in {"best", "worst"}:
            raise ValueError(f"Unsupported mode: {mode}")

        if scores.ndim == 1:
            scores = scores.unsqueeze(0)

        if mode == "best":
            action_idx = torch.argmax(scores, dim=-1)  # (B,)
        else:
            action_idx = torch.argmin(scores, dim=-1)  # (B,)

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
        """Compute (reward_pred, ucb_scores) for all actions.

        Returns:
          - reward_pred: (B, num_actions)
          - ucb_scores: (B, num_actions)
        """
        out = self.forward(state, metric_type=metric_type)
        reward_pred = out["reward_pred"]  # (B, num_actions)
        feature_vector = out["feature_vector"]  # (B, feature_dim)

        if reward_pred.ndim == 1:
            reward_pred = reward_pred.unsqueeze(0)
            feature_vector = feature_vector.unsqueeze(0)

        uncertainty = torch.einsum(
            "bi,aij,bj->ba", feature_vector, self.inverse_lambdas, feature_vector
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
        reward_pred = out["reward_pred"]  # (B, num_actions)
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

    @torch.no_grad()
    def get_best_action(
        self,
        state: torch.Tensor,
        beta: float = 1.0,
        metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Get best (a, b) according to UCB scores (argmax)."""
        _, ucb = self._compute_ucb_scores(state=state, beta=beta, metric_type=metric_type)
        return self._select_action_from_scores_minmax(ucb, mode="best")

    @torch.no_grad()
    def get_worst_action(
        self,
        state: torch.Tensor,
        beta: float = 1.0,
        metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Get worst (a, b) according to UCB scores (argmin)."""
        _, ucb = self._compute_ucb_scores(state=state, beta=beta, metric_type=metric_type)
        return self._select_action_from_scores_minmax(ucb, mode="worst")

    def predict_reward(
        self,
        state: torch.Tensor,
        action: Tuple[torch.Tensor, torch.Tensor],
        metric_type: Union[str, List[str], Tuple[str, ...], None] = None,
    ) -> torch.Tensor:
        """Predict reward for given state and action."""
        out = self.forward(state, metric_type=metric_type)
        reward_pred = out["reward_pred"]  # (B, num_actions)

        action_idx = self._get_action_indices(action)
        batch_indices = torch.arange(reward_pred.size(0), device=reward_pred.device)
        predicted_reward = reward_pred[batch_indices, action_idx]  # (B,)
        return predicted_reward

    def _get_action_indices(self, action: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Convert (a, b) action values to flat index in action space."""
        a_val, b_val = action

        # Ensure tensors are at least 1D
        if a_val.ndim == 0:
            a_val = a_val.unsqueeze(0)
        if b_val.ndim == 0:
            b_val = b_val.unsqueeze(0)

        a_val = a_val.view(-1)
        b_val = b_val.view(-1)

        # Find closest indices
        a_expanded = self.a_values.unsqueeze(0)  # (1, num_a_values)
        a_idx = torch.argmin(torch.abs(a_val.unsqueeze(-1) - a_expanded), dim=-1)  # (N,)

        b_expanded = self.b_values.unsqueeze(0)  # (1, num_b_values)
        b_idx = torch.argmin(torch.abs(b_val.unsqueeze(-1) - b_expanded), dim=-1)  # (N,)

        return a_idx * self.num_b_values + b_idx

    def _update_inverse_covariances(self, feature_vectors: torch.Tensor, action_idx: torch.Tensor):
        """
        Update inverse covariance matrices using Sherman-Morrison formula.

        Note: this is primarily used during training to update per-action uncertainty.
        """
        if action_idx.ndim == 0:
            action_idx = action_idx.unsqueeze(0)
        action_indices = action_idx.flatten()

        unique_actions = torch.unique(action_indices)
        for act_idx in unique_actions:
            act_idx_val = act_idx.item() if isinstance(act_idx, torch.Tensor) else act_idx
            mask = action_indices == act_idx_val
            z_batch = feature_vectors[mask]  # (K, feature_dim)

            lambda_inv = self.inverse_lambdas[act_idx]  # (feature_dim, feature_dim)
            for z in z_batch:
                z = z.unsqueeze(0)  # (1, feature_dim)
                z_lambda_z = torch.mm(torch.mm(z, lambda_inv), z.t()).item()

                lambda_inv_z = torch.mm(lambda_inv, z.t())  # (feature_dim, 1)
                z_lambda_inv = torch.mm(z, lambda_inv)  # (1, feature_dim)

                update_term = torch.mm(lambda_inv_z, z_lambda_inv) / (1.0 + z_lambda_z + EPS)
                lambda_inv = lambda_inv - update_term
                self.action_counts[act_idx] += 1

            self.inverse_lambdas[act_idx] = lambda_inv


__all__ = ["NeuralUCBAgent"]

