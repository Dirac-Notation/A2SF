# policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Dict, Tuple

EPS = 1e-6

class NeuralUCBPolicy(nn.Module):
    """
    NeuralUCB policy network for Sigmoid Cache RL agent (single-step / bandit)
    - Predicts expected reward for each discrete (a, b) action pair
    - Uses UCB (Upper Confidence Bound) for action selection
    - Uncertainty is computed using covariance matrix (Neural-Linear approach)
    """

    def __init__(
        self,
        state_dim: int,
        a_values: torch.Tensor,
        b_values: torch.Tensor,
        lambda_reg: float = 0.1,
    ):
        super().__init__()
        # Discrete action space: (a, b) pairs for sigmoid cache
        self.register_buffer("a_values", a_values)
        self.register_buffer("b_values", b_values)
        self.num_a_values = len(a_values)
        self.num_b_values = len(b_values)
        self.num_actions = self.num_a_values * self.num_b_values  # Total action combinations
        
        # Regularization parameter for covariance matrix
        self.lambda_reg = lambda_reg

        # Backbone network for state encoding
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )

        # Feature dimension (output of backbone)
        self.feature_dim = 512

        # Reward prediction head: predicts reward for each (a, b) pair
        # Output shape: (batch_size, num_a_values * num_b_values)
        self.reward_head = nn.Linear(512, self.num_actions)
        
        # Initialize inverse covariance matrices for each action
        # Lambda^{-1} for each action: (num_actions, feature_dim, feature_dim)
        # Lambda = Z^T * Z + lambda_reg * I, where Z is matrix of feature vectors for this action
        # Initial: Lambda^{-1} = (lambda_reg * I)^{-1} = (1/lambda_reg) * I
        # This gives initial uncertainty = z^T * (1/lambda_reg) * I * z = (1/lambda_reg) * ||z||^2
        # Larger lambda_reg -> smaller initial uncertainty (more confident initially)
        self.register_buffer(
            "inverse_lambdas",
            torch.eye(self.feature_dim, device=a_values.device)
            .unsqueeze(0)
            .repeat(self.num_actions, 1, 1)
            / lambda_reg,
        )

        # Action counts for tracking
        self.register_buffer('action_counts', torch.zeros(self.num_actions, dtype=torch.long))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: (B, state_dim) or (state_dim,)
        Returns:
            dict with reward_pred, feature_vector
            reward_pred: (B, num_actions) - predicted rewards for each (a, b) pair
            feature_vector: (B, feature_dim) - feature representation for covariance computation
        """
        # Ensure state is 2D
        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        h = self.backbone(state)  # (B, feature_dim)
        
        # Predict rewards for all (a, b) pairs using reward head
        # Apply sigmoid to ensure rewards are in [0, 1] range for NeuralUCB stability
        reward_pred = torch.sigmoid(self.reward_head(h))  # (B, num_actions) in [0, 1]
        
        return {"reward_pred": reward_pred, "feature_vector": h}

    @torch.no_grad()
    def act(self, state: torch.Tensor, beta: float = 1.0) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Select action using UCB (Upper Confidence Bound) with covariance-based uncertainty
        
        Args:
            state: (B, state_dim) or (state_dim,)
            beta: exploration parameter (higher = more exploration)
        
        Returns:
            action: tuple of (a, b) tensors from discrete sets
            ucb_value: UCB value of selected action
        """
        out = self.forward(state)
        reward_pred = out["reward_pred"]  # (B, num_actions)
        feature_vector = out["feature_vector"]  # (B, feature_dim)

        if reward_pred.ndim == 1:
            reward_pred = reward_pred.unsqueeze(0)
            feature_vector = feature_vector.unsqueeze(0)

        # Vectorized uncertainty computation using einsum
        # Compute z^T * Lambda^{-1} * z for all batches and all actions at once
        # z: (B, feature_dim), inverse_lambdas: (num_actions, feature_dim, feature_dim)
        # Result: (B, num_actions) where result[b, a] = z[b]^T * Lambda_a^{-1} * z[b]
        uncertainty = torch.einsum("bi,aij,bj->ba", feature_vector, self.inverse_lambdas, feature_vector)  # (B, num_actions)

        # Compute UCB: mean + beta * sqrt(uncertainty)
        ucb = reward_pred + beta * torch.sqrt(uncertainty + EPS)  # (B, num_actions)

        # Select action with highest UCB (flattened index)
        action_idx = torch.argmax(ucb, dim=-1)  # (B,)

        # Convert flat index to (a_idx, b_idx)
        a_idx = action_idx // self.num_b_values  # (B,)
        b_idx = action_idx % self.num_b_values   # (B,)

        # Get actual a and b values
        a_val = self.a_values[a_idx]  # (B,)
        b_val = self.b_values[b_idx]  # (B,)

        # Get UCB value of selected action
        batch_indices = torch.arange(ucb.size(0), device=ucb.device)
        ucb_value = ucb[batch_indices, action_idx]  # (B,)

        return (a_val, b_val), ucb_value

    def predict_reward(self, state: torch.Tensor, action: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Predict reward for given state and action
        
        Args:
            state: (B, state_dim) or (state_dim,)
            action: tuple of (a, b) tensors from discrete sets
        
        Returns:
            predicted_reward: (B,) - predicted reward
        """
        out = self.forward(state)
        reward_pred = out["reward_pred"]  # (B, num_actions)

        # Find closest action indices (flattened)
        action_idx = self._get_action_indices(action)

        # Get predicted reward for selected action
        batch_indices = torch.arange(reward_pred.size(0), device=reward_pred.device)
        predicted_reward = reward_pred[batch_indices, action_idx]  # (B,)
        
        return predicted_reward

    def _get_action_indices(self, action: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Convert (a, b) action values to flat index in action space"""
        a_val, b_val = action
        
        # Ensure tensors are at least 1D
        if a_val.ndim == 0:
            a_val = a_val.unsqueeze(0)
        if b_val.ndim == 0:
            b_val = b_val.unsqueeze(0)
        
        a_val = a_val.view(-1)
        b_val = b_val.view(-1)

        # Find closest a index
        a_expanded = self.a_values.unsqueeze(0)  # (1, num_a_values)
        a_idx = torch.argmin(torch.abs(a_val.unsqueeze(-1) - a_expanded), dim=-1)  # (N,)
        
        # Find closest b index
        b_expanded = self.b_values.unsqueeze(0)  # (1, num_b_values)
        b_idx = torch.argmin(torch.abs(b_val.unsqueeze(-1) - b_expanded), dim=-1)  # (N,)
        
        # Convert to flat index: a_idx * num_b_values + b_idx
        flat_idx = a_idx * self.num_b_values + b_idx  # (N,)
        
        return flat_idx
    
    def _update_inverse_covariances(self, feature_vectors: torch.Tensor, action_idx: torch.Tensor):
        """
        Update inverse covariance matrices using Sherman-Morrison formula
        Lambda^{-1}_{t+1} = Lambda^{-1}_t - (Lambda^{-1}_t * z * z^T * Lambda^{-1}_t) / (1 + z^T * Lambda^{-1}_t * z)
        
        Args:
            feature_vectors: (N, feature_dim) - feature vectors for each sample
            action_idx: (N,) or scalar - flat action indices (a_idx * num_b_values + b_idx)
        """
        # Ensure indices are 1D
        if action_idx.ndim == 0:
            action_idx = action_idx.unsqueeze(0)
        action_indices = action_idx.flatten()
        
        # Update each unique action's covariance matrix
        unique_actions = torch.unique(action_indices)
        
        for act_idx in unique_actions:
            # Get all feature vectors for this action
            act_idx_val = act_idx.item() if isinstance(act_idx, torch.Tensor) else act_idx
            mask = (action_indices == act_idx_val)
            z_batch = feature_vectors[mask]  # (K, feature_dim) where K is number of samples for this action
            
            # Get current inverse lambda for this action
            lambda_inv = self.inverse_lambdas[act_idx]  # (feature_dim, feature_dim)
            
            # Update for each sample (can be batched, but doing sequentially for clarity)
            for z in z_batch:
                z = z.unsqueeze(0)  # (1, feature_dim)
                
                # Compute z^T * Lambda^{-1} * z
                z_lambda_z = torch.mm(torch.mm(z, lambda_inv), z.t()).item()  # scalar
                
                # Sherman-Morrison update
                # Lambda^{-1}_{new} = Lambda^{-1} - (Lambda^{-1} * z * z^T * Lambda^{-1}) / (1 + z^T * Lambda^{-1} * z)
                lambda_inv_z = torch.mm(lambda_inv, z.t())  # (feature_dim, 1)
                z_lambda_inv = torch.mm(z, lambda_inv)  # (1, feature_dim)
                
                update_term = torch.mm(lambda_inv_z, z_lambda_inv) / (1.0 + z_lambda_z + EPS)
                lambda_inv = lambda_inv - update_term
                
                # Update action count
                self.action_counts[act_idx] += 1
            
            # Store updated inverse lambda
            self.inverse_lambdas[act_idx] = lambda_inv
    
