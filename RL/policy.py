# policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Dict, Tuple

EPS = 1e-6

# Default a values: discrete set for a parameter (sigmoid cache)
# Note: These defaults should match A2SFRLConfig defaults
# The actual values used should come from config when initializing policy
DEFAULT_A_VALUES = [0.0, 0.01, 0.1, 10.0]

# Default b values: discrete set for b parameter (sigmoid cache)
# Note: These defaults should match A2SFRLConfig defaults
# The actual values used should come from config when initializing policy
DEFAULT_B_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 512, 1024, 4096, 8192]

class NeuralUCBPolicy(nn.Module):
    """
    NeuralUCB policy network for A2SF RL agent (single-step / bandit) with sigmoid cache
    - Predicts expected reward for each (a, b) action combination
    - Uses UCB (Upper Confidence Bound) for action selection
    - Uncertainty is computed using covariance matrix (Neural-Linear approach)
    """

    def __init__(self, state_dim: int, a_values: torch.Tensor = None, b_values: torch.Tensor = None, lambda_reg: float = 0.1):
        super().__init__()
        if a_values is None:
            a_values = torch.tensor(DEFAULT_A_VALUES)
        if b_values is None:
            b_values = torch.tensor(DEFAULT_B_VALUES)
        self.register_buffer('a_values', a_values)
        self.register_buffer('b_values', b_values)
        self.num_a_values = len(a_values)
        self.num_b_values = len(b_values)
        
        # Total number of actions (all combinations of a and b)
        self.num_actions = self.num_a_values * self.num_b_values
        
        # Regularization parameter for covariance matrix
        self.lambda_reg = lambda_reg

        # Backbone network for state encoding
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )
        
        # Feature dimension (output of backbone)
        self.feature_dim = 512

        # Reward prediction head: predicts reward for each (a, b) combination
        # Output shape: (batch_size, num_a_values, num_b_values)
        self.reward_head = nn.Linear(512, self.num_actions)
        
        # Initialize inverse covariance matrices for each action
        # Lambda^{-1} for each action: (num_actions, feature_dim, feature_dim)
        # Lambda = Z^T * Z + lambda_reg * I, where Z is matrix of feature vectors for this action
        # Initial: Lambda^{-1} = (lambda_reg * I)^{-1} = (1/lambda_reg) * I
        # This gives initial uncertainty = z^T * (1/lambda_reg) * I * z = (1/lambda_reg) * ||z||^2
        # Larger lambda_reg -> smaller initial uncertainty (more confident initially)
        self.register_buffer(
            'inverse_lambdas',
            torch.eye(self.feature_dim, device=a_values.device).unsqueeze(0).repeat(self.num_actions, 1, 1) / lambda_reg
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
            reward_pred: (B, num_a_values, num_b_values) - predicted rewards using theta_a
            feature_vector: (B, feature_dim) - feature representation for covariance computation
        """
        # Ensure state is 2D
        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        h = self.backbone(state)  # (B, feature_dim)
        
        # Predict rewards for all action combinations using reward head
        # Apply sigmoid to ensure rewards are in [0, 1] range for NeuralUCB stability
        reward_flat = torch.sigmoid(self.reward_head(h))  # (B, num_actions) in [0, 1]
        reward_pred = reward_flat.view(-1, self.num_a_values, self.num_b_values)  # (B, num_a_values, num_b_values)
        
        return {"reward_pred": reward_pred, "feature_vector": h}

    @torch.no_grad()
    def act(self, state: torch.Tensor, beta: float = 1.0) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Select action using UCB (Upper Confidence Bound) with covariance-based uncertainty
        
        Args:
            state: (B, state_dim) or (state_dim,)
            beta: exploration parameter (higher = more exploration)
        
        Returns:
            action: tuple of (a, b) where a and b are from discrete sets
            ucb_value: UCB value of selected action
        """
        out = self.forward(state)
        reward_pred = out["reward_pred"]  # (B, num_a_values, num_b_values)
        feature_vector = out["feature_vector"]  # (B, feature_dim)

        if reward_pred.ndim == 2:
            reward_pred = reward_pred.unsqueeze(0)
            feature_vector = feature_vector.unsqueeze(0)

        # Vectorized uncertainty computation using einsum
        # Compute z^T * Lambda^{-1} * z for all batches and all actions at once
        # z: (B, feature_dim), inverse_lambdas: (num_actions, feature_dim, feature_dim)
        # Result: (B, num_actions) where result[b, a] = z[b]^T * Lambda_a^{-1} * z[b]
        uncertainty_all = torch.einsum('bi,aij,bj->ba', feature_vector, self.inverse_lambdas, feature_vector)
        uncertainty = uncertainty_all.view(-1, self.num_a_values, self.num_b_values)  # (B, num_a_values, num_b_values)
        
        # Compute UCB: mean + beta * sqrt(uncertainty)
        ucb = reward_pred + beta * torch.sqrt(uncertainty + EPS)  # (B, num_a_values, num_b_values)
        
        # Select action with highest UCB
        ucb_flat = ucb.view(-1, self.num_actions)  # (B, num_actions)
        action_idx = torch.argmax(ucb_flat, dim=-1)  # (B,)
        
        # Convert flat index to (a_idx, b_idx)
        a_idx = action_idx // self.num_b_values  # (B,)
        b_idx = action_idx % self.num_b_values   # (B,)
        
        a = self.a_values[a_idx]  # (B,)
        b = self.b_values[b_idx]  # (B,)
        
        # Get UCB value of selected action
        batch_indices = torch.arange(ucb.size(0), device=ucb.device)
        ucb_value = ucb[batch_indices, a_idx, b_idx]  # (B,)
        
        return (a, b), ucb_value

    def predict_reward(self, state: torch.Tensor, action: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Predict reward for given state and action
        
        Args:
            state: (B, state_dim) or (state_dim,)
            action: tuple of (a, b) where a and b are from discrete sets
        
        Returns:
            predicted_reward: (B,) - predicted reward
        """
        out = self.forward(state)
        reward_pred = out["reward_pred"]  # (B, num_a_values, num_b_values)
        
        a, b = action
        
        # Find closest a and b indices
        a_values_expanded = self.a_values.unsqueeze(0)  # (1, num_a_values)
        a_idx = torch.argmin(torch.abs(a.unsqueeze(-1) - a_values_expanded), dim=-1)  # (B,)
        
        b_values_expanded = self.b_values.unsqueeze(0)  # (1, num_b_values)
        b_idx = torch.argmin(torch.abs(b.unsqueeze(-1) - b_values_expanded), dim=-1)  # (B,)
        
        # Get predicted reward for selected action
        batch_indices = torch.arange(reward_pred.size(0), device=reward_pred.device)
        predicted_reward = reward_pred[batch_indices, a_idx, b_idx]  # (B,)
        
        return predicted_reward

    def _get_action_indices(self, actions: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert action values to indices"""
        a, b = actions
        a_values_expanded = self.a_values.unsqueeze(0)  # (1, num_a_values)
        a_idx = torch.argmin(torch.abs(a.unsqueeze(-1) - a_values_expanded), dim=-1)  # (N,)
        b_values_expanded = self.b_values.unsqueeze(0)  # (1, num_b_values)
        b_idx = torch.argmin(torch.abs(b.unsqueeze(-1) - b_values_expanded), dim=-1)  # (N,)
        return a_idx, b_idx
    
    def _update_inverse_covariances(self, feature_vectors: torch.Tensor, a_idx: torch.Tensor, b_idx: torch.Tensor):
        """
        Update inverse covariance matrices using Sherman-Morrison formula
        Lambda^{-1}_{t+1} = Lambda^{-1}_t - (Lambda^{-1}_t * z * z^T * Lambda^{-1}_t) / (1 + z^T * Lambda^{-1}_t * z)
        
        Args:
            feature_vectors: (N, feature_dim) - feature vectors for each sample
            a_idx: (N,) - action a indices
            b_idx: (N,) - action b indices
        """
        # Ensure indices are 1D
        a_idx = a_idx.flatten()
        b_idx = b_idx.flatten()
        
        # Convert (a_idx, b_idx) to flat action indices
        action_indices = a_idx * self.num_b_values + b_idx  # (N,)
        
        # Update each unique action's covariance matrix
        unique_actions = torch.unique(action_indices)
        
        for action_idx in unique_actions:
            # Get all feature vectors for this action
            action_idx_val = action_idx.item() if isinstance(action_idx, torch.Tensor) else action_idx
            mask = (action_indices == action_idx_val)
            z_batch = feature_vectors[mask]  # (K, feature_dim) where K is number of samples for this action
            
            # Get current inverse lambda for this action
            lambda_inv = self.inverse_lambdas[action_idx]  # (feature_dim, feature_dim)
            
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
                self.action_counts[action_idx] += 1
            
            # Store updated inverse lambda
            self.inverse_lambdas[action_idx] = lambda_inv
    

    def neural_ucb_update(
        self,
        buffer,
        config,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        NeuralUCB update: minimize prediction error (MSE only)
        Uncertainty is computed from covariance matrices, not learned
        
        Args:
            buffer: Experience buffer containing (state, action, reward)
            config: Configuration object
            optimizer: Optimizer for training
        
        Returns:
            dict with loss statistics
        """
        self.train()

        states, actions, rewards = buffer.get()
        
        # Forward pass to get predictions and feature vectors
        out = self.forward(states)
        reward_pred = out["reward_pred"]  # (N, num_a_values, num_b_values)
        feature_vectors = out["feature_vector"]  # (N, feature_dim)
        
        # Get action indices
        a_idx, b_idx = self._get_action_indices(actions)
        batch_indices = torch.arange(states.size(0), device=states.device)
        
        # Get predicted reward for selected actions
        selected_predict = reward_pred[batch_indices, a_idx, b_idx]  # (N,)
        actual_rewards = rewards.squeeze(-1)  # (N,)
        
        # NeuralUCB: Update neural network (backbone) to minimize prediction error
        # The loss is computed using current theta_a predictions
        loss = F.mse_loss(selected_predict, actual_rewards)
        
        # Optimize backbone network (feature extractor)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
        optimizer.step()

        # Update inverse covariance matrices and theta_a (no gradient)
        # These are updated using closed-form solutions, not gradient descent
        with torch.no_grad():
            self._update_inverse_covariances(feature_vectors, a_idx, b_idx)

        return {
            "prediction_loss": float(loss.item()),
            "total_loss": float(loss.item()),
        }
