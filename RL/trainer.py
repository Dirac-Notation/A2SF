import os
import sys
import json
import random

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

from .main import A2SFRLConfig
from .policy import NeuralUCBPolicy
from .env import A2SFEnv
from .runner import A2SFModelRunner


class RLDataset(Dataset):
    """Dataset for RL training episodes"""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def rl_collate_fn(batch):
    """
    Custom collate function for RL dataset.
    Returns the batch as a list of dictionaries (no tensor conversion).
    """
    return batch

class A2SFTrainer:
    def __init__(self, config: A2SFRLConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(config.seed)
        random.seed(config.seed)        
        
        self.model_runner = A2SFModelRunner(config)
        self.env = A2SFEnv(self.model_runner, config)
        
        # State dimension is fixed to 8192 (output_dim of AttentionEncoder)
        state_dim = 8192 + 1
        
        self.policy = NeuralUCBPolicy(
            state_dim=state_dim,
            a_values=config.a_values,
            b_values=config.b_values
        ).to(self.device)
        
        # Optimizer only includes policy parameters
        # AttentionEncoder is frozen and uses target model's first layer, first head parameters
        all_params = list(self.policy.parameters())
        self.optimizer = optim.SGD(all_params, lr=config.lr)
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler(config)
        
        # Load and split data into training and evaluation sets
        all_data = self.load_training_data()
        training_data_list, eval_data_list = self._split_data(all_data, config.eval_samples, config.seed)
        
        # Create datasets
        self.training_dataset = RLDataset(training_data_list)
        self.eval_dataset = RLDataset(eval_data_list)
        
        # Create data loaders
        # For training: shuffle and use episodes_per_update as batch size
        self.train_loader = DataLoader(
            self.training_dataset,
            batch_size=config.episodes_per_update,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues with model
            pin_memory=False,
            collate_fn=rl_collate_fn  # Use custom collate function
        )
        
        # For evaluation: no shuffle, process all at once
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=1,  # Process one at a time for evaluation
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=rl_collate_fn  # Use custom collate function
        )
        
        print(f"Loaded {len(self.training_dataset)} training samples")
        print(f"Loaded {len(self.eval_dataset)} evaluation samples")
        print(f"Training batches per epoch: {len(self.train_loader)}")
        os.makedirs(config.save_dir, exist_ok=True)
        
    
    def load_training_data(self) -> List[Dict[str, Any]]:
        training_data_path = "datasets/training_data.jsonl"
        
        training_data = open(training_data_path, 'r', encoding='utf-8')
        training_data = [json.loads(line) for line in training_data]
        
        # for data in training_data:
        #     data["input_prompt"] = self.model_runner.prepare_prompt(data["input_prompt"], data["dataset"])
        
        print(f"Loaded {len(training_data)} total samples from {training_data_path}")
        return training_data
    
    def _split_data(self, all_data: List[Dict[str, Any]], eval_samples: int, seed: int) -> tuple:
        """
        Split data into training and evaluation sets completely.
        Uses fixed seed to ensure reproducibility.
        
        Args:
            all_data: All loaded data samples
            eval_samples: Number of samples to use for evaluation
            seed: Random seed for splitting
        
        Returns:
            tuple: (training_data, eval_data)
        """
        if len(all_data) <= eval_samples:
            raise ValueError(f"Not enough data: {len(all_data)} total samples, but {eval_samples} evaluation samples requested")
        
        # Use fixed seed for reproducible split
        rng = random.Random(seed)
        
        # Shuffle data with fixed seed
        shuffled_data = all_data.copy()
        rng.shuffle(shuffled_data)
        
        # Split: first eval_samples for evaluation, rest for training
        eval_data = shuffled_data[:eval_samples]
        training_data = shuffled_data[eval_samples:]
        
        # Verify no overlap
        eval_set = {id(d) for d in eval_data}
        train_set = {id(d) for d in training_data}
        assert len(eval_set & train_set) == 0, "Training and evaluation data must be completely separate!"
        
        return training_data, eval_data
    
    def _neural_ucb_update(
        self,
        state: torch.Tensor,
        action: Tuple[torch.Tensor, torch.Tensor],
        reward: torch.Tensor,
        config,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        NeuralUCB update: minimize prediction error (MSE only)
        Uncertainty is computed from covariance matrices, not learned
        Online update: each sample is updated immediately.
        
        Args:
            state: Encoded state tensor (already on device)
            action: Tuple of (a, b) action values
            reward: Observed reward (scalar tensor)
            config: Configuration object
            optimizer: Optimizer for training
        
        Returns:
            dict with loss statistics
        """
        self.policy.train()
        # Encoder is frozen, keep it in eval mode
        self.env.context_encoder.eval()
        
        # Ensure state has batch dimension
        if state.ndim == 1:
            state = state.unsqueeze(0)  # (1, state_dim) for batch dimension
        
        # Forward pass to get predictions and feature vectors (with gradient)
        out = self.policy.forward(state)
        reward_pred = out["reward_pred"]  # (1, num_a_values, num_b_values)
        feature_vector = out["feature_vector"]  # (1, feature_dim)
        
        # Get action indices
        a_idx, b_idx = self.policy._get_action_indices(action)
        # Ensure indices are scalars for single sample
        if a_idx.ndim > 0:
            a_idx = a_idx[0]
        if b_idx.ndim > 0:
            b_idx = b_idx[0]
        
        # Get predicted reward for selected action (single sample, batch_idx=0)
        selected_predict = reward_pred[0, a_idx, b_idx].unsqueeze(0)  # (1,)
        actual_reward = reward.unsqueeze(0) if reward.ndim == 0 else reward  # (1,)

        # NeuralUCB: Update neural network (backbone) to minimize prediction error
        # The loss is computed using current theta_a predictions
        prediction_loss = F.mse_loss(selected_predict, actual_reward)
        
        # L2 regularization: penalize large weights for stability
        # Only include policy parameters (encoder is frozen)
        l2_loss = 0.0
        for param in self.policy.parameters():
            if param.requires_grad:
                l2_loss += torch.sum(param ** 2)
        l2_loss = config.l2_coef * l2_loss
        
        # Total loss: prediction loss + L2 regularization
        total_loss = prediction_loss + l2_loss
        
        # Optimize backbone network (feature extractor) and encoder
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update inverse covariance matrices (no gradient)
        # These are updated using closed-form solutions, not gradient descent
        with torch.no_grad():
            self.policy._update_inverse_covariances(feature_vector, a_idx, b_idx)

        return {
            "prediction_loss": float(prediction_loss.item()),
            "l2_loss": float(l2_loss.item()),
            "total_loss": float(total_loss.item()),
        }
    
    def _create_scheduler(self, config) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on config"""
        if config.scheduler_type == "none":
            return None
        elif config.scheduler_type == "step":
            return lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.scheduler_step_size,
                gamma=config.scheduler_gamma
            )
        elif config.scheduler_type == "cosine":
            return lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.scheduler_T_max,
                eta_min=config.lr * 0.01  # Minimum learning rate (1% of initial)
            )
        elif config.scheduler_type == "exponential":
            return lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=config.scheduler_gamma
            )
        else:
            raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")

    def train(self, num_iterations: int = 1000):
        print(f"Starting training for {num_iterations} iterations")
        
        # Create an iterator that cycles through the data loader
        train_iter = iter(self.train_loader)
        
        for iteration in range(num_iterations):
            # Initialize stats for this iteration
            iteration_rewards = []
            iteration_actions_a = []
            iteration_actions_b = []
            iteration_losses = []
            
            # Get a batch from the data loader
            # If we've exhausted the loader, create a new iterator (with new shuffle)
            # DataLoader with shuffle=True will automatically shuffle when creating a new iterator
            try:
                batch = next(train_iter)
            except StopIteration:
                # Create a new iterator from the existing DataLoader
                # DataLoader automatically shuffles when shuffle=True
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Process each episode in the batch
            for episode_data in batch:
                # Encode state (only once)
                state = self.env.encode_to_state(
                    prompt=episode_data["input_prompt"],
                    generation_length=episode_data["generation_length"],
                    answer=episode_data["generated_text"],
                    dataset=episode_data["dataset"],
                )
                state = state.to(self.device)

                # Get action (forward pass happens inside act, but we'll reuse state for update)
                action, ucb_value = self.policy.act(state, beta=self.config.ucb_beta)
                
                # Get reward
                reward, info = self.env.step(action)

                # Online update: update immediately with this sample
                # Reuse the same state (no need to re-encode)
                loss_stats = self._neural_ucb_update(
                    state=state,
                    action=action,
                    reward=reward,
                    config=self.config,
                    optimizer=self.optimizer
                )
                iteration_losses.append(loss_stats)
                
                a, b = action
                iteration_rewards.append(reward.item() if isinstance(reward, torch.Tensor) else reward)
                iteration_actions_a.append(round(a.item(), 5) if isinstance(a, torch.Tensor) else a)
                iteration_actions_b.append(round(b.item(), 5) if isinstance(b, torch.Tensor) else b)
            
            # Average loss stats across episodes in this iteration
            avg_loss_stats = {
                "prediction_loss": sum(l["prediction_loss"] for l in iteration_losses) / len(iteration_losses),
                "l2_loss": sum(l["l2_loss"] for l in iteration_losses) / len(iteration_losses),
                "total_loss": sum(l["total_loss"] for l in iteration_losses) / len(iteration_losses),
            }
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            self._log_progress(iteration, avg_loss_stats, iteration_rewards, iteration_actions_a, iteration_actions_b)
            
            if iteration % self.config.eval_frequency == 0 and iteration > 0:
                self._save_checkpoint(iteration)
                self._evaluate(iteration)
    
    def _evaluate(self, iteration: int):
        print(f"Evaluating at iteration {iteration}")
        
        # Set to evaluation mode
        self.policy.eval()
        self.env.context_encoder.eval()
        
        eval_rewards = []
        eval_actions_a = []
        eval_actions_b = []
        
        for batch in tqdm(self.eval_loader, desc="Evaluating"):
            # DataLoader with batch_size=1 returns a list with one element
            episode_data = batch[0]
            
            state = self.env.encode_to_state(
                prompt=episode_data["input_prompt"],
                generation_length=episode_data["generation_length"],
                answer=episode_data["generated_text"],
                dataset=episode_data["dataset"],
            )
            state = state.to(self.device)

            with torch.no_grad():
                action, ucb_value = self.policy.act(state, beta=self.config.ucb_beta)

            reward, info = self.env.step(action)
            
            a, b = action

            eval_rewards.append(reward.item())
            eval_actions_a.append(round(a.item(), 5) if isinstance(a, torch.Tensor) else a)
            eval_actions_b.append(round(b.item(), 5) if isinstance(b, torch.Tensor) else b)
        
        avg_eval_reward = sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0.0
        avg_eval_reward = round(avg_eval_reward, 4)
        
        # Round individual rewards to 4 decimal places
        eval_rewards_rounded = [round(r, 4) for r in eval_rewards]
        
        print("Evaluation Results:")
        print(f"  Avg Reward:   {avg_eval_reward:.4f}")
        print()
        
        eval_data = {
            "iteration": iteration,
            "eval_avg_reward": avg_eval_reward,
            "eval_rewards": eval_rewards_rounded,
            "eval_actions_a": eval_actions_a,
            "eval_actions_b": eval_actions_b,
        }
        
        with open(os.path.join(self.config.save_dir, "evaluation_progress.jsonl"), "a") as f:
            f.write(json.dumps(eval_data) + "\n")

    def _log_progress(self, iteration: int, loss_stats: Dict[str, float], iteration_rewards: List, iteration_actions_a: List, iteration_actions_b: List):
        avg_reward = sum(iteration_rewards) / len(iteration_rewards) if iteration_rewards else 0.0
        
        prediction_loss = round(loss_stats.get("prediction_loss", 0.0), 4)
        total_loss = round(loss_stats.get("total_loss", 0.0), 4)
        l2_loss = round(loss_stats.get("l2_loss", 0.0), 4)
        
        print(f"Iteration {iteration}:")
        print(f"  Avg Reward:        {avg_reward:.4f}")
        print(f"  Prediction Loss:   {prediction_loss:.4f}")
        print(f"  L2 Loss:           {l2_loss:.4f}")
        print(f"  Total Loss:        {total_loss:.4f}")
        print()
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        progress_data = {
            "iteration": iteration,
            "avg_reward": round(avg_reward.item() if isinstance(avg_reward, torch.Tensor) else avg_reward, 4),
            "prediction_loss": prediction_loss,
            "l2_loss": l2_loss,
            "total_loss": total_loss,
            "learning_rate": round(current_lr, 6),
            "actions_a": iteration_actions_a,
            "actions_b": iteration_actions_b,
        }
        
        with open(os.path.join(self.config.save_dir, "training_progress.jsonl"), "a") as f:
            f.write(json.dumps(progress_data) + "\n")

    def _save_checkpoint(self, iteration: int):
        checkpoint_path = os.path.join(self.config.save_dir, f"policy_{iteration}.pt")
        checkpoint_data = {
            "iteration": iteration,
            "policy_state_dict": self.policy.state_dict(),
            "attention_encoder_state_dict": {},  # Encoder is frozen, no state to save
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        if self.scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint_data, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Update config from checkpoint if available
        if "config" in checkpoint:
            checkpoint_config = checkpoint["config"]
            # Update important config values that affect model initialization
            self.config.a_values = checkpoint_config.a_values
            self.config.b_values = checkpoint_config.b_values
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        
        # Encoder is frozen and uses target model's parameters, no need to load state
        if "attention_encoder_state_dict" in checkpoint and checkpoint["attention_encoder_state_dict"]:
            print("Note: Attention encoder uses target model's first layer parameters (frozen)")
        
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        return checkpoint["iteration"]