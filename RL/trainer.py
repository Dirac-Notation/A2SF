import os
import json
import time
import random
import torch
import torch.optim as optim
from typing import List, Dict, Any
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import A2SFRLConfig
from .policy import A2SFPolicy, ppo_update
from .buffer import RolloutBuffer
from .env import A2SFEnv
from .runner import A2SFRunner
from .features import ContextEncoder

class A2SFTrainer:
    """Trainer for A2SF RL agent"""
    
    def __init__(self, config: A2SFRLConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Set random seeds
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        
        # Initialize components
        self.runner = A2SFRunner(config)
        self.env = A2SFEnv(self.runner, config)
        
        # Load training data
        self.training_data = self.runner.load_training_data(config.max_samples_per_task)
        print(f"Loaded {len(self.training_data)} training samples")
        
        # Initialize policy network
        # Calculate state dimension
        context_encoder = ContextEncoder(config.sentence_transformer_model)
        context_dim = context_encoder.embedding_dim
        state_dim = context_dim
        
        self.policy = A2SFPolicy(state_dim, config.action_min, config.action_max).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        
        # Initialize buffer
        self.buffer = RolloutBuffer(device=self.device)
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        # Training statistics
        self.training_stats = {
            "iterations": [],
            "rewards": [],
            "accuracy_scores": [],
            "losses": []
        }
    
    def train(self, num_iterations: int = 1000):
        """
        Main training loop
        
        Args:
            num_iterations: Number of training iterations
        """
        print(f"Starting training for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            start_time = time.time()
            
            # Collect experiences
            self._collect_experiences()
            
            # Update policy
            if self.buffer.size() > 0:
                loss_stats = ppo_update(self.policy, self.buffer, self.config, self.optimizer)
                self.buffer.clear()
            else:
                loss_stats = {"policy_loss": 0.0, "value_loss": 0.0}
            
            # Log progress
            if iteration % self.config.log_frequency == 0:
                self._log_progress(iteration, loss_stats, time.time() - start_time)
            
            # Save checkpoint
            if iteration % 50 == 0 and iteration > 0:
                self._save_checkpoint(iteration)
            
            # Evaluate
            if iteration % self.config.eval_frequency == 0 and iteration > 0:
                self._evaluate(iteration)
    
    def _collect_experiences(self):
        """Collect experiences for PPO update"""
        # Sample random episodes
        episodes = random.sample(self.training_data, min(self.config.episodes_per_update, len(self.training_data)))

        for episode_data in episodes:
            # Reset environment
            answers = episode_data.get("answers", [])
            if not answers:
                print(f"Warning: No answers found for episode, skipping...")
                continue
                
            state = self.env.reset(
                prompt=episode_data["prompt"],
                task=episode_data["task"],
                tokens=episode_data["prompt"].split(),  # Simple tokenization
                answers=answers
            )
            
            # Get action from policy
            action, log_prob, value = self.policy.act(state)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            self.buffer.add(state, action, log_prob, reward, value, torch.tensor(done))
            
            # Store statistics
            self.training_stats["rewards"].append(reward.item())
            self.training_stats["accuracy_scores"].append(info["accuracy_score"])
    
    def _log_progress(self, iteration: int, loss_stats: Dict[str, float], iteration_time: float):
        """Log training progress"""
        if not self.training_stats["rewards"]:
            return
        
        # Compute statistics
        recent_rewards = self.training_stats["rewards"][-self.config.episodes_per_update:]
        recent_accuracy = self.training_stats["accuracy_scores"][-self.config.episodes_per_update:]
        
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        avg_accuracy = sum(recent_accuracy) / len(recent_accuracy)
        
        print(f"Iteration {iteration}:")
        print(f"  Avg Reward: {avg_reward:.4f}")
        print(f"  Avg Accuracy: {avg_accuracy:.4f}")
        print(f"  Policy Loss: {loss_stats['policy_loss']:.4f}")
        print(f"  Value Loss: {loss_stats['value_loss']:.4f}")
        print(f"  Time: {iteration_time:.2f}s")
        print()
        
        # Save to progress file
        progress_data = {
            "iteration": iteration,
            "avg_reward": avg_reward,
            "avg_accuracy": avg_accuracy,
            "policy_loss": loss_stats["policy_loss"],
            "value_loss": loss_stats["value_loss"],
            "iteration_time": iteration_time
        }
        
        with open(os.path.join(self.config.save_dir, "progress.jsonl"), "a") as f:
            f.write(json.dumps(progress_data) + "\n")
    
    def _save_checkpoint(self, iteration: int):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.save_dir, f"policy_{iteration}.pt")
        torch.save({
            "iteration": iteration,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "training_stats": self.training_stats
        }, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def _evaluate(self, iteration: int):
        """Evaluate policy on validation set"""
        print(f"Evaluating at iteration {iteration}")
        
        # Sample evaluation episodes
        eval_episodes = random.sample(self.training_data, min(self.config.eval_samples, len(self.training_data)))
        
        eval_rewards = []
        eval_accuracy = []
        
        for episode_data in eval_episodes:
            # Reset environment
            answers = episode_data.get("answers", [])
            if not answers:
                print(f"Warning: No answers found for evaluation episode, skipping...")
                continue
                
            state = self.env.reset(
                prompt=episode_data["prompt"],
                task=episode_data["task"],
                tokens=episode_data["prompt"].split(),
                answers=answers
            )
            
            # Get action from policy (no exploration during evaluation)
            with torch.no_grad():
                action, _, _ = self.policy.act(state)
            
            # Take step
            _, reward, _, info = self.env.step(action)
            
            eval_rewards.append(reward.item())
            eval_accuracy.append(info["accuracy_score"])
        
        # Compute evaluation statistics
        avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
        avg_eval_accuracy = sum(eval_accuracy) / len(eval_accuracy)
        
        print(f"Evaluation Results:")
        print(f"  Avg Reward: {avg_eval_reward:.4f}")
        print(f"  Avg Accuracy: {avg_eval_accuracy:.4f}")
        print()
        
        # Save evaluation results
        eval_data = {
            "iteration": iteration,
            "eval_avg_reward": avg_eval_reward,
            "eval_avg_accuracy": avg_eval_accuracy,
            "eval_rewards": eval_rewards,
            "eval_accuracy": eval_accuracy
        }
        
        with open(os.path.join(self.config.save_dir, "evaluation.jsonl"), "a") as f:
            f.write(json.dumps(eval_data) + "\n")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        
        return checkpoint["iteration"]
