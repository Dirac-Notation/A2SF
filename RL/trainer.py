import os
import json
import time
import random
import torch
import torch.optim as optim
from typing import List, Dict, Any

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
    def __init__(self, config: A2SFRLConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        
        self.runner = A2SFRunner(config)
        self.env = A2SFEnv(self.runner, config)
        
        # Load all data for both training and evaluation
        self.training_data = self.runner.load_training_data(config.max_samples_per_task)
        
        # Use fixed random indices for evaluation to test generalization across tasks
        random.seed(42)  # Fixed seed for consistent evaluation data selection
        eval_size = min(config.eval_samples, len(self.training_data))
        all_indices = list(range(len(self.training_data)))
        self.eval_indices = random.sample(all_indices, eval_size)  # Random but fixed indices
        
        print(f"Loaded {len(self.training_data)} total samples")
        print(f"Using {len(self.eval_indices)} samples for evaluation")
        
        # Log evaluation task distribution for verification
        eval_tasks = [self.training_data[i]["task"] for i in self.eval_indices]
        task_counts = {}
        for task in eval_tasks:
            task_counts[task] = task_counts.get(task, 0) + 1
        print(f"Evaluation task distribution: {task_counts}")
        
        # Reset random seed for training
        random.seed(config.seed)
        
        context_encoder = ContextEncoder(config.sentence_transformer_model)
        state_dim = context_encoder.embedding_dim
        
        self.policy = A2SFPolicy(state_dim, config.action_min, config.action_max).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.buffer = RolloutBuffer(device=self.device)
        
        os.makedirs(config.save_dir, exist_ok=True)
        
        self.training_stats = {
            "rewards": [],
            "accuracy_scores": []
        }
    
    def train(self, num_iterations: int = 1000):
        print(f"Starting training for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            start_time = time.time()
            
            self._collect_experiences()
            
            if self.buffer.size() > 0:
                loss_stats = ppo_update(self.policy, self.buffer, self.config, self.optimizer)
                
                for key, value in loss_stats.items():
                    if torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value)):
                        print(f"Warning: {key} is {value}, setting to 0.0")
                        loss_stats[key] = 0.0
                
                self.buffer.clear()
            else:
                loss_stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
            
            if iteration % self.config.log_frequency == 0:
                self._log_progress(iteration, loss_stats, time.time() - start_time)
            
            if iteration % 50 == 0 and iteration > 0:
                self._save_checkpoint(iteration)
            
            if iteration % self.config.eval_frequency == 0 and iteration > 0:
                self._evaluate(iteration)
    
    def _collect_experiences(self):
        episodes = random.sample(
            self.training_data,
            min(self.config.episodes_per_update, len(self.training_data))
        )

        for episode_data in episodes:
            state = self.env.reset(
                prompt=episode_data["input_prompt"],
                task=episode_data["task"],
                tokens=episode_data["input_prompt"].split(),
                answers=episode_data["answers"],
                dataset=episode_data.get("dataset", None)
            )
            
            if isinstance(state, torch.Tensor):
                state = state.to(self.device)

            action, log_prob, value = self.policy.act(state)
            next_state, reward, done, info = self.env.step(action)

            if not isinstance(reward, torch.Tensor):
                reward = torch.tensor(reward, device=self.device, dtype=value.dtype)
            else:
                reward = reward.to(self.device).to(value.dtype)

            self.buffer.add(state, action, log_prob, reward, value)
            
            self.training_stats["rewards"].append(float(reward.item()))
            self.training_stats["accuracy_scores"].append(float(info["accuracy_score"]))
    
    def _log_progress(self, iteration: int, loss_stats: Dict[str, float], iteration_time: float):
        if not self.training_stats["rewards"]:
            return
        
        n = min(self.config.episodes_per_update, len(self.training_stats["rewards"]))
        recent_rewards = self.training_stats["rewards"][-n:]
        recent_accuracy = self.training_stats["accuracy_scores"][-n:]
        
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        avg_accuracy = sum(recent_accuracy) / len(recent_accuracy)
        
        policy_loss = loss_stats.get("policy_loss", 0.0)
        value_loss = loss_stats.get("value_loss", 0.0)
        entropy = loss_stats.get("entropy", 0.0)
        
        print(f"Iteration {iteration}:")
        print(f"  Avg Reward:  {avg_reward:.4f}")
        print(f"  Avg Accuracy:{avg_accuracy:.4f}")
        print(f"  Policy Loss: {policy_loss:.4f}")
        print(f"  Value  Loss: {value_loss:.4f}")
        print(f"  Entropy:     {entropy:.4f}")
        print(f"  Time:        {iteration_time:.2f}s")
        print()
        
        progress_data = {
            "iteration": iteration,
            "avg_reward": avg_reward,
            "avg_accuracy": avg_accuracy,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "iteration_time": iteration_time
        }
        
        with open(os.path.join(self.config.save_dir, "progress.jsonl"), "a") as f:
            f.write(json.dumps(progress_data) + "\n")
    
    def _save_checkpoint(self, iteration: int):
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
        print(f"Evaluating at iteration {iteration}")
        
        # Use fixed evaluation data by indices (no random sampling)
        eval_episodes = [self.training_data[i] for i in self.eval_indices]
        
        eval_rewards = []
        eval_accuracy = []
        
        for episode_data in eval_episodes:
            state = self.env.reset(
                prompt=episode_data["input_prompt"],
                task=episode_data["task"],
                tokens=episode_data["input_prompt"].split(),
                answers=episode_data["answers"],
                dataset=episode_data.get("dataset", None)
            )
            if isinstance(state, torch.Tensor):
                state = state.to(self.device)

            with torch.no_grad():
                out = self.policy(state)
                alpha, beta = out["alpha"], out["beta"]
                a01 = (alpha / (alpha + beta)).clamp(1e-6, 1 - 1e-6)
                action = self.policy.action_min + a01 * (self.policy.action_max - self.policy.action_min)

            _, reward, _, info = self.env.step(action)

            if not isinstance(reward, torch.Tensor):
                reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
            eval_rewards.append(float(reward.item()))
            eval_accuracy.append(float(info["accuracy_score"]))
        
        avg_eval_reward = sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0.0
        avg_eval_accuracy = sum(eval_accuracy) / len(eval_accuracy) if eval_accuracy else 0.0
        
        print("Evaluation Results:")
        print(f"  Avg Reward:   {avg_eval_reward:.4f}")
        print(f"  Avg Accuracy: {avg_eval_accuracy:.4f}")
        print()
        
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        return checkpoint["iteration"]