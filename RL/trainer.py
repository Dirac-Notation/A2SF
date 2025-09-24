import os
import json
import random
import torch
import torch.optim as optim
from typing import Dict
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import A2SFRLConfig
from .policy import A2SFPolicy, ppo_update
from .buffer import RolloutBuffer
from .env import A2SFEnv
from .runner import A2SFRunner

class A2SFTrainer:
    def __init__(self, config: A2SFRLConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        
        self.runner = A2SFRunner(config)
        self.env = A2SFEnv(self.runner, config)
        
        self.training_data = self.runner.load_training_data()
        print(f"Loaded {len(self.training_data)} training samples")
        
        self.eval_episodes = random.sample(self.training_data, self.config.eval_samples)
        
        self.policy = A2SFPolicy(config.max_context).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        
        self.buffer = RolloutBuffer(device=self.device)
        
        os.makedirs(config.save_dir, exist_ok=True)
        
        self.training_stats = {
            "iterations": [],
            "rewards": [],
            "losses": []
        }
    
    def train(self, num_iterations: int = 1000):
        print(f"Starting training for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            
            self._collect_experiences()
            
            loss_stats = ppo_update(self.policy, self.buffer, self.config, self.optimizer)
            self.buffer.clear()
            
            if iteration % self.config.log_frequency == 0:
                self._log_progress(iteration, loss_stats)
            
            if iteration % 50 == 0 and iteration > 0:
                self._save_checkpoint(iteration)
            
            if iteration % self.config.eval_frequency == 0 and iteration > 0:
                self._evaluate(iteration)
    
    def _collect_experiences(self):
        episodes = random.sample(self.training_data, self.config.episodes_per_update)

        for episode_data in tqdm(episodes):
            state = self.env.reset(
                prompt=episode_data["input_prompt"],
                selected_indices=episode_data["selected_indices"],
                dataset=episode_data["dataset"],
            )
            
            state = state.to(self.device)

            action, log_prob, value = self.policy.act(state)
            
            reward, info = self.env.step(action)

            self.buffer.add(state, action, log_prob, reward, value)
            
            self.training_stats["rewards"].append(reward)
    
    def _log_progress(self, iteration: int, loss_stats: Dict[str, float]):
        n = self.config.episodes_per_update
        recent_rewards = self.training_stats["rewards"][-n:]
        
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        
        policy_loss = loss_stats.get("policy_loss", 0.0)
        value_loss = loss_stats.get("value_loss", 0.0)
        entropy = loss_stats.get("entropy", 0.0)
        
        print(f"Iteration {iteration}:")
        print(f"  Avg Reward:  {avg_reward:.4f}")
        print(f"  Policy Loss: {policy_loss:.4f}")
        print(f"  Value  Loss: {value_loss:.4f}")
        print(f"  Entropy:     {entropy:.4f}")
        print()
        
        progress_data = {
            "iteration": iteration,
            "avg_reward": avg_reward.item(),
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
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
        
        eval_rewards = []
        
        for episode_data in self.eval_episodes:
            state = self.env.reset(
                prompt=episode_data["input_prompt"],
                selected_indices=episode_data["selected_indices"],
                dataset=episode_data["dataset"],
            )
            state = state.to(self.device)

            with torch.no_grad():
                out = self.policy(state)
                alpha, beta = out["alpha"], out["beta"]
                action = (alpha / (alpha + beta)).clamp(1e-6, 1 - 1e-6)

            reward, info = self.env.step(action)

            eval_rewards.append(reward.item())
        
        avg_eval_reward = sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0.0
        
        print("Evaluation Results:")
        print(f"  Avg Reward:   {avg_eval_reward:.4f}")
        print()
        
        eval_data = {
            "iteration": iteration,
            "eval_avg_reward": avg_eval_reward,
            "eval_rewards": eval_rewards,
        }
        
        with open(os.path.join(self.config.save_dir, "evaluation.jsonl"), "a") as f:
            f.write(json.dumps(eval_data) + "\n")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        return checkpoint["iteration"]