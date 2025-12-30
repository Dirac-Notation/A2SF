import os
import json
import random
import torch
import torch.optim as optim
from typing import Dict, List, Any
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import A2SFRLConfig
from .policy import A2SFPolicy
from .buffer import RolloutBuffer
from .env import A2SFEnv
from .runner import A2SFModelRunner

class A2SFTrainer:
    def __init__(self, config: A2SFRLConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(config.seed)
        random.seed(config.seed)        
        
        self.model_runner = A2SFModelRunner(config)
        self.env = A2SFEnv(self.model_runner, config)
        self.policy = A2SFPolicy(config.max_context).to(self.device)
        self.buffer = RolloutBuffer(device=self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        
        self.training_data = self.load_training_data()
        self.eval_episodes = random.sample(self.training_data, self.config.eval_samples)
        print(f"Loaded {len(self.training_data)} training samples")
        print(f"Loaded {len(self.eval_episodes)} evaluation samples")
        os.makedirs(config.save_dir, exist_ok=True)
        
        self.training_stats = {
            "iterations": [],
            "rewards": [],
            "losses": [],
            "actions_a": [],
            "actions_b": []
        }
    
    def load_training_data(self) -> List[Dict[str, Any]]:
        training_data_path = "datasets/training_data.jsonl"
        
        training_data = open(training_data_path, 'r', encoding='utf-8')
        training_data = [json.loads(line) for line in training_data]
        
        # for data in training_data:
        #     data["input_prompt"] = self.model_runner.prepare_prompt(data["input_prompt"], data["dataset"])
        
        print(f"Loaded {len(training_data)} training samples from {training_data_path}")
        return training_data
    
    def train(self, num_iterations: int = 1000):
        print(f"Starting training for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            
            episodes = random.sample(self.training_data, self.config.episodes_per_update)

            for episode_data in tqdm(episodes):
                state = self.env.encode_to_state(
                    prompt=episode_data["input_prompt"],
                    selected_indices=episode_data["selected_indices"],
                    dataset=episode_data["dataset"],
                )
                
                state = state.to(self.device)

                action, log_prob, value = self.policy.act(state)
                
                reward, info = self.env.step(action)

                self.buffer.add(state, action, log_prob, reward, value)
                
                a, b = action
                self.training_stats["rewards"].append(reward)
                self.training_stats["actions_a"].append(round(a.item(), 5) if isinstance(a, torch.Tensor) else a)
                self.training_stats["actions_b"].append(round(b.item(), 5) if isinstance(b, torch.Tensor) else b)
            
            loss_stats = self.policy.ppo_update(self.buffer, self.config, self.optimizer)
            self.buffer.clear()
            
            if iteration % self.config.log_frequency == 0:
                self._log_progress(iteration, loss_stats)
            
            if iteration % 50 == 0 and iteration > 0:
                self._save_checkpoint(iteration)
            
            if iteration % self.config.eval_frequency == 0 and iteration > 0:
                self._evaluate(iteration)
    
    def _evaluate(self, iteration: int):
        print(f"Evaluating at iteration {iteration}")
        
        eval_rewards = []
        
        for episode_data in self.eval_episodes:
            state = self.env.encode_to_state(
                prompt=episode_data["input_prompt"],
                selected_indices=episode_data["selected_indices"],
                dataset=episode_data["dataset"],
            )
            state = state.to(self.device)

            with torch.no_grad():
                out = self.policy(state)
                a_logits = out["a_logits"]  # (B, num_a_values)
                b_logits = out["b_logits"]  # (B, num_b_values)
                
                # For a: use mode (most likely value from Categorical)
                a_idx = torch.argmax(a_logits, dim=-1)
                a = self.policy.a_values[a_idx]
                
                # For b: use mode (most likely value from Categorical)
                b_idx = torch.argmax(b_logits, dim=-1)
                b = self.policy.b_values[b_idx]
                
                action = (a, b)

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

    def _log_progress(self, iteration: int, loss_stats: Dict[str, float]):
        n = self.config.episodes_per_update
        recent_rewards = self.training_stats["rewards"][-n:]
        recent_actions_a = self.training_stats["actions_a"][-n:]
        recent_actions_b = self.training_stats["actions_b"][-n:]
        
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        
        policy_loss = loss_stats.get("policy_loss", 0.0)
        policy_loss_a = loss_stats.get("policy_loss_a", 0.0)
        policy_loss_b = loss_stats.get("policy_loss_b", 0.0)
        value_loss = loss_stats.get("value_loss", 0.0)
        entropy = loss_stats.get("entropy", 0.0)
        
        print(f"Iteration {iteration}:")
        print(f"  Avg Reward:  {avg_reward:.4f}")
        print(f"  Policy Loss: {policy_loss:.4f} (a: {policy_loss_a:.4f}, b: {policy_loss_b:.4f})")
        print(f"  Value  Loss: {value_loss:.4f}")
        print(f"  Entropy:     {entropy:.4f}")
        print()
        
        progress_data = {
            "iteration": iteration,
            "avg_reward": avg_reward.item() if isinstance(avg_reward, torch.Tensor) else avg_reward,
            "policy_loss": policy_loss,
            "policy_loss_a": policy_loss_a,
            "policy_loss_b": policy_loss_b,
            "value_loss": value_loss,
            "entropy": entropy,
            "actions_a": recent_actions_a,
            "actions_b": recent_actions_b,
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
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        return checkpoint["iteration"]