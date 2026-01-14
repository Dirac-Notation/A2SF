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

from .main import A2SFRLConfig
from .policy import NeuralUCBPolicy
from .buffer import NeuralUCBBuffer
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
        
        # Calculate state dimension: just embedding_dim (CLS token only)
        # Get embedding_dim from context encoder
        embedding_dim = self.env.context_encoder.embedding_dim
        state_dim = embedding_dim
        
        self.policy = NeuralUCBPolicy(
            state_dim=state_dim,
            a_values=config.a_values,
            b_values=config.b_values
        ).to(self.device)
        self.buffer = NeuralUCBBuffer(device=self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        
        self.training_data = self.load_training_data()
        self.eval_episodes = random.sample(self.training_data, self.config.eval_samples)
        print(f"Loaded {len(self.training_data)} training samples")
        print(f"Loaded {len(self.eval_episodes)} evaluation samples")
        os.makedirs(config.save_dir, exist_ok=True)
        
    
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
            # Initialize stats for this iteration
            iteration_rewards = []
            iteration_actions_a = []
            iteration_actions_b = []
            
            episodes = random.sample(self.training_data, self.config.episodes_per_update)

            for episode_data in tqdm(episodes):
                state = self.env.encode_to_state(
                    prompt=episode_data["input_prompt"],
                    generated_text_full=episode_data["generated_text"],
                    dataset=episode_data["dataset"],
                )
                
                state = state.to(self.device)

                action, ucb_value = self.policy.act(state, beta=self.config.ucb_beta)
                
                reward, info = self.env.step(action)

                self.buffer.add(state, action, reward)
                
                a, b = action
                iteration_rewards.append(reward)
                iteration_actions_a.append(round(a.item(), 5) if isinstance(a, torch.Tensor) else a)
                iteration_actions_b.append(round(b.item(), 5) if isinstance(b, torch.Tensor) else b)
            
            loss_stats = self.policy.neural_ucb_update(self.buffer, self.config, self.optimizer)
            self.buffer.clear()
            
            if iteration % self.config.log_frequency == 0:
                self._log_progress(iteration, loss_stats, iteration_rewards, iteration_actions_a, iteration_actions_b)
            
            if iteration % 50 == 0 and iteration > 0:
                self._save_checkpoint(iteration)
            
            if iteration % self.config.eval_frequency == 0 and iteration > 0:
                self._evaluate(iteration)
    
    def _evaluate(self, iteration: int):
        print(f"Evaluating at iteration {iteration}")
        
        eval_rewards = []
        eval_actions_a = []
        eval_actions_b = []
        
        for episode_data in self.eval_episodes:
            state = self.env.encode_to_state(
                prompt=episode_data["input_prompt"],
                generated_text_full=episode_data["generated_text"],
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
        
        print("Evaluation Results:")
        print(f"  Avg Reward:   {avg_eval_reward:.4f}")
        print()
        
        eval_data = {
            "iteration": iteration,
            "eval_avg_reward": avg_eval_reward,
            "eval_rewards": eval_rewards,
            "eval_actions_a": eval_actions_a,
            "eval_actions_b": eval_actions_b,
        }
        
        with open(os.path.join(self.config.save_dir, "evaluation_progress.jsonl"), "a") as f:
            f.write(json.dumps(eval_data) + "\n")

    def _log_progress(self, iteration: int, loss_stats: Dict[str, float], iteration_rewards: List, iteration_actions_a: List, iteration_actions_b: List):
        avg_reward = sum(iteration_rewards) / len(iteration_rewards) if iteration_rewards else 0.0
        
        prediction_loss = loss_stats.get("prediction_loss", 0.0)
        total_loss = loss_stats.get("total_loss", 0.0)
        
        print(f"Iteration {iteration}:")
        print(f"  Avg Reward:        {avg_reward:.4f}")
        print(f"  Prediction Loss:   {prediction_loss:.4f}")
        print()
        
        progress_data = {
            "iteration": iteration,
            "avg_reward": avg_reward.item() if isinstance(avg_reward, torch.Tensor) else avg_reward,
            "prediction_loss": prediction_loss,
            "total_loss": total_loss,
            "actions_a": iteration_actions_a,
            "actions_b": iteration_actions_b,
        }
        
        with open(os.path.join(self.config.save_dir, "training_progress.jsonl"), "a") as f:
            f.write(json.dumps(progress_data) + "\n")

    def _save_checkpoint(self, iteration: int):
        checkpoint_path = os.path.join(self.config.save_dir, f"policy_{iteration}.pt")
        torch.save({
            "iteration": iteration,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Update config from checkpoint if available
        if "config" in checkpoint:
            checkpoint_config = checkpoint["config"]
            # Update important config values that affect model initialization
            self.config.context_encoder_model = checkpoint_config.context_encoder_model
            self.config.a_values = checkpoint_config.a_values
            self.config.b_values = checkpoint_config.b_values
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        return checkpoint["iteration"]