#!/usr/bin/env python3
"""
Main script for training A2SF RL agent
"""

import argparse
import os
import json
import torch
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class A2SFRLConfig:
    # ----- Model Configuration -----
    model: str = "llama3"  # llama, llama2, llama3, opt
    
    # ----- Policy Action Space -----
    # Discrete candidate values for Sigmoid cache parameters (a, b)
    # a: sigmoid steepness parameter
    a_values: torch.Tensor = field(
        default_factory=lambda: torch.tensor(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            # [0.0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            dtype=torch.float32,
        )
    )
    # b: sigmoid shift parameter
    b_values: torch.Tensor = field(
        default_factory=lambda: torch.tensor(
            [1, 16, 128, 1024],
            # [0],
            dtype=torch.float32,
        )
    )
    
    # ----- NeuralUCB Hyperparameters -----
    lr: float = 1e-1
    ucb_beta_max: float = 1.0  # Initial exploration parameter for UCB
    ucb_beta_min: float = 0.1   # Final exploration parameter for UCB
    l2_coef: float = 1e-5  # L2 regularization coefficient for weight decay
    
    # ----- Training Configuration -----
    epochs: int = 40  # Number of full passes over training dataset
    episodes_per_update: int = 4  # Number of episodes per update

    # ----- Dataset Paths (pre-split) -----
    train_data_path: str = "datasets/training_data.jsonl"
    eval_data_path: str = "datasets/eval_data.jsonl"
    
    # ----- Misc -----
    seed: int = 42
    save_dir: str = "runs/a2sf_rl"
    resume: Optional[str] = None  # Path to checkpoint to resume from
    
    @property
    def model_name(self) -> str:
        """Backward compatibility: model_name property"""
        return self.model
    
    @property
    def device(self) -> str:
        """Device is determined only by CUDA availability (CUDA_VISIBLE_DEVICES로 마스킹)."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    @property
    def ucb_beta(self) -> float:
        """Backward compatibility: initial UCB beta."""
        return self.ucb_beta_max
    
    @classmethod
    def from_args(cls):
        """Create configuration from command line arguments (no GPU args; GPU는 CUDA_VISIBLE_DEVICES로 제어)"""
        # Create default config instance to get default values
        default_config = cls()
        
        parser = argparse.ArgumentParser(description="Train A2SF RL Agent")

        # Minimal command line arguments (GPU는 CLI CUDA_VISIBLE_DEVICES로만 제어)
        parser.add_argument('--model', type=str, default=default_config.model, choices=["llama", "llama2", "llama3", "opt"], help="Model name")
        parser.add_argument('--save_dir', type=str, default=default_config.save_dir, help="Directory to save checkpoints and logs")
        parser.add_argument('--resume', type=str, default=default_config.resume, help="Path to checkpoint to resume from (e.g., runs/a2sf_rl/policy_300.pt)")
        parser.add_argument('--train_data_path', type=str, default=default_config.train_data_path, help="Path to fixed training split jsonl")
        parser.add_argument('--eval_data_path', type=str, default=default_config.eval_data_path, help="Path to fixed evaluation split jsonl")
        parser.add_argument('--epochs', type=int, default=default_config.epochs, help="Number of training epochs")

        args = parser.parse_args()
        
        # Get seed from environment variable if set, otherwise use default
        seed = int(default_config.seed)
        
        return cls(
            model=args.model,
            save_dir=args.save_dir,
            seed=seed,
            # All other fields use defaults
            a_values=default_config.a_values,
            b_values=default_config.b_values,
            lr=default_config.lr,
            ucb_beta_max=default_config.ucb_beta_max,
            ucb_beta_min=default_config.ucb_beta_min,
            l2_coef=default_config.l2_coef,
            epochs=args.epochs,
            episodes_per_update=default_config.episodes_per_update,
            train_data_path=args.train_data_path,
            eval_data_path=args.eval_data_path,
            resume=args.resume,
        )

def main():
    """Main function"""
    # Import here to avoid circular import
    from .trainer import A2SFTrainer
    
    # Create configuration from command line arguments
    config = A2SFRLConfig.from_args()
    
    # Print configuration
    print("Sigmoid Cache RL Training Configuration:")
    print(f"  Model: {config.model}")
    print(f"  Device: {config.device}")
    print(f"  Seed: {config.seed}")
    print(f"  a_values: {config.a_values.tolist()}")
    print(f"  b_values: {config.b_values.tolist()}")
    print(f"  Episodes per update: {config.episodes_per_update}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Epochs: {config.epochs}")
    print(f"  UCB beta (max -> min): {config.ucb_beta_max} -> {config.ucb_beta_min}")
    print(f"  Train data: {config.train_data_path}")
    print(f"  Eval data:  {config.eval_data_path}")
    print(f"  Save directory: {config.save_dir}")
    print()
    
    # Create trainer
    trainer = A2SFTrainer(config)
    
    # Resume from checkpoint if specified
    start_iteration = 0
    if config.resume:
        start_iteration = trainer.load_checkpoint(config.resume)
        print(f"Resuming training from iteration {start_iteration}")

    print(f"  Iterations per epoch: {trainer.iterations_per_epoch}")
    print(f"  Total iterations: {trainer.total_iterations}")
    print(f"  LR Scheduler: cosine (T_max: {trainer.scheduler_t_max})")
    
    # Train
    final_iteration = trainer.train(num_epochs=config.epochs)
    
    # Save final model with the same structure as periodic checkpoints
    final_checkpoint_path = os.path.join(config.save_dir, "policy_final.pt")
    torch.save(
        {
            "iteration": final_iteration,
            "policy_state_dict": trainer.policy.state_dict(),
            "attention_encoder_state_dict": {},  # Keep identical to trainer checkpoints
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "config": config,
            "scheduler_state_dict": trainer.scheduler.state_dict(),
        },
        final_checkpoint_path,
    )
    
    print(f"Training completed. Final model saved to: {final_checkpoint_path}")

if __name__ == "__main__":
    main()
