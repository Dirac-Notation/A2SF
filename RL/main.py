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
            dtype=torch.float32,
        )
    )
    # b: sigmoid shift parameter
    b_values: torch.Tensor = field(
        default_factory=lambda: torch.tensor(
            [1, 8, 16, 32, 64, 128],
            dtype=torch.float32,
        )
    )
    
    # ----- NeuralUCB Hyperparameters -----
    lr: float = 1e-2
    ucb_beta: float = 0.5  # Exploration parameter for UCB
    l2_coef: float = 1e-6  # L2 regularization coefficient for weight decay
    
    # ----- Learning Rate Scheduler -----
    scheduler_T_max: int = 1000  # For CosineAnnealingLR: maximum iterations
    
    # ----- Training Configuration -----
    iterations: int = 1000  # Number of training iterations
    episodes_per_update: int = 16  # Number of episodes per update
    
    # ----- Evaluation Configuration -----
    eval_frequency: int = 50
    eval_samples: int = 100
    
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
            ucb_beta=default_config.ucb_beta,
            l2_coef=default_config.l2_coef,
            iterations=default_config.iterations,
            episodes_per_update=default_config.episodes_per_update,
            eval_frequency=default_config.eval_frequency,
            eval_samples=default_config.eval_samples,
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
    print(f"  LR Scheduler: cosine (T_max: {config.scheduler_T_max})")
    print(f"  UCB beta: {config.ucb_beta}")
    print(f"  Save directory: {config.save_dir}")
    print()
    
    # Create trainer
    trainer = A2SFTrainer(config)
    
    # Resume from checkpoint if specified
    start_iteration = 0
    if config.resume:
        start_iteration = trainer.load_checkpoint(config.resume)
        print(f"Resuming training from iteration {start_iteration}")
    
    # Train
    trainer.train(num_iterations=config.iterations)
    
    # Save final model
    final_checkpoint_path = os.path.join(config.save_dir, "policy_final.pt")
    torch.save({
        "policy_state_dict": trainer.policy.state_dict(),
        "attention_encoder_state_dict": trainer.env.context_encoder.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "config": config,
    }, final_checkpoint_path)
    
    print(f"Training completed. Final model saved to: {final_checkpoint_path}")

if __name__ == "__main__":
    main()
