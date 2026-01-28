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
    gpus: List[int] = field(default_factory=lambda: [0])
    
    # ----- Policy Action Space -----
    # Discrete candidate values for A2SF forgetting factor (shared across layers)
    forgetting_values: torch.Tensor = field(
        default_factory=lambda: torch.tensor(
            [0.0, 0.35, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0],
            dtype=torch.float32,
        )
    )
    
    # ----- NeuralUCB Hyperparameters -----
    lr: float = 1e-2
    ucb_beta: float = 1.0  # Exploration parameter for UCB
    l2_coef: float = 1e-5  # L2 regularization coefficient for weight decay
    
    # ----- Learning Rate Scheduler -----
    scheduler_type: str = "cosine"  # "step", "cosine", "exponential", "none"
    scheduler_step_size: int = 200  # For StepLR: decay every N iterations
    scheduler_gamma: float = 0.5  # For StepLR/ExponentialLR: decay factor
    scheduler_T_max: int = 3000  # For CosineAnnealingLR: maximum iterations
    
    # ----- Training Configuration -----
    iterations: int = 3000  # Number of training iterations
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
        """Device is automatically determined from gpus"""
        if torch.cuda.is_available() and len(self.gpus) > 0:
            return f"cuda:{self.gpus[0]}"
        return "cpu"
    
    @classmethod
    def from_args(cls):
        """Create configuration from command line arguments (minimal args: gpu, model, save_dir)"""
        # Create default config instance to get default values
        default_config = cls()
        
        parser = argparse.ArgumentParser(description="Train A2SF RL Agent")

        # Minimal command line arguments
        parser.add_argument('--gpu', type=int, nargs='+', default=default_config.gpus, help="GPU ID(s) to use (default: [0])")
        parser.add_argument('--model', type=str, default=default_config.model, choices=["llama", "llama2", "llama3", "opt"], help="Model name")
        parser.add_argument('--save_dir', type=str, default=default_config.save_dir, help="Directory to save checkpoints and logs")
        parser.add_argument('--resume', type=str, default=default_config.resume, help="Path to checkpoint to resume from (e.g., runs/a2sf_rl/policy_300.pt)")

        args = parser.parse_args()
        
        # Get seed from environment variable if set, otherwise use default
        seed = int(default_config.seed)
        
        # Create config from args
        # Handle gpu: nargs='+' returns list, but ensure it's always a list
        gpus = args.gpu if isinstance(args.gpu, list) else [args.gpu]

        return cls(
            model=args.model,
            gpus=gpus,
            save_dir=args.save_dir,
            seed=seed,
            # All other fields use defaults
            forgetting_values=default_config.forgetting_values,
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
    print("A2SF RL Training Configuration:")
    print(f"  Model: {config.model}")
    print(f"  GPUs: {config.gpus}")
    print(f"  Device: {config.device}")
    print(f"  Seed: {config.seed}")
    print(f"  Episodes per update: {config.episodes_per_update}")
    print(f"  Learning rate: {config.lr}")
    print(f"  LR Scheduler: {config.scheduler_type}")
    if config.scheduler_type != "none":
        if config.scheduler_type == "step":
            print(f"    Step size: {config.scheduler_step_size}, Gamma: {config.scheduler_gamma}")
        elif config.scheduler_type == "cosine":
            print(f"    T_max: {config.scheduler_T_max}")
        elif config.scheduler_type == "exponential":
            print(f"    Gamma: {config.scheduler_gamma}")
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
