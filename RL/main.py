#!/usr/bin/env python3
"""
Main script for training A2SF RL agent
"""

import argparse
import os
import json
import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class A2SFRLConfig:
    # ----- Model Configuration -----
    model_name: str = "llama3"  # llama, llama2, llama3, opt
    gpus: List[int] = field(default_factory=lambda: [0])
    
    # ----- Context Features -----
    context_encoder_model: str = "jinaai/jina-embeddings-v2-small-en"
    
    # ----- Policy Action Spaces -----
    a_values: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0, 0.001, 0.01, 0.1, 10.0]))
    b_values: torch.Tensor = field(default_factory=lambda: torch.tensor([1, 2, 4, 8, 16, 32, 64, 128, 512, 1024, 4096, 8192]))
    
    # ----- NeuralUCB Hyperparameters -----
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    ucb_beta: float = 1.0  # Exploration parameter for UCB
    
    # Training configuration
    episodes_per_update: int = 32
    
    # ----- Evaluation Configuration -----
    eval_frequency: int = 100
    eval_samples: int = 50
    
    # ----- Misc -----
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "runs/a2sf_rl"
    log_frequency: int = 5

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train A2SF RL Agent")
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default="llama3", choices=["llama", "llama2", "llama3", "opt"], help="Model name")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help="List of GPU IDs")
    
    # Context Features
    parser.add_argument('--context_encoder_model', type=str, default="jinaai/jina-embeddings-v2-small-en", help="Context encoder model name")
    
    # NeuralUCB Hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--ucb_beta', type=float, default=1.0, help="UCB exploration parameter")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm for clipping")
    
    # Training configuration
    parser.add_argument('--iterations', type=int, default=1000, help="Number of training iterations")
    parser.add_argument('--episodes_per_update', type=int, default=128, help="Number of episodes per update")
    
    # Evaluation configuration
    parser.add_argument('--eval_frequency', type=int, default=100, help="Evaluation frequency")
    parser.add_argument('--eval_samples', type=int, default=50, help="Number of samples for evaluation")
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default="runs/a2sf_rl", help="Directory to save checkpoints and logs")
    parser.add_argument('--log_frequency', type=int, default=1, help="Logging frequency")
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    
    return parser.parse_args()

def create_config_from_args(args) -> A2SFRLConfig:
    """Create configuration from command line arguments"""
    config = A2SFRLConfig()
    
    # Model configuration
    config.model_name = args.model_name
    config.gpus = args.gpus
    
    # Context Features
    config.context_encoder_model = args.context_encoder_model
    
    # NeuralUCB Hyperparameters
    config.lr = args.lr
    config.ucb_beta = args.ucb_beta
    config.max_grad_norm = args.max_grad_norm
    
    # Training configuration
    config.episodes_per_update = args.episodes_per_update
    
    # Evaluation configuration
    config.eval_frequency = args.eval_frequency
    config.eval_samples = args.eval_samples
    
    # Logging and saving
    config.save_dir = args.save_dir
    config.log_frequency = args.log_frequency
    
    # Misc
    config.seed = args.seed
    config.device = args.device
    
    return config

def main():
    """Main function"""
    # Import here to avoid circular import
    from .trainer import A2SFTrainer
    
    args = parse_args()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Print configuration
    print("A2SF RL Training Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  GPUs: {config.gpus}")
    print(f"  Context encoder: {config.context_encoder_model}")
    print(f"  Episodes per update: {config.episodes_per_update}")
    print(f"  Learning rate: {config.lr}")
    print(f"  UCB beta: {config.ucb_beta}")
    print(f"  Save directory: {config.save_dir}")
    print()
    
    # Create trainer
    trainer = A2SFTrainer(config)
    
    # Resume from checkpoint if specified
    start_iteration = 0
    if args.resume:
        start_iteration = trainer.load_checkpoint(args.resume)
        print(f"Resuming training from iteration {start_iteration}")
    
    # Train
    trainer.train(num_iterations=args.iterations)
    
    # Save final model
    final_checkpoint_path = os.path.join(config.save_dir, "policy_final.pt")
    torch.save({
        "policy_state_dict": trainer.policy.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "config": config,
    }, final_checkpoint_path)
    
    print(f"Training completed. Final model saved to: {final_checkpoint_path}")

if __name__ == "__main__":
    main()
