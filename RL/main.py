#!/usr/bin/env python3
"""
Main script for training A2SF RL agent
"""

import argparse
import os
import json
import torch

from .config import A2SFRLConfig
from .trainer import A2SFTrainer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train A2SF RL Agent")
    
    # Model configuration
    parser.add_argument('--model', type=str, default="llama3", choices=["llama", "llama2", "llama3", "opt"], help="Model name")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help="List of GPU IDs")
    
    # Training configuration
    parser.add_argument('--iterations', type=int, default=1000, help="Number of training iterations")
    parser.add_argument('--episodes_per_update', type=int, default=128, help="Number of episodes per PPO update")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--minibatch_size', type=int, default=64, help="Minibatch size")
    
    # Evaluation configuration
    parser.add_argument('--eval_frequency', type=int, default=100, help="Evaluation frequency")
    parser.add_argument('--eval_samples', type=int, default=50, help="Number of samples for evaluation")
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default="runs/a2sf_rl", help="Directory to save checkpoints and logs")
    parser.add_argument('--log_frequency', type=int, default=5, help="Logging frequency")
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    
    return parser.parse_args()

def create_config_from_args(args) -> A2SFRLConfig:
    """Create configuration from command line arguments"""
    config = A2SFRLConfig()
    
    # Model configuration
    config.model_name = args.model
    config.gpus = args.gpus
    
    # Training configuration
    config.episodes_per_update = args.episodes_per_update
    config.lr = args.lr
    config.minibatch_size = args.minibatch_size
    
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
    args = parse_args()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Print configuration
    print("A2SF RL Training Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  GPUs: {config.gpus}")
    print(f"  Episodes per update: {config.episodes_per_update}")
    print(f"  Learning rate: {config.lr}")
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
        "training_stats": trainer.training_stats
    }, final_checkpoint_path)
    
    print(f"Training completed. Final model saved to: {final_checkpoint_path}")

if __name__ == "__main__":
    main()
