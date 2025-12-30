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
    parser.add_argument('--model_name', type=str, default="llama3", choices=["llama", "llama2", "llama3", "opt"], help="Model name")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help="List of GPU IDs")
    
    # Context Features
    parser.add_argument('--sentence_transformer_model', type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model name")
    parser.add_argument('--context_window', type=int, default=64, help="Context window size")
    parser.add_argument('--max_context', type=int, default=128, help="Maximum context size")
    
    # PPO Hyperparameters
    parser.add_argument('--ppo_clip', type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    
    parser.add_argument('--value_coef', type=float, default=1.0, help="Value loss coefficient")
    parser.add_argument('--entropy_coef', type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm for clipping")
    
    # Training configuration
    parser.add_argument('--iterations', type=int, default=1000, help="Number of training iterations")
    parser.add_argument('--episodes_per_update', type=int, default=128, help="Number of episodes per PPO update")
    parser.add_argument('--update_epochs', type=int, default=4, help="Number of update epochs")
    parser.add_argument('--minibatch_size', type=int, default=64, help="Minibatch size")
    
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
    config.sentence_transformer_model = args.sentence_transformer_model
    config.context_window = args.context_window
    config.max_context = args.max_context
    
    # PPO Hyperparameters
    config.ppo_clip = args.ppo_clip
    config.lr = args.lr
    config.value_coef = args.value_coef
    config.entropy_coef = args.entropy_coef
    config.max_grad_norm = args.max_grad_norm
    
    # Training configuration
    config.episodes_per_update = args.episodes_per_update
    config.update_epochs = args.update_epochs
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
    print(f"  Context window: {config.context_window}, Max context: {config.max_context}")
    print(f"  Episodes per update: {config.episodes_per_update}")
    print(f"  Update epochs: {config.update_epochs}")
    print(f"  Minibatch size: {config.minibatch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  PPO clip: {config.ppo_clip}")
    print(f"  Value coef: {config.value_coef}, Entropy coef: {config.entropy_coef}")
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
