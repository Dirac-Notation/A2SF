#!/bin/bash

# Example training script for A2SF RL agent

echo "Starting A2SF RL Training Example"
echo "================================="

# Basic training example
echo "Running basic training..."
python -m RL.main \
    --model_name llama3 \
    --gpus 0 \
    --lr 3e-4 \
    --ucb_beta 0.2 \
    --max_grad_norm 1.0 \
    --episodes_per_update 32 \
    --eval_frequency 100 \
    --eval_samples 64 \
    --iterations 1000 \
    --save_dir runs/a2sf_rl_example

echo "Training completed!"
echo "Check the results in runs/a2sf_rl_example/"
