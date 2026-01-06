#!/bin/bash

# Example training script for A2SF RL agent

echo "Starting A2SF RL Training Example"
echo "================================="

# Basic training example
echo "Running basic training..."
python RL/run_training.py \
    --model llama3 \
    --gpus 0 \
    --value_coef 1.0 \
    --entropy_coef 0.01 \
    --max_grad_norm 1.0 \
    --episodes_per_update 64 \
    --update_epochs 4 \
    --minibatch_size 16 \
    --eval_frequency 100 \
    --eval_samples 64 \
    --rbo_p 0.95 \
    --iterations 1600 \

echo "Training completed!"
echo "Check the results in runs/a2sf_rl_example/"
