#!/bin/bash

# Example training script for A2SF RL agent

echo "Starting A2SF RL Training Example"
echo "================================="

# Basic training example
echo "Running basic training..."
python RL/run_training.py \
    --model llama3 \
    --gpus 0 \
    --iterations 1600 \
    --episodes_per_update 1 \
    --lr 3e-4 \
    --accuracy_weight 1.0 \
    --max_samples_per_task 10 \
    --eval_frequency 100 \
    --save_dir runs/a2sf_rl_example

echo "Training completed!"
echo "Check the results in runs/a2sf_rl_example/"
