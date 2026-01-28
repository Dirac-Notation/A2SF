#!/bin/bash

# Example training script for A2SF RL agent

gpu="6,7"
model="llama3"
save_dir="runs/a2sf_rl"

echo "Starting A2SF RL Training Example"
echo "================================="

# Basic training example
echo "Running basic training..."

CUDA_VISIBLE_DEVICES=$gpu python -m RL.main \
    --model $model \
    --save_dir $save_dir

echo "Training completed!"
echo "Check the results in $save_dir/"
