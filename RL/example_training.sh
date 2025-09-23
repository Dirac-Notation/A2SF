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

echo "Training completed!"
echo "Check the results in runs/a2sf_rl_example/"
