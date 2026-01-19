#!/bin/bash

# Example training script for A2SF RL agent

echo "Starting A2SF RL Training Example"
echo "================================="

# Basic training example
echo "Running basic training..."
python -m RL.main \
    --model llama3 \
    --gpu 0 \
    --save_dir runs/a2sf_rl_test

echo "Training completed!"
echo "Check the results in runs/a2sf_rl_test/"
