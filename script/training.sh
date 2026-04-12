#!/bin/bash

# Example training script for A2SF RL agent

usage() {
    echo "Usage: bash RL/training.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -g, --gpu                  GPU device ID (default: 0)"
    echo "  -s, --save_dir             Directory to save checkpoints and logs (default: runs/a2sf_rl)"
    echo "  -b, --token_budget         Token budget for KV cache compression (default: 128)"
    echo "  -d, --train_data_path      Path to training data JSONL file (default: RL/training/data/training_data.jsonl)"
    echo "  -e, --episodes_per_update  Number of episodes per parameter update (default: 32)"
    echo "  -h, --help                 Show this help message"
    exit 0
}

gpu=0
save_dir="runs/a2sf_rl"
token_budget=128
train_data_path="RL/training/data/training_data.jsonl"
episodes_per_update=32
model="llama3-1b"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        -g|--gpu) gpu="$2"; shift 2 ;;
        -s|--save_dir) save_dir="$2"; shift 2 ;;
        -b|--token_budget) token_budget="$2"; shift 2 ;;
        -d|--train_data_path) train_data_path="$2"; shift 2 ;;
        -e|--episodes_per_update) episodes_per_update="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

echo "Starting A2SF RL Training Example"
echo "================================="
echo "GPU: $gpu"
echo "Save dir: $save_dir"
echo "Token budget: $token_budget"
echo "Train data path: $train_data_path"
echo "Episodes per update: $episodes_per_update"

echo "Running basic training..."

CUDA_VISIBLE_DEVICES=$gpu python -m RL.training.run \
    --model $model \
    --save_dir $save_dir \
    --token_budget $token_budget \
    --train_data_path $train_data_path \
    --episodes_per_update $episodes_per_update

echo "Training completed!"
echo "Check the results in $save_dir/"
