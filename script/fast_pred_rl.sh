#!/bin/bash

set -e

usage() {
    echo "Usage: bash script/fast_pred_rl.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -b, --budget          Token budget (default: 1024)"
    echo "  -m, --model           Model name (default: llama3-1b)"
    echo "  -c, --rl_checkpoint   RL checkpoint path (default: runs/a2sf_rl/policy_final.pt)"
    echo "  -g, --gpus_per_model  GPUs per model instance (default: 1)"
    echo "  -h, --help            Show this help message"
    exit 0
}

budget=1024
model="llama3-1b"
rl_checkpoint="runs/a2sf_rl/policy_final.pt"
gpus_per_model=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        -b|--budget) budget="$2"; shift 2 ;;
        -m|--model) model="$2"; shift 2 ;;
        -c|--rl_checkpoint) rl_checkpoint="$2"; shift 2 ;;
        -g|--gpus_per_model) gpus_per_model="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

echo "Running LongBench RL prediction"
echo "================================="
echo "Model: $model"
echo "Budget: $budget"
echo "RL checkpoint: $rl_checkpoint"
echo "GPUs per model: $gpus_per_model"

# longbench_RL.py 내부에서 보이는 GPU들을 모두 사용해 멀티프로세스로 처리합니다.
python longbench_RL.py \
    --model "$model" \
    --budget "$budget" \
    --rl_checkpoint "$rl_checkpoint" \
    --gpus_per_model "$gpus_per_model"

echo ""
echo "All prediction tasks completed. Running evaluation..."
output_dir="result_txt/pred/${model}_sigmoid_${budget}_RL"
python longbench_eval.py "$output_dir"
