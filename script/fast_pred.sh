#!/bin/bash

set -e

usage() {
    echo "Usage: bash script/fast_pred.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -b, --budget          Token budget (default: 256)"
    echo "  -m, --model           Model name (default: llama3-1b)"
    echo "  -t, --method          Compression method (default: snap)"
    echo "  -w, --window          Window size (default: 16)"
    echo "  -g, --gpus_per_model  GPUs per model instance (default: 1)"
    echo "  -h, --help            Show this help message"
    exit 0
}

budget=256
model="llama3-1b"
method="snap"
window=16
gpus_per_model=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        -b|--budget) budget="$2"; shift 2 ;;
        -m|--model) model="$2"; shift 2 ;;
        -t|--method) method="$2"; shift 2 ;;
        -w|--window) window="$2"; shift 2 ;;
        -g|--gpus_per_model) gpus_per_model="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

echo "Running LongBench prediction"
echo "================================="
echo "Model: $model"
echo "Method: $method"
echo "Window: $window"
echo "Budget: $budget"
echo "GPUs per model: $gpus_per_model"

# longbench.py 내부에서 보이는 GPU들을 모두 사용해 멀티프로세스로 처리합니다.
python longbench.py \
    --model "$model" \
    --method "$method" \
    --window "$window" \
    --budget "$budget" \
    --gpus_per_model "$gpus_per_model"

echo ""
echo "All prediction tasks completed. Running evaluation..."
output_dir="result_txt/pred/${model}_${method}_${window}_${budget}"
python longbench_eval.py "$output_dir"
