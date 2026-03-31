#!/bin/bash

set -e

# 기본값 (CLI에서 덮어쓸 수 있음)
budget=256
model=llama3
method=snap
window=16
gpus_per_model=1

usage() {
  echo "Usage: $0 [-b budget] [-m model] [-t method] [-w window] [-g gpus_per_model]"
  echo "  -b: token budget (default: ${budget})"
  echo "  -m: model name (default: ${model})"
  echo "  -t: compression method (default: ${method})"
  echo "  -w: window size (default: ${window})"
  echo "  -g: GPUs per model instance (default: ${gpus_per_model})"
}

while getopts "b:m:t:w:g:h" opt; do
  case "$opt" in
    b) budget="$OPTARG" ;;
    m) model="$OPTARG" ;;
    t) method="$OPTARG" ;;
    w) window="$OPTARG" ;;
    g) gpus_per_model="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

echo "Running LongBench prediction with:"
echo "  model=${model}, method=${method}, window=${window}, budget=${budget}, gpus_per_model=${gpus_per_model}"

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