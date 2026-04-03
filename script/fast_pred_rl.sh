#!/bin/bash

set -e

# 기본값 (CLI에서 덮어쓸 수 있음)
budget=1024
model=llama3-1b
rl_checkpoint=runs/a2sf_rl/policy_final.pt
gpus_per_model=1

usage() {
  echo "Usage: $0 [-b budget] [-m model] [-c rl_checkpoint] [-g gpus_per_model]"
  echo "  -b: token budget (default: ${budget})"
  echo "  -m: model name (default: ${model})"
  echo "  -c: RL checkpoint path (default: ${rl_checkpoint})"
  echo "  -g: GPUs per model instance (default: ${gpus_per_model})"
}

while getopts "b:m:c:g:h" opt; do
  case "$opt" in
    b) budget="$OPTARG" ;;
    m) model="$OPTARG" ;;
    c) rl_checkpoint="$OPTARG" ;;
    g) gpus_per_model="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

echo "Running LongBench RL prediction with:"
echo "  model=${model}, budget=${budget}, rl_checkpoint=${rl_checkpoint}, gpus_per_model=${gpus_per_model}"

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
