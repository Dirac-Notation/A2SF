#!/bin/bash

set -e

budget=256
model=llama3
method=snap
window=16
gpus_per_model=1

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