#!/bin/bash

set -e

budget=1024
model=llama3
rl_checkpoint=policy_final.pt
gpus_per_model=1

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
