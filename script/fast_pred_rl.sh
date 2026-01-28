#!/bin/bash

# Ctrl+C(SIGINT)를 받으면 현재 스크립트가 실행한 모든 자식 프로세스를 종료하도록 설정
trap "kill 0" EXIT

budget=128
model=llama3
rl_checkpoint=policy_500.pt

python longbench_RL.py --gpus 0 1 --model $model --budget $budget --task 0 1 --rl_checkpoint $rl_checkpoint --skip_eval &

python longbench_RL.py --gpus 2 3 --model $model --budget $budget --task 2 3 4 --rl_checkpoint $rl_checkpoint --skip_eval &

python longbench_RL.py --gpus 4 5 --model $model --budget $budget --task 5 --rl_checkpoint $rl_checkpoint &

# 모든 백그라운드 작업이 끝날 때까지 기다립니다.
wait

# 모든 예측 작업이 완료된 후 평가 실행
echo ""
echo "All prediction tasks completed. Running evaluation..."
output_dir="result_txt/pred/${model}_sigmoid_${budget}_RL"
python longbench_eval.py $output_dir