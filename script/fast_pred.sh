#!/bin/bash

# Ctrl+C(SIGINT)를 받으면 현재 스크립트가 실행한 모든 자식 프로세스를 종료하도록 설정
trap "kill 0" EXIT

budget=128
method=snap
model=llama2
window=16

CUDA_VISIBLE_DEVICES=0,1 python longbench.py --model $model --method $method --budget $budget --window $window --task 0 1 &

CUDA_VISIBLE_DEVICES=2,3 python longbench.py --model $model --method $method --budget $budget --window $window --task 2 3 4 &

CUDA_VISIBLE_DEVICES=4,5 python longbench.py --model $model --method $method --budget $budget --window $window --task 5 &

# 모든 백그라운드 작업이 끝날 때까지 기다립니다.
wait

# 모든 예측 작업이 완료된 후 평가 실행
echo ""
echo "All prediction tasks completed. Running evaluation..."
output_dir="result_txt/pred/${model}_${method}_${window}_${budget}"
python longbench_eval.py $output_dir