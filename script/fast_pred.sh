#!/bin/bash

# Ctrl+C(SIGINT)를 받으면 현재 스크립트가 실행한 모든 자식 프로세스를 종료하도록 설정
trap "kill 0" EXIT

budget=256
model=llama3
method=snap
window=16

# Bin packing 기준 8 GPU 그룹 (fast_pred_rl.sh 와 동일)
# Group 1: 병목 (~106분)
CUDA_VISIBLE_DEVICES=0 python longbench.py --model $model --method $method --window $window --budget $budget --datasets narrativeqa &

# Group 2 (~70분)
CUDA_VISIBLE_DEVICES=1 python longbench.py --model $model --method $method --window $window --budget $budget --datasets repobench-p &

# Group 3 (~69분)
CUDA_VISIBLE_DEVICES=2 python longbench.py --model $model --method $method --window $window --budget $budget --datasets qmsum 2wikimqa qasper &

# Group 4 (~68분)
CUDA_VISIBLE_DEVICES=3 python longbench.py --model $model --method $method --window $window --budget $budget --datasets gov_report trec multifieldqa_en &

# Group 5 (~67분)
CUDA_VISIBLE_DEVICES=4 python longbench.py --model $model --method $method --window $window --budget $budget --datasets passage_count hotpotqa &

# Group 6 (~66분)
CUDA_VISIBLE_DEVICES=5 python longbench.py --model $model --method $method --window $window --budget $budget --datasets musique triviaqa &

# Group 7 (~60분)
CUDA_VISIBLE_DEVICES=6 python longbench.py --model $model --method $method --window $window --budget $budget --datasets multi_news passage_retrieval_en &

# Group 8 (~36분, 잔여·경량)
CUDA_VISIBLE_DEVICES=7 python longbench.py --model $model --method $method --window $window --budget $budget --datasets samsum lcc &

# 모든 백그라운드 작업이 끝날 때까지 기다립니다.
wait

# 모든 예측 작업이 완료된 후 평가 실행
echo ""
echo "All prediction tasks completed. Running evaluation..."
output_dir="result_txt/pred/${model}_${method}_${window}_${budget}"
python longbench_eval.py $output_dir