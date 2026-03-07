#!/bin/bash

# Ctrl+C(SIGINT)를 받으면 현재 스크립트가 실행한 모든 자식 프로세스를 종료하도록 설정
trap "kill 0" EXIT

budget=512
model=llama3
rl_checkpoint=policy_1700.pt

# 요청한 실행 시간 기준으로 4개 그룹 구성
# 그룹 1: 약 158분 (가장 무거운 gov_report 중심)
CUDA_VISIBLE_DEVICES=0,1 python longbench_RL.py --model $model --budget $budget --datasets gov_report lcc 2wikimqa --rl_checkpoint $rl_checkpoint --skip_eval &

# 그룹 2: 약 168분
CUDA_VISIBLE_DEVICES=2,3 python longbench_RL.py --model $model --budget $budget --datasets qmsum samsum musique triviaqa --rl_checkpoint $rl_checkpoint --skip_eval &

# 그룹 3: 약 164분
CUDA_VISIBLE_DEVICES=4,5 python longbench_RL.py --model $model --budget $budget --datasets multi_news qasper passage_count hotpotqa --rl_checkpoint $rl_checkpoint --skip_eval &

# 그룹 4: 약 170분 (작은 데이터셋들을 묶어 밸런싱)
CUDA_VISIBLE_DEVICES=6,7 python longbench_RL.py --model $model --budget $budget --datasets repobench-p passage_retrieval_en trec narrativeqa multifieldqa_en --rl_checkpoint $rl_checkpoint --skip_eval &

# 모든 백그라운드 작업이 끝날 때까지 기다립니다.
wait

# 모든 예측 작업이 완료된 후 평가 실행
echo ""
echo "All prediction tasks completed. Running evaluation..."
output_dir="result_txt/pred/${model}_sigmoid_${budget}_RL"
python longbench_eval.py $output_dir