#!/bin/bash

# Ctrl+C(SIGINT)를 받으면 현재 스크립트가 실행한 모든 자식 프로세스를 종료하도록 설정
trap "kill 0" EXIT

budget=128
model=llama3
rl_checkpoint=policy_950.pt

# 16개 데이터셋을 4개 그룹으로 나눔 (code와 summarization 데이터셋이 한 그룹에 몰리지 않도록 분산)
# 그룹 1: repobench-p (code), narrativeqa, hotpotqa, trec
CUDA_VISIBLE_DEVICES=0,1 python longbench_RL.py --model $model --budget $budget --datasets repobench-p narrativeqa hotpotqa trec --rl_checkpoint $rl_checkpoint --skip_eval &

# 그룹 2: lcc (code), qasper, 2wikimqa, triviaqa
CUDA_VISIBLE_DEVICES=2,3 python longbench_RL.py --model $model --budget $budget --datasets lcc qasper 2wikimqa passage_retrieval_en --rl_checkpoint $rl_checkpoint --skip_eval &

# 그룹 3: gov_report (summarization), multifieldqa_en, musique, samsum
CUDA_VISIBLE_DEVICES=4,5 python longbench_RL.py --model $model --budget $budget --datasets gov_report multifieldqa_en triviaqa samsum --rl_checkpoint $rl_checkpoint --skip_eval &

# 그룹 4: qmsum (summarization), multi_news, passage_retrieval_en, passage_count
CUDA_VISIBLE_DEVICES=6,7 python longbench_RL.py --model $model --budget $budget --datasets qmsum multi_news musique passage_count --rl_checkpoint $rl_checkpoint --skip_eval &

# 모든 백그라운드 작업이 끝날 때까지 기다립니다.
wait

# 모든 예측 작업이 완료된 후 평가 실행
echo ""
echo "All prediction tasks completed. Running evaluation..."
output_dir="result_txt/pred/${model}_sigmoid_${budget}_RL"
python longbench_eval.py $output_dir