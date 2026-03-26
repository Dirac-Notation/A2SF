#!/bin/bash

# Ctrl+C(SIGINT)를 받으면 현재 스크립트가 실행한 모든 자식 프로세스를 종료하도록 설정
trap "kill 0" EXIT

budget=128
model=llama3
method=snap
window=16

# GPU 0: 가장 무거운 작업
CUDA_VISIBLE_DEVICES=0 python longbench.py --model $model --method $method --window $window --budget $budget --datasets gov_report --skip_eval &

# GPU 1: 두 번째로 무거운 작업
CUDA_VISIBLE_DEVICES=1 python longbench.py --model $model --method $method --window $window --budget $budget --datasets qmsum --skip_eval &

# GPU 2: 세 번째로 무거운 작업
CUDA_VISIBLE_DEVICES=2 python longbench.py --model $model --method $method --window $window --budget $budget --datasets repobench-p --skip_eval &

# GPU 3: 중량급 + 경량급 조합
CUDA_VISIBLE_DEVICES=3 python longbench.py --model $model --method $method --window $window --budget $budget --datasets multi_news multifieldqa_en --skip_eval &

# GPU 4: 중급 3개 조합
CUDA_VISIBLE_DEVICES=4 python longbench.py --model $model --method $method --window $window --budget $budget --datasets lcc hotpotqa 2wikimqa --skip_eval &

# GPU 5: 중급 3개 조합
CUDA_VISIBLE_DEVICES=5 python longbench.py --model $model --method $method --window $window --budget $budget --datasets samsum narrativeqa triviaqa --skip_eval &

# GPU 6: 중급 3개 조합
CUDA_VISIBLE_DEVICES=6 python longbench.py --model $model --method $method --window $window --budget $budget --datasets musique passage_retrieval_en trec --skip_eval &

# GPU 7: 중급 2개 조합
CUDA_VISIBLE_DEVICES=7 python longbench.py --model $model --method $method --window $window --budget $budget --datasets passage_count qasper --skip_eval &

# 모든 백그라운드 작업이 끝날 때까지 기다립니다.
wait

# 모든 예측 작업이 완료된 후 평가 실행
echo ""
echo "All prediction tasks completed. Running evaluation..."
output_dir="result_txt/pred/${model}_${method}_${window}_${budget}"
python longbench_eval.py $output_dir