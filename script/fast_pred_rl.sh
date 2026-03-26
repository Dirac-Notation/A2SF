#!/bin/bash

trap "kill 0" EXIT

budget=128
model=llama3
rl_checkpoint=policy_3477.pt

# GPU 0: 가장 무거운 작업 (약 48분)
CUDA_VISIBLE_DEVICES=0 python longbench_RL.py --model $model --budget $budget --datasets gov_report --rl_checkpoint $rl_checkpoint &

# GPU 1: 두 번째로 무거운 작업 (약 44분)
CUDA_VISIBLE_DEVICES=1 python longbench_RL.py --model $model --budget $budget --datasets qmsum --rl_checkpoint $rl_checkpoint &

# GPU 2: 세 번째로 무거운 작업 (약 42분)
CUDA_VISIBLE_DEVICES=2 python longbench_RL.py --model $model --budget $budget --datasets repobench-p --rl_checkpoint $rl_checkpoint &

# GPU 3: 중량급 + 경량급 조합 (약 45.7분)
CUDA_VISIBLE_DEVICES=3 python longbench_RL.py --model $model --budget $budget --datasets multi_news multifieldqa_en --rl_checkpoint $rl_checkpoint &

# GPU 4: 중급 3개 조합 (약 49.6분)
CUDA_VISIBLE_DEVICES=4 python longbench_RL.py --model $model --budget $budget --datasets lcc hotpotqa 2wikimqa --rl_checkpoint $rl_checkpoint &

# GPU 5: 중급 3개 조합 (약 50.9분)
CUDA_VISIBLE_DEVICES=5 python longbench_RL.py --model $model --budget $budget --datasets samsum narrativeqa triviaqa --rl_checkpoint $rl_checkpoint &

# GPU 6: 중급 3개 조합 (약 50.5분)
CUDA_VISIBLE_DEVICES=6 python longbench_RL.py --model $model --budget $budget --datasets musique passage_retrieval_en trec --rl_checkpoint $rl_checkpoint &

# GPU 7: 중급 2개 조합 (약 33.4분)
CUDA_VISIBLE_DEVICES=7 python longbench_RL.py --model $model --budget $budget --datasets passage_count qasper --rl_checkpoint $rl_checkpoint &

# 모든 백그라운드 작업이 끝날 때까지 기다립니다.
wait

# 모든 예측 작업이 완료된 후 평가 실행
echo ""
echo "All prediction tasks completed. Running evaluation..."
output_dir="result_txt/pred/${model}_sigmoid_${budget}_RL"
python longbench_eval.py $output_dir