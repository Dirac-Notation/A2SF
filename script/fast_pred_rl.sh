#!/bin/bash

trap "kill 0" EXIT

budget=1024
model=llama3
rl_checkpoint=policy_final.pt

# Bin packing 기준 8 GPU 그룹 (목표 ~67–68분/그룹, narrativeqa 단독 병목)
# Group 1: 병목 (~106분)
CUDA_VISIBLE_DEVICES=0 python longbench_RL.py --model $model --budget $budget --datasets narrativeqa --rl_checkpoint $rl_checkpoint &

# Group 2 (~70분)
CUDA_VISIBLE_DEVICES=1 python longbench_RL.py --model $model --budget $budget --datasets repobench-p --rl_checkpoint $rl_checkpoint &

# Group 3 (~69분)
CUDA_VISIBLE_DEVICES=2 python longbench_RL.py --model $model --budget $budget --datasets qmsum 2wikimqa qasper --rl_checkpoint $rl_checkpoint &

# Group 4 (~68분)
CUDA_VISIBLE_DEVICES=3 python longbench_RL.py --model $model --budget $budget --datasets gov_report trec multifieldqa_en --rl_checkpoint $rl_checkpoint &

# Group 5 (~67분)
CUDA_VISIBLE_DEVICES=4 python longbench_RL.py --model $model --budget $budget --datasets passage_count hotpotqa --rl_checkpoint $rl_checkpoint &

# Group 6 (~66분)
CUDA_VISIBLE_DEVICES=5 python longbench_RL.py --model $model --budget $budget --datasets musique triviaqa --rl_checkpoint $rl_checkpoint &

# Group 7 (~60분)
CUDA_VISIBLE_DEVICES=6 python longbench_RL.py --model $model --budget $budget --datasets multi_news passage_retrieval_en --rl_checkpoint $rl_checkpoint &

# Group 8 (~36분, 잔여·경량)
CUDA_VISIBLE_DEVICES=7 python longbench_RL.py --model $model --budget $budget --datasets samsum lcc --rl_checkpoint $rl_checkpoint &

# 모든 백그라운드 작업이 끝날 때까지 기다립니다.
wait

# 모든 예측 작업이 완료된 후 평가 실행
echo ""
echo "All prediction tasks completed. Running evaluation..."
output_dir="result_txt/pred/${model}_sigmoid_${budget}_RL"
python longbench_eval.py $output_dir
