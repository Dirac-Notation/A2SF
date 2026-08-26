#!/usr/bin/env bash
# 8B champion recovery sweep: maximize LB across encoder(1v/2v) x reward(MaxO/GT) x loss(mse/listwise) x seed.
cd /home/smp9898/A2SF
PY=$HOME/miniconda3/envs/A2SF/bin/python
# config defs: "tag states lbstates reward loss"
declare -A CFG=(
  [2vMaxOmse]="runs/states/old_8b_2v_none.pt runs/fast_lb_eval/lb_states_8b_2v_none.pt action_scores_maxo_by_budget mse"
  [2vMaxOlw]="runs/states/old_8b_2v_none.pt runs/fast_lb_eval/lb_states_8b_2v_none.pt action_scores_maxo_by_budget listwise"
  [2vGTlw]="runs/states/old_8b_2v_none.pt runs/fast_lb_eval/lb_states_8b_2v_none.pt action_scores_gt_by_budget listwise"
  [1vMaxOmse]="runs/states/old_8b_none.pt runs/fast_lb_eval/lb_states_8b_myv_none.pt action_scores_maxo_by_budget mse"
  [1vMaxOlw]="runs/states/old_8b_none.pt runs/fast_lb_eval/lb_states_8b_myv_none.pt action_scores_maxo_by_budget listwise"
)
run() {
  local tag=$1 seed=$2 gpu=$3
  read st lb rew loss <<< "${CFG[$tag]}"
  local dir="runs/r8b_${tag}_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu $PY RL/train.py --model llama3-8b --budget 128 \
    --states_file "$st" --data_file datasets/training/scored/llama3-8b/train.jsonl \
    --val_data_file datasets/training/scored/llama3-8b/validation.jsonl \
    --score_field "$rew" --val_score_field "$rew" \
    --mini_attn_ckpt runs/mini_attn_v5_8b/mini_attn_best.pt --extra_view none --loss "$loss" --loss_temp 0.1 \
    --epochs 200 --ucb_topk 4 --ucb_beta 1.0 --seed "$seed" --save_dir "$dir" >/dev/null 2>&1
  local lbv=$(CUDA_VISIBLE_DEVICES=$gpu $PY script/fast_lb_eval.py --rl_checkpoint "$dir/policy_best.pt" \
    --run_name "$tag" --budget 128 --states_path "$lb" \
    --index_path runs/fast_lb_eval/index_llama3-8b_128.pt 2>/dev/null | grep "Overall Average" | grep -oE "[0-9]+\.[0-9]+")
  echo "$tag seed $seed LB $lbv"
  rm -rf "$dir"
}
export -f run; export PY
# build job list: all configs x seeds 1..6
JOBS=()
for tag in 2vMaxOmse 2vMaxOlw 2vGTlw 1vMaxOmse 1vMaxOlw; do
  for s in 1 2 3 4 5 6; do JOBS+=("$tag $s"); done
done
# 6 GPUs (2-7), round-robin
i=0
for job in "${JOBS[@]}"; do
  gpu=$(( 2 + i % 6 ))
  ( run $job $gpu ) &
  i=$((i+1))
  if (( i % 6 == 0 )); then wait; fi
done
wait
echo "SWEEP8B_RECOVER_DONE"
