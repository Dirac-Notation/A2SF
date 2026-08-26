#!/usr/bin/env bash
# 8B champion seed sweep: train+eval listwise+GT recipe across seeds 1-12, 4 GPUs parallel.
cd /home/smp9898/A2SF
PY=$HOME/miniconda3/envs/A2SF/bin/python
one() {
  local seed=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu $PY RL/train.py --model llama3-8b --budget 128 \
    --states_file runs/states/old_8b_none.pt --data_file datasets/training/scored/llama3-8b/train.jsonl \
    --val_data_file datasets/training/scored/llama3-8b/validation.jsonl \
    --score_field action_scores_gt_by_budget --val_score_field action_scores_gt_by_budget \
    --mini_attn_ckpt runs/mini_attn_v5_8b/mini_attn_best.pt --extra_view none --loss listwise --loss_temp 0.1 \
    --epochs 200 --ucb_topk 4 --ucb_beta 1.0 --seed "$seed" --save_dir "runs/sw8b_$seed" >/dev/null 2>&1
  CUDA_VISIBLE_DEVICES=$gpu $PY script/fast_lb_eval.py --rl_checkpoint "runs/sw8b_$seed/policy_best.pt" \
    --run_name "sw8b_$seed" --budget 128 --states_path runs/fast_lb_eval/lb_states_8b_myv_none.pt \
    --index_path runs/fast_lb_eval/index_llama3-8b_128.pt 2>/dev/null | grep "Overall Average" | sed "s/^/seed $seed /"
}
worker() { local gpu=$1; shift; for s in "$@"; do one "$s" "$gpu"; done; }
worker 2 1 2 3 &
worker 3 4 5 6 &
worker 4 7 8 9 &
worker 5 10 11 12 &
wait
echo "SWEEP8B_DONE"
