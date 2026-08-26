#!/usr/bin/env bash
# one_cv.sh <pref> <gpu> <epochs> [--no_single_view]
cd /home/smp9898/A2SF
PY=$HOME/miniconda3/envs/A2SF/bin/python
pref=$1; gpu=$2; ep=$3; shift 3; sv="$@"
export CUDA_VISIBLE_DEVICES=$gpu
$PY RL/train.py --model llama3-1b --budget 128 --states_file runs/states/${pref}_train.pt \
  --data_file datasets/cv/${pref}_train.jsonl --val_data_file datasets/cv/${pref}_val.jsonl \
  --score_field action_scores_gt_by_budget --val_score_field action_scores_gt_by_budget \
  --mini_attn_ckpt runs/mini_attn_v5_8b/mini_attn_best.pt --extra_view none --loss listwise --loss_temp 0.1 \
  --epochs $ep --ucb_topk 4 --ucb_beta 1.0 --seed 42 --save_dir runs/${pref}_r $sv >/dev/null 2>&1
$PY script/fast_lb_eval.py --rl_checkpoint runs/${pref}_r/policy_best.pt --run_name ${pref} --budget 128 \
  --states_path runs/fast_lb_eval/${pref}_test_states.pt --index_path runs/fast_lb_eval/${pref}_test_index.pt 2>/dev/null \
  | grep "Overall Average" | sed "s/^/${pref}(ep${ep}): /"
rm -rf runs/${pref}_r
