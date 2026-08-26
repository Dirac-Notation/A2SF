#!/usr/bin/env bash
# ============================================================================
# REPRODUCE LB=26.94 (champion-class per-prompt RL, LLaMA-3.2-1B, budget 128)
# ----------------------------------------------------------------------------
# This is the listwise+GT configuration that hit LB 26.94 (= historical champion
# 26.95 within noise). It is a *lucky-tail* single seed: the 50-seed mean of this
# exact config is 26.60 +/- 0.31 (>27 only 2% of seeds). 26.94 is the +1 sigma
# tail, NOT reliably reproducible skill. Default seed 42 is the original run.
#
# 5-axis recipe (everything needed from scratch):
#   Dataset     : OLD data  datasets/training/scored/llama3-1b/{train,validation}.jsonl
#                 (train 3341 / val 499). Per-action LB-scored, field below.
#   Input/State : mini-attn encoder stats, extra_view=none, single-view, 53-d.
#                 Precomputed -> runs/states/old_none.pt
#                 (encoder ckpt runs/mini_attn_v5/mini_attn_best.pt)
#   Architecture: NeuralUCBAgent (MLPResidualBlock x2 + per-task linear head,
#                 13 paired-sigmoid actions). epochs 200, ucb_topk 4, ucb_beta 1.0
#   Reward      : GT  -> --score_field action_scores_gt_by_budget  (val too)
#   Loss        : listwise ranking (top-K softmax-CE, temp 0.1)   [the key lever]
#   Seed        : 42 (train.py default; original run used default)
#   Eval        : fast_lb_eval, CONSISTENT lb_states_myv_none (my-config 53-d)
#                 + index_llama3-1b_128.pt -> LB-GT (no decode). Train/eval encoder
#                 config MUST match (the eval-consistency bug otherwise deflates LB).
#
# Comparators (same data/input/eval): MaxO+MSE = 26.73, listwise+GT = 26.94 (+0.21).
# NOTE listwise+GT is best on OLD data but WORST (24.86) on faithful data -> the
# combo is data-specific, not universal.
# ============================================================================
set -e
cd "$(dirname "$0")/.."
SEED=${1:-42}
RUN=runs/repro_2694_seed${SEED}

python RL/train.py --model llama3-1b --budget 128 \
  --states_file     runs/states/old_none.pt \
  --data_file       datasets/training/scored/llama3-1b/train.jsonl \
  --val_data_file   datasets/training/scored/llama3-1b/validation.jsonl \
  --score_field     action_scores_gt_by_budget \
  --val_score_field action_scores_gt_by_budget \
  --mini_attn_ckpt  runs/mini_attn_v5/mini_attn_best.pt \
  --extra_view none --loss listwise --loss_temp 0.1 \
  --epochs 200 --ucb_topk 4 --ucb_beta 1.0 --seed ${SEED} \
  --save_dir ${RUN}

python script/fast_lb_eval.py \
  --rl_checkpoint ${RUN}/policy_best.pt --run_name repro_2694_seed${SEED} \
  --budget 128 \
  --states_path runs/fast_lb_eval/lb_states_myv_none.pt \
  --index_path  runs/fast_lb_eval/index_llama3-1b_128.pt
