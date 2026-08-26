#!/usr/bin/env bash
# Smoke test for the NeuralUCB (per-prompt, submitted-method) track.
# Proves the version-locked workbench is runnable end-to-end in minutes:
#   [1] load the champion-repro checkpoint and fast-eval it on the LB index
#   [2] 2-epoch training run of the submitted trainer (same flags as repro_2694.sh)
# Run after any cleanup/refactor that could touch this track. Full repro stays
# script/repro_2694.sh (200 epochs, seed 42).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-runs/_smoke_neuralucb}
rm -rf "$OUT"; mkdir -p "$OUT"

echo "[1/2] checkpoint load + fast LB eval (runs/repro_2694_seed42)"
python script/fast_lb_eval.py \
  --rl_checkpoint runs/repro_2694_seed42/policy_best.pt \
  --run_name _smoke_neuralucb \
  --budget 128 \
  --states_path runs/fast_lb_eval/lb_states_myv_none.pt \
  --index_path  runs/fast_lb_eval/index_llama3-1b_128.pt

echo "[2/2] 2-epoch training smoke (submitted trainer)"
python RL/train_perprompt_submitted.py --model llama3-1b --budget 128 \
  --states_file     runs/states/old_none.pt \
  --data_file       datasets/training/scored/llama3-1b/train.jsonl \
  --val_data_file   datasets/training/scored/llama3-1b/validation.jsonl \
  --score_field     action_scores_gt_by_budget \
  --val_score_field action_scores_gt_by_budget \
  --mini_attn_ckpt  runs/mini_attn_v5/mini_attn_best.pt \
  --extra_view none --loss listwise --loss_temp 0.1 \
  --epochs 2 --ucb_topk 4 --ucb_beta 1.0 --seed 0 \
  --save_dir "$OUT"

echo "SMOKE OK (eval + 2-epoch train both ran)"
