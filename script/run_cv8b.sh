#!/usr/bin/env bash
cd /home/smp9898/A2SF
PY=$HOME/miniconda3/envs/A2SF/bin/python
export CUDA_VISIBLE_DEVICES=2
$PY RL/train.py --model llama3-8b --budget 128 \
  --states_file runs/states/cv_llama3-8b_train.pt --data_file datasets/cv/cv_llama3-8b_train.jsonl \
  --val_data_file datasets/cv/cv_llama3-8b_val.jsonl \
  --score_field action_scores_gt_by_budget --val_score_field action_scores_gt_by_budget \
  --mini_attn_ckpt runs/mini_attn_v5_8b/mini_attn_best.pt --extra_view none --loss listwise --loss_temp 0.1 \
  --epochs 200 --ucb_topk 4 --ucb_beta 1.0 --seed 42 --save_dir runs/cv8b_gtlw
$PY script/fast_lb_eval.py --rl_checkpoint runs/cv8b_gtlw/policy_best.pt \
  --run_name cv8b_gtlw --budget 128 --states_path runs/fast_lb_eval/cv_llama3-8b_test_states.pt \
  --index_path runs/fast_lb_eval/cv_llama3-8b_test_index.pt
echo "CV8B_DONE"
