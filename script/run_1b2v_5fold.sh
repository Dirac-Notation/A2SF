#!/usr/bin/env bash
cd /home/smp9898/A2SF; PY=$HOME/miniconda3/envs/A2SF/bin/python
one(){ local fold=$1 gpu=$2; local pf=v2f${fold}
  $PY script/cv_unified.py --model llama3-1b --lb_states runs/fast_lb_eval/lb_states_1b_2v_none.pt --out_prefix $pf --fold $fold --nfolds 5 >/dev/null 2>&1
  $PY -c "import torch;ix=torch.load('runs/fast_lb_eval/${pf}_test_index.pt');ix['datasets']=sorted(set(k.split('/')[0] for k in ix if '/' in k));torch.save(ix,'runs/fast_lb_eval/${pf}_test_index.pt')"
  CUDA_VISIBLE_DEVICES=$gpu $PY RL/train.py --model llama3-1b --budget 128 --states_file runs/states/${pf}_train.pt \
    --data_file datasets/cv/${pf}_train.jsonl --val_data_file datasets/cv/${pf}_val.jsonl \
    --score_field action_scores_gt_by_budget --val_score_field action_scores_gt_by_budget \
    --mini_attn_ckpt runs/mini_attn_v5/mini_attn_best.pt --extra_view none --num_views 2 --loss listwise --loss_temp 0.1 \
    --epochs 200 --ucb_topk 4 --ucb_beta 1.0 --seed 42 --save_dir runs/${pf}_r >/dev/null 2>&1
  CUDA_VISIBLE_DEVICES=$gpu $PY script/fast_lb_eval.py --rl_checkpoint runs/${pf}_r/policy_best.pt --run_name $pf --budget 128 \
    --states_path runs/fast_lb_eval/${pf}_test_states.pt --index_path runs/fast_lb_eval/${pf}_test_index.pt 2>/dev/null | grep "Overall Average"|sed "s/^/2v fold${fold}: /"
  rm -rf runs/${pf}_r
}
one 0 0 & one 1 1 & one 2 2 & wait
one 3 0 & one 4 1 & wait
echo V2_5FOLD_DONE
