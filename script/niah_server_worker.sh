#!/bin/bash
# Runs ON a server: NIAH full + all 13 WAITS sigmoid actions at one budget,
# spread across 8 GPUs (one run per GPU), in waves. Appends to result_txt/needle/sweep_results.csv.
# args: <model> <budget>
set -u
model="$1"; bud="${2:-128}"
PY=$HOME/miniconda3/envs/A2SF/bin/python
DS=datasets/needle/needle_eval.jsonl
cd ~/A2SF || exit 1
LOGD=logs/campaign; mkdir -p "$LOGD" result_txt/needle
: > result_txt/needle/sweep_results.csv   # fresh (rows: model,method,ab,budget,acc,correct/total)

# 13 actions as "a:b"
AB=("0:1" "0.01:1" "0.01:16" "0.01:32" "0.01:128" "0.1:1" "0.1:16" "0.1:32" "0.1:128" "10:1" "10:16" "10:32" "10:128")

run_action() {  # gpu a b
  local g="$1" a="$2" b="$3"
  CUDA_VISIBLE_DEVICES="$g" HF_HUB_OFFLINE=1 $PY evaluate_needle.py --model "$model" \
    --method sigmoid --budget "$bud" --window 32 --sigmoid_a "$a" --sigmoid_b "$b" \
    --dataset "$DS" > "$LOGD/niah_${model}_a${a}_b${b}.log" 2>&1
}

# full needs 2 GPUs (30k-token contexts OOM on 1 GPU); run it on GPU 0,1 in background
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 $PY evaluate_needle.py --model "$model" \
  --method full --budget "$bud" --dataset "$DS" > "$LOGD/niah_${model}_full.log" 2>&1 &
fullpid=$!

# 13 actions across the remaining 6 GPUs (2..7) in waves
GPUS=(2 3 4 5 6 7)
i=0
while [ $i -lt ${#AB[@]} ]; do
  pids=()
  for g in "${GPUS[@]}"; do
    [ $i -ge ${#AB[@]} ] && break
    ab="${AB[$i]}"
    run_action "$g" "${ab%%:*}" "${ab##*:}" & pids+=($!)
    i=$((i+1))
  done
  wait "${pids[@]}"
done
wait "$fullpid"
echo "[$(date +%H:%M:%S)] NIAH $model b$bud DONE"
