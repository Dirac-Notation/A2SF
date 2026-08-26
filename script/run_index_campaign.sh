#!/bin/bash
# Fast-eval index campaign (v5): for each model, run all 13 sigmoid actions over
# LongBench (budget 128), store per-action predictions as the ALL/ dir, then build
# index.pt (per-sample per-action preds+scores+answers) via build_lb_index_from_all.
# Reuses longbench.py cross-server weighted sharding (eslab17 w1 + eslab19 w1.5).
# Resumable: an action whose ALL/<run> dir already has 16 jsonl is skipped.
#
#   bash script/run_index_campaign.sh         # full (4 models x 13 actions)
#   bash script/run_index_campaign.sh test    # single action (llama3-1b action12)
cd /home/smp9898/A2SF || exit 1
PY=$HOME/miniconda3/envs/A2SF/bin/python
RPY='$HOME/miniconda3/envs/A2SF/bin/python'
WEIGHTS="1,1.5"; GPUS="0,1,2,3,4,5,6,7"
LOG=logs/index_campaign.log; mkdir -p logs runs/fast_lb_eval
declare -A GPM=( [llama3-1b]=1 [llama3-8b]=2 [qwen2]=2 [mistral-7b]=2 )
B=128
# action_idx:a:b  (= RL.waits_model SIGMOID_A_VALUES/B_VALUES)
ACTIONS=( "00:0:1" "01:0.01:1" "02:0.01:16" "03:0.01:32" "04:0.01:128"
          "05:0.1:1" "06:0.1:16" "07:0.1:32" "08:0.1:128"
          "09:10:1" "10:10:16" "11:10:32" "12:10:128" )

run_action() {  # model aidx a b
  local model=$1 aidx=$2 a=$3 b=$4 gpm=${GPM[$1]}
  local run_name="${model}_action${aidx}_${B}"
  local alldir="result_txt/backup/${model}/${B}/ALL/${run_name}"
  if [ "$(ls "$alldir"/*.jsonl 2>/dev/null | wc -l)" -ge 16 ]; then
    echo "[$(date +%m-%d_%H:%M)] SKIP $run_name (ALL exists)" | tee -a $LOG; return 0; fi
  local out="result_txt/pred/${B}/${run_name}"
  echo "[$(date +%m-%d_%H:%M)] START $run_name (a=$a b=$b gpm=$gpm)" | tee -a $LOG
  rm -rf "$out"; ssh eslab19 "rm -rf A2SF/$out" 2>/dev/null
  local A="--model $model --method sigmoid --sigmoid_a $a --window $b --budget $B --gpus_per_model $gpm --run_name $run_name --shard_count 2 --shard_weights $WEIGHTS"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPUS $PY longbench.py $A --shard_id 0 > logs/idx_s0.log 2>&1 & local P0=$!
  ssh eslab19 "cd A2SF && HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPUS $RPY longbench.py $A --shard_id 1" > logs/idx_s1.log 2>&1 & local P1=$!
  wait $P0; local r0=$?; wait $P1; local r1=$?
  if [ $r0 -ne 0 ] || [ $r1 -ne 0 ]; then echo "[$(date +%m-%d_%H:%M)] FAIL $run_name (rc=$r0/$r1)" | tee -a $LOG; return 1; fi
  rm -rf /tmp/idx_s1; rsync -az eslab19:A2SF/$out/ /tmp/idx_s1/ 2>/dev/null
  for f in /tmp/idx_s1/*.jsonl; do [ -e "$f" ] && cat "$f" >> "$out/$(basename "$f")"; done
  local n=$(cat "$out"/*.jsonl 2>/dev/null | wc -l)
  mkdir -p "$alldir"; rsync -a "$out/" "$alldir/"
  echo "[$(date +%m-%d_%H:%M)] DONE $run_name ($n lines) -> $alldir" | tee -a $LOG
}

build_index() {  # model
  local model=$1
  local out="runs/fast_lb_eval/index_${model}_${B}.pt"
  echo "[$(date +%m-%d_%H:%M)] BUILD index $model -> $out" | tee -a $LOG
  $PY script/build_lb_index_from_all.py --all_dir "result_txt/backup/${model}/${B}/ALL" --out "$out" >> $LOG 2>&1 \
    && echo "[$(date +%m-%d_%H:%M)] INDEX OK $out" | tee -a $LOG \
    || echo "[$(date +%m-%d_%H:%M)] INDEX FAIL $model" | tee -a $LOG
}

if [ "$1" = "test" ]; then
  echo "=== INDEX TEST $(date) ===" | tee -a $LOG
  run_action llama3-1b 12 10 128
  exit 0
fi

echo "=== INDEX CAMPAIGN START $(date) ===" | tee -a $LOG
for model in llama3-1b llama3-8b qwen2 mistral-7b; do
  for act in "${ACTIONS[@]}"; do
    IFS=':' read aidx a b <<< "$act"
    run_action "$model" "$aidx" "$a" "$b"
  done
  build_index "$model"
done
echo "=== INDEX CAMPAIGN COMPLETE $(date) ===" | tee -a $LOG
