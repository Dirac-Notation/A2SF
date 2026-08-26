#!/bin/bash
# Baseline measurement campaign (v5).
#   models  x  {full, TOVA, SnapKV-16, SnapKV-32, H2O}  x  {128, 256, 512}
# Cross-server weighted sharding: eslab17(3090,w1) + eslab19(4090,w1.5).
# (eslab18 freed for another user, eslab20 busy — using 17+19 only.)
# Each cell: launch 3 shards -> wait all -> merge 18/19 into 17 -> score -> copy to backup.
# Resumable: a cell whose backup result.json already exists is skipped.
#
# Usage:
#   bash script/run_baseline_campaign.sh            # full matrix
#   bash script/run_baseline_campaign.sh test       # single cell (llama3-1b SnapKV-16 @128)
cd /home/smp9898/A2SF || exit 1
PY=$HOME/miniconda3/envs/A2SF/bin/python
RPY='$HOME/miniconda3/envs/A2SF/bin/python'
WEIGHTS="1,1.5"
GPUS="0,1,2,3,4,5,6,7"
LOG=logs/campaign.log
mkdir -p logs

declare -A GPM=( [llama3-1b]=1 [llama3-8b]=2 [qwen2]=2 [mistral-7b]=2 )

run_cell() {  # model method window budget run_name bdir
  local model=$1 method=$2 window=$3 budget=$4 run_name=$5 bdir=$6
  local gpm=${GPM[$model]}
  if [ -f "$bdir/result.json" ]; then echo "[$(date +%m-%d_%H:%M)] SKIP $run_name (already in backup)" | tee -a $LOG; return 0; fi
  local out="result_txt/pred/$budget/$run_name"
  echo "[$(date +%m-%d_%H:%M)] START $run_name (model=$model method=$method w=$window B=$budget gpm=$gpm)" | tee -a $LOG
  rm -rf "$out"; ssh eslab19 "rm -rf A2SF/$out" 2>/dev/null
  local A="--model $model --method $method --window $window --budget $budget --gpus_per_model $gpm --run_name $run_name --shard_count 2 --shard_weights $WEIGHTS"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPUS $PY longbench.py $A --shard_id 0 > logs/camp_s0.log 2>&1 & local P0=$!
  ssh eslab19 "cd A2SF && HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPUS $RPY longbench.py $A --shard_id 1" > logs/camp_s1.log 2>&1 & local P1=$!
  wait $P0; local r0=$?; wait $P1; local r1=$?
  if [ $r0 -ne 0 ] || [ $r1 -ne 0 ]; then
    echo "[$(date +%m-%d_%H:%M)] FAIL $run_name (rc=$r0/$r1) -- see logs/camp_s*.log" | tee -a $LOG; return 1; fi
  rm -rf /tmp/camp_s1
  rsync -az eslab19:A2SF/$out/ /tmp/camp_s1/ 2>/dev/null
  for f in /tmp/camp_s1/*.jsonl; do [ -e "$f" ] && cat "$f" >> "$out/$(basename "$f")"; done
  local nlines=$(cat "$out"/*.jsonl 2>/dev/null | wc -l)
  $PY longbench_eval.py "$out" > /dev/null 2>&1
  if [ ! -f "$out/result.json" ]; then echo "[$(date +%m-%d_%H:%M)] FAIL $run_name (no result.json after scoring)" | tee -a $LOG; return 1; fi
  mkdir -p "$bdir"; rsync -a "$out/" "$bdir/"
  local ov=$($PY -c "import json;print(json.load(open('$out/result.json'))['overall_average'])" 2>/dev/null)
  echo "[$(date +%m-%d_%H:%M)] DONE $run_name = $ov  ($nlines lines) -> $bdir" | tee -a $LOG
}

MODELS=(llama3-1b llama3-8b qwen2 mistral-7b)
BUDGETS=(128 256 512)
METHODS=( "TOVA:1" "SnapKV-16:16" "SnapKV-32:32" "H2O:32768" )

if [ "$1" = "test" ]; then
  echo "=== CAMPAIGN TEST CELL $(date) ===" | tee -a $LOG
  run_cell llama3-1b snap 16 128 "llama3-1b_SnapKV-16_128" "result_txt/backup/llama3-1b/128/llama3-1b_SnapKV-16_128"
  echo "=== TEST DONE ===" | tee -a $LOG
  exit 0
fi

echo "=== CAMPAIGN START $(date) ===" | tee -a $LOG
for model in "${MODELS[@]}"; do
  run_cell "$model" full 16 128 "${model}_full" "result_txt/backup/$model/${model}_full"
  for budget in "${BUDGETS[@]}"; do
    for mw in "${METHODS[@]}"; do
      pretty="${mw%%:*}"; window="${mw##*:}"
      run_cell "$model" snap "$window" "$budget" "${model}_${pretty}_${budget}" "result_txt/backup/$model/$budget/${model}_${pretty}_${budget}"
    done
  done
done
echo "=== CAMPAIGN COMPLETE $(date) ===" | tee -a $LOG
