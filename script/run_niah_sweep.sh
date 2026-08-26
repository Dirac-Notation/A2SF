#!/bin/bash
# NIAH sweep: full + 13 WAITS actions @ budget 128 for 8B(18)/qwen(20)/mistral(17).
# Per-server gated on that server's wave-2 completion, then 8-GPU sweep. Merges results.
set -u
ROOT=/home/smp9898/A2SF
RPY='export PATH=$HOME/miniconda3/envs/A2SF/bin:$PATH; cd ~/A2SF'
DRV=$ROOT/logs/campaign/driver.log
BUD=128
cd $ROOT

# sync eval code + worker + dataset to remote servers
for h in eslab18 eslab20; do
  rsync -az evaluate_needle.py $h:A2SF/ 2>/dev/null
  rsync -az script/niah_server_worker.sh $h:A2SF/script/ 2>/dev/null
  rsync -az datasets/needle/needle_eval.jsonl $h:A2SF/datasets/needle/ 2>/dev/null
done

gate(){ until grep -q "$1" "$DRV" 2>/dev/null; do sleep 120; done; }

# 17 (local) = mistral
( gate '17 WAVE2 COMPLETE'
  echo "[$(date +%H:%M:%S)] 17 NIAH START mistral-7b" >> $DRV
  bash script/niah_server_worker.sh mistral-7b $BUD >> $DRV 2>&1
  cp result_txt/needle/sweep_results.csv /tmp/niah_17.csv
  echo "[$(date +%H:%M:%S)] 17 NIAH COMPLETE" >> $DRV ) &
# 18 = 8b
( gate '18 WAVE2 COMPLETE'
  echo "[$(date +%H:%M:%S)] 18 NIAH START llama3-8b" >> $DRV
  ssh eslab18 "$RPY; bash script/niah_server_worker.sh llama3-8b $BUD" >> $DRV 2>&1
  rsync -az eslab18:A2SF/result_txt/needle/sweep_results.csv /tmp/niah_18.csv 2>/dev/null
  echo "[$(date +%H:%M:%S)] 18 NIAH COMPLETE" >> $DRV ) &
# 20 = qwen
( gate '20 WAVE2 COMPLETE'
  echo "[$(date +%H:%M:%S)] 20 NIAH START qwen2" >> $DRV
  ssh eslab20 "$RPY; bash script/niah_server_worker.sh qwen2 $BUD" >> $DRV 2>&1
  rsync -az eslab20:A2SF/result_txt/needle/sweep_results.csv /tmp/niah_20.csv 2>/dev/null
  echo "[$(date +%H:%M:%S)] 20 NIAH COMPLETE" >> $DRV ) &
wait

# merge into one file
echo "model,method,action,budget,accuracy,correct_total" > result_txt/needle/niah_sweep_b${BUD}.csv
cat /tmp/niah_17.csv /tmp/niah_18.csv /tmp/niah_20.csv >> result_txt/needle/niah_sweep_b${BUD}.csv 2>/dev/null
echo "[$(date +%H:%M:%S)] NIAH SWEEP ALL COMPLETE -> result_txt/needle/niah_sweep_b${BUD}.csv" >> $DRV
