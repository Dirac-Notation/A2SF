#!/bin/bash
# Investigate WAITS combined with ChunkKV / PyramidKV (orthogonal selector/budget).
# WAITS-live (sigmoid metric-routing via waits_table) vs +Chunk vs +Pyr, budget 128, 4 models.
# (WAITS+Keyformer is architecturally impossible: both are scorers.)
# Gated after n_sink rerun to avoid GPU contention.
set -u
ROOT=/home/smp9898/A2SF
PY=$HOME/miniconda3/envs/A2SF/bin/python
RPY='export PATH=$HOME/miniconda3/envs/A2SF/bin:$PATH; cd ~/A2SF'
BUD=128
LOGD=$ROOT/logs/campaign
RES=$LOGD/waits_combo_results.csv
DRV=$LOGD/driver.log
[ -f "$RES" ] || echo "run_name,overall" > "$RES"

args_for(){ # variant model
  local wt="runs/waits_tables/waits_$2.json"
  case "$1" in
    WAITS)       echo "--waits_table $wt";;
    WAITS-Chunk) echo "--waits_table $wt --chunk_size 10";;
    WAITS-Pyr)   echo "--waits_table $wt --pyramid_kv";;
  esac; }

run_job(){ # host model variant
  local host="$1" model="$2" var="$3"
  local run="${var}_${model}_${BUD}"
  local log="$LOGD/${run}.log"; local ea; ea="$(args_for "$var" "$model")"
  local cmd="longbench.py --model $model --budget $BUD --gpus_per_model 1 --run_name $run $ea"
  echo "[$(date +%H:%M:%S)] $host START $run" >> $DRV
  rm -rf "$ROOT/result_txt/pred/$BUD/$run"
  if [ "$host" = "local" ]; then ( cd $ROOT && HF_HUB_OFFLINE=1 $PY $cmd ) > "$log" 2>&1
  else ssh $host "rm -rf A2SF/result_txt/pred/$BUD/$run" 2>/dev/null
       ssh $host "$RPY; HF_HUB_OFFLINE=1 python $cmd" > "$log" 2>&1
       rsync -az $host:A2SF/result_txt/pred/$BUD/$run/ $ROOT/result_txt/pred/$BUD/$run/ >> "$log" 2>&1; fi
  local ov; ov=$( cd $ROOT && $PY longbench_eval.py result_txt/pred/$BUD/$run 2>>"$log" | grep "Overall Average" | grep -oE '[0-9]+\.[0-9]+' | tail -1 )
  echo "${run},${ov:-FAIL}" >> "$RES"
  echo "[$(date +%H:%M:%S)] $host DONE  $run -> ${ov:-FAIL}" >> $DRV
}

gate(){ until grep -q 'KVZIP ALL COMPLETE' "$DRV" 2>/dev/null; do sleep 180; done; }
gate

( for v in WAITS WAITS-Chunk WAITS-Pyr; do run_job local mistral-7b $v; done; echo "[$(date +%H:%M:%S)] COMBO mistral DONE" >> $DRV ) &
( for v in WAITS WAITS-Chunk WAITS-Pyr; do run_job eslab18 llama3-8b $v; done; echo "[$(date +%H:%M:%S)] COMBO 8b DONE" >> $DRV ) &
( for v in WAITS WAITS-Chunk WAITS-Pyr; do run_job eslab20 qwen2 $v; done
  for v in WAITS WAITS-Chunk WAITS-Pyr; do run_job eslab20 llama3-1b $v; done
  echo "[$(date +%H:%M:%S)] COMBO qwen+1b DONE" >> $DRV ) &
wait
echo "[$(date +%H:%M:%S)] WAITS COMBO RERUN COMPLETE" >> $DRV
