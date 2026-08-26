#!/bin/bash
# After KVZip + WAITS-combo: (1) Ada-KV (head-wise adaptive budget, pad-to-max sim) on
# TOVA/H2O/SnapKV/WAITS @ budget 128; (2) PyramidKV @ budget 64 (low-budget check).
# Then move all completed pred dirs to backup. Gated on 'WAITS COMBO RERUN COMPLETE'.
set -u
ROOT=/home/smp9898/A2SF; PY=$HOME/miniconda3/envs/A2SF/bin/python
RPY='export PATH=$HOME/miniconda3/envs/A2SF/bin:$PATH; cd ~/A2SF'
LOGD=$ROOT/logs/campaign; RES=$LOGD/ada_results.csv; DRV=$LOGD/driver.log
[ -f "$RES" ] || echo "run_name,overall" > "$RES"

# variant -> (budget, args)  ; Ada wraps each scorer with --ada_kv
args_for(){ local m=$2
  case "$1" in
    AdaTOVA)   echo "128|--method snap --window 1 --ada_kv";;
    AdaH2O)    echo "128|--method snap --window 32768 --ada_kv";;
    AdaSnapKV) echo "128|--method snap --window 16 --ada_kv";;
    AdaWAITS)  echo "128|--waits_table runs/waits_tables/waits_${m}.json --ada_kv";;
    PyramidKV64) echo "64|--method snap --window 16 --pyramid_kv";;
  esac; }

run_job(){ local host="$1" model="$2" var="$3"
  local spec; spec="$(args_for "$var" "$model")"; local bud="${spec%%|*}" ea="${spec#*|}"
  local run="${var}_${model}_${bud}"; local log="$LOGD/${run}.log"
  local cmd="longbench.py --model $model --budget $bud --gpus_per_model 1 --run_name $run $ea"
  echo "[$(date +%H:%M:%S)] $host START $run" >> $DRV; rm -rf "$ROOT/result_txt/pred/$bud/$run"
  if [ "$host" = "local" ]; then ( cd $ROOT && HF_HUB_OFFLINE=1 $PY $cmd ) > "$log" 2>&1
  else ssh $host "rm -rf A2SF/result_txt/pred/$bud/$run" 2>/dev/null; ssh $host "$RPY; HF_HUB_OFFLINE=1 python $cmd" > "$log" 2>&1
       rsync -az $host:A2SF/result_txt/pred/$bud/$run/ $ROOT/result_txt/pred/$bud/$run/ >> "$log" 2>&1; fi
  local ov; ov=$( cd $ROOT && $PY longbench_eval.py result_txt/pred/$bud/$run 2>>"$log" | grep "Overall Average" | grep -oE '[0-9]+\.[0-9]+' | tail -1 )
  echo "${run},${ov:-FAIL}" >> "$RES"; echo "[$(date +%H:%M:%S)] $host DONE $run -> ${ov:-FAIL}" >> $DRV; }

until grep -q 'WAITS COMBO RERUN COMPLETE' "$DRV" 2>/dev/null; do sleep 180; done

VARS="AdaTOVA AdaH2O AdaSnapKV AdaWAITS PyramidKV64"
( for v in $VARS; do run_job local mistral-7b $v; done; echo "[$(date +%H:%M:%S)] ADA mistral DONE" >> $DRV ) &
( for v in $VARS; do run_job eslab18 llama3-8b $v; done; echo "[$(date +%H:%M:%S)] ADA 8b DONE" >> $DRV ) &
( for v in $VARS; do run_job eslab20 qwen2 $v; done
  for v in $VARS; do run_job eslab20 llama3-1b $v; done
  echo "[$(date +%H:%M:%S)] ADA qwen+1b DONE" >> $DRV ) &
wait
echo "[$(date +%H:%M:%S)] ADA+PYR64 ALL COMPLETE" >> $DRV

# move completed pred dirs (have result.json) -> backup/<model>/<budget>/
$PY $ROOT/script/move_to_backup.py >> $DRV 2>&1
echo "[$(date +%H:%M:%S)] BACKUP MOVE DONE" >> $DRV
