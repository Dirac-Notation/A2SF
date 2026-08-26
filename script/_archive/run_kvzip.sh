#!/bin/bash
# KVZip (faithful reconstruction-based) on LongBench, budget 128, 4 models.
# Gated after WAITS combo. KVZip is attention-BASED, query-agnostic, ~2x prefill + manual decode (slow).
set -u
ROOT=/home/smp9898/A2SF; PY=$HOME/miniconda3/envs/A2SF/bin/python
RPY='export PATH=$HOME/miniconda3/envs/A2SF/bin:$PATH; cd ~/A2SF'
BUD=128; LOGD=$ROOT/logs/campaign; RES=$LOGD/kvzip_results.csv; DRV=$LOGD/driver.log
[ -f "$RES" ] || echo "run_name,overall" > "$RES"
run_job(){ local host="$1" model="$2"; local run="kvzip_${model}_${BUD}"; local log="$LOGD/${run}.log"
  local cmd="longbench.py --model $model --method kvzip --budget $BUD --gpus_per_model 2 --n_sink 4 --run_name $run"
  echo "[$(date +%H:%M:%S)] $host START $run" >> $DRV; rm -rf "$ROOT/result_txt/pred/$BUD/$run"
  if [ "$host" = "local" ]; then ( cd $ROOT && HF_HUB_OFFLINE=1 $PY $cmd ) > "$log" 2>&1
  else ssh $host "rm -rf A2SF/result_txt/pred/$BUD/$run" 2>/dev/null; ssh $host "$RPY; HF_HUB_OFFLINE=1 python $cmd" > "$log" 2>&1
       rsync -az $host:A2SF/result_txt/pred/$BUD/$run/ $ROOT/result_txt/pred/$BUD/$run/ >> "$log" 2>&1; fi
  local ov; ov=$( cd $ROOT && $PY longbench_eval.py result_txt/pred/$BUD/$run 2>>"$log" | grep "Overall Average" | grep -oE '[0-9]+\.[0-9]+' | tail -1 )
  echo "${run},${ov:-FAIL}" >> "$RES"; echo "[$(date +%H:%M:%S)] $host DONE $run -> ${ov:-FAIL}" >> $DRV; }
until grep -q 'WAITS COMBO ALL COMPLETE' "$DRV" 2>/dev/null; do sleep 180; done
( run_job local mistral-7b ) & ( run_job eslab18 llama3-8b ) & ( run_job eslab20 qwen2; run_job eslab20 llama3-1b ) &
wait; echo "[$(date +%H:%M:%S)] KVZIP ALL COMPLETE" >> $DRV
