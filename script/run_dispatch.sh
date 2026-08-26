#!/bin/bash
# Work-stealing dispatcher: 3 servers (17/18/20) each pull jobs from a shared flock'd
# queue (logs/campaign/queue.txt) until empty -> no idle GPU while jobs remain.
# Single instance only. OOM-hardened (expandable_segments). One job per server (uses all GPUs).
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # local jobs inherit this
ROOT=/home/smp9898/A2SF; PY=$HOME/miniconda3/envs/A2SF/bin/python
RPY="export PATH=\$HOME/miniconda3/envs/A2SF/bin:\$PATH; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd ~/A2SF"
LOGD=$ROOT/logs/campaign; QUEUE=$LOGD/queue.txt; LOCK=$LOGD/queue.lock
RES=$LOGD/table_results.csv; DRV=$LOGD/dispatch.log
[ -f "$RES" ] || echo "run_name,overall" > "$RES"

pop_job(){ exec 9>"$LOCK"; flock 9
  local job; job=$(head -1 "$QUEUE" 2>/dev/null)
  [ -n "$job" ] && sed -i '1d' "$QUEUE"
  flock -u 9; echo "$job"; }

run_one(){ local host="$1" job="$2"
  local model budget run flags; IFS='|' read -r model budget run flags <<< "$job"
  local log="$LOGD/${run}.log"
  local cmd="longbench.py --model $model --budget $budget --gpus_per_model 1 --run_name $run $flags"
  echo "[$(date +%H:%M:%S)] $host START $run" >> $DRV
  if [ "$host" = "local" ]; then
    rm -rf "$ROOT/result_txt/pred/$budget/$run"
    ( cd $ROOT && HF_HUB_OFFLINE=1 $PY $cmd ) > "$log" 2>&1
  else
    ssh $host "rm -rf A2SF/result_txt/pred/$budget/$run" 2>/dev/null
    ssh $host "$RPY; HF_HUB_OFFLINE=1 python $cmd" > "$log" 2>&1
    rsync -az $host:A2SF/result_txt/pred/$budget/$run/ $ROOT/result_txt/pred/$budget/$run/ >> "$log" 2>&1
  fi
  local ov; ov=$( cd $ROOT && $PY longbench_eval.py result_txt/pred/$budget/$run 2>>"$log" | grep "Overall Average" | grep -oE '[0-9]+\.[0-9]+' | tail -1 )
  echo "${run},${ov:-FAIL}" >> "$RES"
  echo "[$(date +%H:%M:%S)] $host DONE $run -> ${ov:-FAIL}" >> $DRV; }

worker(){ local host="$1"; local job
  while :; do job=$(pop_job); [ -z "$job" ] && break; run_one "$host" "$job"; done
  echo "[$(date +%H:%M:%S)] $host QUEUE EMPTY" >> $DRV; }

echo "[$(date +%H:%M:%S)] DISPATCH START ($(wc -l < $QUEUE) jobs)" >> $DRV
worker local & worker eslab18 & worker eslab20 &
wait
echo "[$(date +%H:%M:%S)] DISPATCH ALL DONE" >> $DRV
$PY $ROOT/script/move_table_to_backup.py >> $DRV 2>&1
echo "[$(date +%H:%M:%S)] BACKUP MOVE DONE" >> $DRV
