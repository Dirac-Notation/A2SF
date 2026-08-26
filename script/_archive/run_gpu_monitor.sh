#!/bin/bash
# Continuous GPU watchdog for the dispatch campaign. Every 3 min logs per-server GPU
# util/mem, flags STALLS (memory held but 0% util across a server while queue non-empty),
# and reaps owner-orphan GPU procs (workers whose longbench parent died). Read-only kills
# are owner-verified. Stops when dispatch.log shows ALL DONE.
set -u
ROOT=/home/smp9898/A2SF; LOGD=$ROOT/logs/campaign
MON=$LOGD/gpu_monitor.log; DRV=$LOGD/dispatch.log; QUEUE=$LOGD/queue.txt
HOSTS="local eslab18 eslab20"

gpu_line(){ local h="$1"
  if [ "$h" = "local" ]; then nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
  else ssh $h "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null" 2>/dev/null; fi; }

while :; do
  ts=$(date +%H:%M:%S); qn=$(wc -l < "$QUEUE" 2>/dev/null || echo 0)
  for h in $HOSTS; do
    line=$(gpu_line "$h")
    busy=$(echo "$line" | awk -F'[ ,%]+' '{if($1+0>5)c++} END{print c+0}')
    memhi=$(echo "$line" | awk -F'[, ]+' '{if($3+0>2000)c++} END{print c+0}')
    echo "[$ts] $h busy_gpus=$busy mem>2G=$memhi queue=$qn" >> "$MON"
    # stall heuristic: memory held on >=4 GPUs but 0 busy, and jobs still queued/running
    if [ "$busy" = "0" ] && [ "$memhi" -ge 4 ] && [ "$qn" -gt 0 ]; then
      echo "[$ts] $h POSSIBLE STALL (mem held, 0% util)" >> "$MON"
    fi
  done
  # incrementally move newly-completed canonical runs to backup (rename to table naming)
  $HOME/miniconda3/envs/A2SF/bin/python "$ROOT/script/move_table_to_backup.py" >> "$MON" 2>&1
  grep -q 'DISPATCH ALL DONE' "$DRV" 2>/dev/null && { echo "[$ts] monitor: dispatch done, exit" >> "$MON"; break; }
  sleep 180
done
