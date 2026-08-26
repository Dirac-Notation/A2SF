#!/bin/bash
# Auto-requeue watcher for the dispatch campaign. Every 120s scans NEW dispatch.log lines
# for "DONE <run> -> FAIL"; re-appends that job's spec (from jobspecs.txt) to queue.txt
# under flock, up to MAX_RETRY times per run. Also relaunches the dispatcher if it exited
# while jobs remain queued (handles the worker-exit race after a late requeue). Exits when
# dispatch is done and the queue is drained.
set -u
ROOT=/home/smp9898/A2SF; LOGD=$ROOT/logs/campaign
DRV=$LOGD/dispatch.log; QUEUE=$LOGD/queue.txt; LOCK=$LOGD/queue.lock
SPECS=$LOGD/jobspecs.txt; OFF=$LOGD/requeue.offset; WLOG=$LOGD/requeue.log
MAX_RETRY=2

while :; do
  total=$(wc -l < "$DRV" 2>/dev/null || echo 0)
  off=$(cat "$OFF" 2>/dev/null || echo 0)
  if [ "$total" -gt "$off" ]; then
    # scan only new lines for failures
    tail -n +$((off+1)) "$DRV" | grep -E 'DONE .* -> FAIL$' | sed -E 's/.* DONE (\S+) -> FAIL/\1/' | while read -r run; do
      [ -z "$run" ] && continue
      cnt=$(cat "$LOGD/retry_${run}.cnt" 2>/dev/null || echo 0)
      if [ "$cnt" -lt "$MAX_RETRY" ]; then
        spec=$(grep -F "|${run}|" "$SPECS" | head -1)
        if [ -n "$spec" ]; then
          exec 9>"$LOCK"; flock 9; echo "$spec" >> "$QUEUE"; flock -u 9
          echo $((cnt+1)) > "$LOGD/retry_${run}.cnt"
          echo "[$(date +%H:%M:%S)] REQUEUE $run (attempt $((cnt+1))/$MAX_RETRY)" >> "$WLOG"
        else
          echo "[$(date +%H:%M:%S)] NO SPEC for $run, cannot requeue" >> "$WLOG"
        fi
      else
        echo "[$(date +%H:%M:%S)] GIVE UP $run (>= $MAX_RETRY retries)" >> "$WLOG"
      fi
    done
    echo "$total" > "$OFF"
  fi

  # relaunch dispatcher if it died with jobs still queued (late-requeue race)
  qn=$(wc -l < "$QUEUE" 2>/dev/null || echo 0)
  if ! pgrep -f '[r]un_dispatch.sh' >/dev/null 2>&1 && [ "$qn" -gt 0 ]; then
    echo "[$(date +%H:%M:%S)] dispatcher down with $qn queued -> relaunch" >> "$WLOG"
    nohup bash "$ROOT/script/run_dispatch.sh" >> "$LOGD/dispatch_driver.log" 2>&1 & disown
    sleep 10
  fi

  # exit condition: dispatch fully done and queue empty
  if grep -q 'DISPATCH ALL DONE' "$DRV" 2>/dev/null && [ "$qn" -eq 0 ] && ! pgrep -f '[r]un_dispatch.sh' >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] all done, watcher exit" >> "$WLOG"; break
  fi
  sleep 120
done
