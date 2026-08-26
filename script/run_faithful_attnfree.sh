#!/bin/bash
# FAITHFUL re-run of attention-free scorers in their ORIGINAL form: pure top-k by the
# method's own score, NO forced recent window, NO sink (recent_budget=0, n_sink=0).
#   - KeyDiff vanilla: S = topk(-CosSim(K), N)  [paper: SW is a SEPARATE variant]
#   - L2-norm (Devoto): keep lowest-L2 keys, pure top-k
# The earlier recent_budget=16 / n_sink=4 numbers ALTERED these methods (= KeyDiff+SlidingWindow
# / +sink) and are NOT canonical. Gated on KVZip completion to avoid stealing its GPUs.
set -u
ROOT=/home/smp9898/A2SF; PY=$HOME/miniconda3/envs/A2SF/bin/python
RPY='export PATH=$HOME/miniconda3/envs/A2SF/bin:$PATH; cd ~/A2SF'
LOGD=$ROOT/logs/campaign; RES=$LOGD/faithful_results.csv; DRV=$LOGD/driver.log
[ -f "$RES" ] || echo "run_name,overall,note" > "$RES"

run_job(){ local host="$1" model="$2" method="$3"
  local run="${method}_vanilla_${model}_128"; local log="$LOGD/${run}.log"
  local cmd="longbench.py --model $model --budget 128 --gpus_per_model 1 --run_name $run \
    --method $method --recent_budget 0 --n_sink 0"
  echo "[$(date +%H:%M:%S)] $host START $run" >> $DRV; rm -rf "$ROOT/result_txt/pred/128/$run"
  if [ "$host" = "local" ]; then ( cd $ROOT && HF_HUB_OFFLINE=1 $PY $cmd ) > "$log" 2>&1
  else ssh $host "rm -rf A2SF/result_txt/pred/128/$run" 2>/dev/null
       ssh $host "$RPY; HF_HUB_OFFLINE=1 python $cmd" > "$log" 2>&1
       rsync -az $host:A2SF/result_txt/pred/128/$run/ $ROOT/result_txt/pred/128/$run/ >> "$log" 2>&1; fi
  local ov; ov=$( cd $ROOT && $PY longbench_eval.py result_txt/pred/128/$run 2>>"$log" | grep "Overall Average" | grep -oE '[0-9]+\.[0-9]+' | tail -1 )
  echo "${run},${ov:-FAIL},recent0_sink0_vanilla" >> "$RES"
  echo "[$(date +%H:%M:%S)] $host DONE $run -> ${ov:-FAIL}" >> $DRV; }

# event-driven gate: wait until all 3 KVZip result.json exist (don't steal GPUs)
echo "[$(date +%H:%M:%S)] faithful: waiting for KVZip completion" >> $DRV
until [ -f "$ROOT/result_txt/pred/128/kvzip_qwen2_128/result.json" ] \
   && { [ -f "$ROOT/result_txt/pred/128/kvzip_llama3-8b_128/result.json" ] || ssh eslab18 "test -f A2SF/result_txt/pred/128/kvzip_llama3-8b_128/result.json"; } \
   && [ -f "$ROOT/result_txt/pred/128/kvzip_mistral-7b_128/result.json" ]; do sleep 180; done 2>/dev/null
echo "[$(date +%H:%M:%S)] faithful: KVZip done, launching vanilla attn-free" >> $DRV

METHODS="keydiff l2norm"
( for m in $METHODS; do run_job local      mistral-7b $m; done; echo "[$(date +%H:%M:%S)] FAITHFUL mistral DONE" >> $DRV ) &
( for m in $METHODS; do run_job eslab18    llama3-8b  $m; done; echo "[$(date +%H:%M:%S)] FAITHFUL 8b DONE" >> $DRV ) &
( for m in $METHODS; do run_job eslab20    qwen2      $m; done
  for m in $METHODS; do run_job eslab20    llama3-1b  $m; done
  echo "[$(date +%H:%M:%S)] FAITHFUL qwen+1b DONE" >> $DRV ) &
wait
echo "[$(date +%H:%M:%S)] FAITHFUL ATTN-FREE ALL COMPLETE" >> $DRV
