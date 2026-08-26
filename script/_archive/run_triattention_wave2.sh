#!/bin/bash
# Wave 2: TriAttention for all 4 models, CONSISTENT directive-compliant recipe
# (synthetic calibration, no freq_scale_sq). Per-server gated: each server starts its
# wave-2 (calibrate -> run -> score) the moment its wave-1 list finishes (no idle wait).
set -u
ROOT=/home/smp9898/A2SF
PY=$HOME/miniconda3/envs/A2SF/bin/python
RPY='export PATH=$HOME/miniconda3/envs/A2SF/bin:$PATH; cd ~/A2SF'
BUD=128
LOGD=$ROOT/logs/campaign
RES=$LOGD/results.csv
DRV=$LOGD/driver.log

calib() {  # host model
  local host="$1" model="$2"
  local out="runs/triattention_stats/${model}_synth_stats.pt"
  local cmd="CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python script/calibrate_triattention.py --model $model --out $out"
  echo "[$(date +%H:%M:%S)] $host CALIB $model" >> $DRV
  if [ "$host" = "local" ]; then ( cd $ROOT && eval "${cmd/python/$PY}" ) > $LOGD/calib_${model}.log 2>&1
  else ssh $host "$RPY; $cmd" > $LOGD/calib_${model}.log 2>&1; fi
}

tri_run() {  # host model
  local host="$1" model="$2"
  local run="triattention_${model}_${BUD}"
  local log="$LOGD/${run}.log"
  local stats="runs/triattention_stats/${model}_synth_stats.pt"
  local cmd="longbench.py --model $model --budget $BUD --gpus_per_model 1 --run_name $run --method triattention --triattention_stats $stats"
  echo "[$(date +%H:%M:%S)] $host START $run" >> $DRV
  rm -rf "$ROOT/result_txt/pred/$BUD/$run"          # append-mode: clear before run
  [ "$host" != "local" ] && ssh $host "rm -rf A2SF/result_txt/pred/$BUD/$run" 2>/dev/null
  if [ "$host" = "local" ]; then ( cd $ROOT && HF_HUB_OFFLINE=1 $PY $cmd ) > "$log" 2>&1
  else ssh $host "$RPY; HF_HUB_OFFLINE=1 python $cmd" > "$log" 2>&1
       rsync -az $host:A2SF/result_txt/pred/$BUD/$run/ $ROOT/result_txt/pred/$BUD/$run/ >> "$log" 2>&1; fi
  local ov; ov=$( cd $ROOT && $PY longbench_eval.py result_txt/pred/$BUD/$run 2>>"$log" | grep "Overall Average" | grep -oE '[0-9]+\.[0-9]+' | tail -1 )
  echo "${run},${ov:-FAIL}" >> $RES
  echo "[$(date +%H:%M:%S)] $host DONE  $run -> ${ov:-FAIL}" >> $DRV
}

gate() { until grep -q "$1" "$DRV" 2>/dev/null; do sleep 120; done; }

# 17: after local wave-1 -> calib 1b+mistral, run mistral+1b TriAttention
( gate 'local RERUN COMPLETE'
  calib local llama3-1b; calib local mistral-7b
  tri_run local mistral-7b; tri_run local llama3-1b
  echo "[$(date +%H:%M:%S)] 17 WAVE2 COMPLETE" >> $DRV ) &
# 18: after eslab18 wave-1 -> calib+run 8b
( gate 'eslab18 LIST COMPLETE'
  calib eslab18 llama3-8b; tri_run eslab18 llama3-8b
  echo "[$(date +%H:%M:%S)] 18 WAVE2 COMPLETE" >> $DRV ) &
# 20: after eslab20 wave-1 -> calib+run qwen
( gate 'eslab20 LIST COMPLETE'
  calib eslab20 qwen2; tri_run eslab20 qwen2
  echo "[$(date +%H:%M:%S)] 20 WAVE2 COMPLETE" >> $DRV ) &
wait
echo "[$(date +%H:%M:%S)] WAVE2 ALL COMPLETE" >> $DRV
