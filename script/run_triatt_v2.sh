#!/bin/bash
# After Ada chain: improved TriAttention calibration (v2: real text + freq_scale_sq) then
# re-run TriAtt (v2 stats + n_sink=4) on all 4 models @ budget 128. Calibrate+run on same server.
set -u
ROOT=/home/smp9898/A2SF; PY=$HOME/miniconda3/envs/A2SF/bin/python
RPY='export PATH=$HOME/miniconda3/envs/A2SF/bin:$PATH; cd ~/A2SF'
LOGD=$ROOT/logs/campaign; RES=$LOGD/triatt_v2_results.csv; DRV=$LOGD/driver.log
[ -f "$RES" ] || echo "run_name,overall" > "$RES"
declare -A REC=( [llama3-1b]=recipe_v3_1b [llama3-8b]=recipe_v3_8b [qwen2]=recipe_v3_qwen [mistral-7b]=recipe_v3_mistral )

do_model(){ local host="$1" m="$2"
  local stats="runs/triattention_stats/${m}_v2_stats.pt"
  local calib="CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python script/calibrate_triattention_v2.py --model $m --recipe datasets/training/raw/${REC[$m]}/train.jsonl --out $stats"
  local run="triattention_v2_${m}_128"
  local rcmd="HF_HUB_OFFLINE=1 python longbench.py --model $m --method triattention --triattention_stats $stats --n_sink 4 --budget 128 --gpus_per_model 1 --run_name $run"
  echo "[$(date +%H:%M:%S)] $host CALIB-v2 $m" >> $DRV
  if [ "$host" = local ]; then ( cd $ROOT && eval "${calib/python/$PY}" ) > $LOGD/calib_v2_${m}.log 2>&1
       ( cd $ROOT && rm -rf result_txt/pred/128/$run; eval "${rcmd/python/$PY}" ) > $LOGD/${run}.log 2>&1
  else ssh $host "$RPY; $calib" > $LOGD/calib_v2_${m}.log 2>&1
       ssh $host "rm -rf A2SF/result_txt/pred/128/$run" 2>/dev/null
       ssh $host "$RPY; $rcmd" > $LOGD/${run}.log 2>&1
       rsync -az $host:A2SF/result_txt/pred/128/$run/ $ROOT/result_txt/pred/128/$run/ >> $LOGD/${run}.log 2>&1; fi
  local ov; ov=$( cd $ROOT && $PY longbench_eval.py result_txt/pred/128/$run 2>>$LOGD/${run}.log | grep "Overall Average" | grep -oE '[0-9]+\.[0-9]+' | tail -1 )
  echo "${run},${ov:-FAIL}" >> $RES; echo "[$(date +%H:%M:%S)] $host DONE $run -> ${ov:-FAIL}" >> $DRV; }

until grep -q 'ADA+PYR64 ALL COMPLETE' "$DRV" 2>/dev/null; do sleep 180; done
( do_model local mistral-7b ) & ( do_model eslab18 llama3-8b ) & ( do_model eslab20 qwen2; do_model eslab20 llama3-1b ) &
wait
echo "[$(date +%H:%M:%S)] TRIATT-V2 ALL COMPLETE" >> $DRV
$PY $ROOT/script/move_to_backup.py >> $DRV 2>&1
