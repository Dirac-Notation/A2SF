#!/bin/bash
# Re-run ChunkKV / PyramidKV / Keyformer at WINDOW 16 (their base = SnapKV-16) for a
# fair comparison. Part A: qwen now on eslab20 free GPUs 2-7. Part B: mistral/8b/1b
# after NIAH frees the servers.
set -u
ROOT=/home/smp9898/A2SF
PY=$HOME/miniconda3/envs/A2SF/bin/python
RPY='export PATH=$HOME/miniconda3/envs/A2SF/bin:$PATH; cd ~/A2SF'
BUD=128
LOGD=$ROOT/logs/campaign
RES=$LOGD/window16_results.csv
DRV=$LOGD/driver.log
[ -f "$RES" ] || echo "run_name,overall" > "$RES"

args_for(){ case "$1" in
  chunkkv)   echo "--method snap --window 16 --chunk_size 10";;
  pyramidkv) echo "--method snap --window 16 --pyramid_kv";;
  keyformer) echo "--method keyformer --window 16";;
esac; }

run_job(){  # host model method gpustr
  local host="$1" model="$2" meth="$3" gpus="$4"
  local run="${meth}16_${model}_${BUD}"
  local log="$LOGD/${run}.log"
  local ea; ea="$(args_for "$meth")"
  local cmd="longbench.py --model $model --budget $BUD --gpus_per_model 1 --run_name $run $ea"
  echo "[$(date +%H:%M:%S)] $host START $run (GPUs $gpus)" >> $DRV
  rm -rf "$ROOT/result_txt/pred/$BUD/$run"
  if [ "$host" = "local" ]; then
    ( cd $ROOT && CUDA_VISIBLE_DEVICES=$gpus HF_HUB_OFFLINE=1 $PY $cmd ) > "$log" 2>&1
  else
    ssh $host "rm -rf A2SF/result_txt/pred/$BUD/$run" 2>/dev/null
    ssh $host "$RPY; CUDA_VISIBLE_DEVICES=$gpus HF_HUB_OFFLINE=1 python $cmd" > "$log" 2>&1
    rsync -az $host:A2SF/result_txt/pred/$BUD/$run/ $ROOT/result_txt/pred/$BUD/$run/ >> "$log" 2>&1
  fi
  local ov; ov=$( cd $ROOT && $PY longbench_eval.py result_txt/pred/$BUD/$run 2>>"$log" | grep "Overall Average" | grep -oE '[0-9]+\.[0-9]+' | tail -1 )
  echo "${run},${ov:-FAIL}" >> "$RES"
  echo "[$(date +%H:%M:%S)] $host DONE  $run -> ${ov:-FAIL}" >> $DRV
}

# ---- Part A: qwen NOW on eslab20 GPUs 2-7 (qwen full uses 0,1) ----
( for meth in chunkkv pyramidkv keyformer; do run_job eslab20 qwen2 $meth 2,3,4,5,6,7; done
  echo "[$(date +%H:%M:%S)] W16 qwen DONE" >> $DRV ) &

# ---- Part B: rest after NIAH frees servers ----
gate(){ until grep -q 'NIAH SWEEP ALL COMPLETE' "$DRV" 2>/dev/null; do sleep 180; done; }
( gate
  for meth in chunkkv pyramidkv keyformer; do run_job local mistral-7b $meth 0,1,2,3,4,5,6,7; done
  echo "[$(date +%H:%M:%S)] W16 mistral DONE" >> $DRV ) &
( gate
  for meth in chunkkv pyramidkv keyformer; do run_job eslab18 llama3-8b $meth 0,1,2,3,4,5,6,7; done
  echo "[$(date +%H:%M:%S)] W16 8b DONE" >> $DRV ) &
( gate
  until grep -q 'W16 qwen DONE' "$DRV" 2>/dev/null; do sleep 120; done   # wait Part A (qwen) off eslab20
  for meth in chunkkv pyramidkv keyformer; do run_job eslab20 llama3-1b $meth 0,1,2,3,4,5,6,7; done
  echo "[$(date +%H:%M:%S)] W16 1b DONE" >> $DRV ) &
wait
echo "[$(date +%H:%M:%S)] WINDOW16 RECOMPARE ALL COMPLETE" >> $DRV
