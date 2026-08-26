#!/bin/bash
# Technique campaign: 7 techniques x 4 models @ budget 128.
# TriAttention only on 1B for now (8B/qwen/mistral need calibration stats -> added later).
# Static per-server assignment; each server runs its list sequentially with 8-way DP.
set -u
ROOT=/home/smp9898/A2SF
PY=$HOME/miniconda3/envs/A2SF/bin/python
RPY='export PATH=$HOME/miniconda3/envs/A2SF/bin:$PATH; cd ~/A2SF'
BUD=128
LOGD=$ROOT/logs/campaign
RES=$LOGD/results.csv
mkdir -p $LOGD
[ -f $RES ] || echo "run_name,overall" > $RES

args_for() {
  case "$1" in
    keyformer)    echo "--method keyformer --window 32" ;;
    pyramidkv)    echo "--method snap --window 32 --pyramid_kv" ;;
    chunkkv)      echo "--method snap --window 32 --chunk_size 10" ;;
    streamingllm) echo "--method streamingllm --window 32" ;;
    keydiff)      echo "--method keydiff --window 32" ;;
    l2norm)       echo "--method l2norm --window 32" ;;
    triattention) echo "--method triattention --triattention_stats runs/triattention_stats/MODEL_stats.pt" ;;
  esac
}

run_job() {
  local host="$1" model="$2" tech="$3"
  local run="${tech}_${model}_${BUD}"
  local log="$LOGD/${run}.log"
  local ea; ea="$(args_for $tech)"; ea="${ea/MODEL/$model}"
  local cmd="longbench.py --model $model --budget $BUD --gpus_per_model 1 --run_name $run $ea"
  echo "[$(date +%H:%M:%S)] $host START $run :: $cmd" >> $LOGD/driver.log
  rm -rf "$ROOT/result_txt/pred/$BUD/$run"          # append-mode: clear before run
  [ "$host" != "local" ] && ssh $host "rm -rf A2SF/result_txt/pred/$BUD/$run" 2>/dev/null
  if [ "$host" = "local" ]; then
    ( cd $ROOT && HF_HUB_OFFLINE=1 $PY $cmd ) > "$log" 2>&1
  else
    ssh $host "$RPY; HF_HUB_OFFLINE=1 python $cmd" > "$log" 2>&1
    rsync -az $host:A2SF/result_txt/pred/$BUD/$run/ $ROOT/result_txt/pred/$BUD/$run/ >> "$log" 2>&1
  fi
  local ov
  ov=$( cd $ROOT && $PY longbench_eval.py result_txt/pred/$BUD/$run 2>>"$log" | grep "Overall Average" | grep -oE '[0-9]+\.[0-9]+' | tail -1 )
  echo "${run},${ov:-FAIL}" >> $RES
  echo "[$(date +%H:%M:%S)] $host DONE  $run -> ${ov:-FAIL}" >> $LOGD/driver.log
}

worker() {
  local host="$1"; shift
  for spec in "$@"; do
    run_job "$host" "${spec%%:*}" "${spec##*:}"
  done
  echo "[$(date +%H:%M:%S)] $host LIST COMPLETE" >> $LOGD/driver.log
}

# 17 (local): mistral x6 + 1b{triattention,l2norm}
worker local \
  mistral-7b:keyformer mistral-7b:pyramidkv mistral-7b:chunkkv \
  mistral-7b:streamingllm mistral-7b:keydiff mistral-7b:l2norm \
  llama3-1b:triattention llama3-1b:l2norm &
P17=$!
# 18: 8b x6 + 1b{keyformer,pyramidkv,chunkkv}
worker eslab18 \
  llama3-8b:keyformer llama3-8b:pyramidkv llama3-8b:chunkkv \
  llama3-8b:streamingllm llama3-8b:keydiff llama3-8b:l2norm \
  llama3-1b:keyformer llama3-1b:pyramidkv llama3-1b:chunkkv &
P18=$!
# 20: qwen x6 + 1b{streamingllm,keydiff}
worker eslab20 \
  qwen2:keyformer qwen2:pyramidkv qwen2:chunkkv \
  qwen2:streamingllm qwen2:keydiff qwen2:l2norm \
  llama3-1b:streamingllm llama3-1b:keydiff &
P20=$!

wait $P17 $P18 $P20
echo "[$(date +%H:%M:%S)] ALL SERVERS COMPLETE" >> $LOGD/driver.log
