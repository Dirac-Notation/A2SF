#!/bin/bash
# Re-run attention-free scorers WITH sink preservation (n_sink=4) for normal accuracy.
# KeyDiff/L2-norm/TriAttention don't rank attention-sink tokens high, so the selector was
# evicting sinks -> crippled. n_sink=4 always keeps the first 4 tokens. New run_names
# (*_sink_*) keep the old (n_sink=0) results for comparison. Gated after window16 + NIAH.
set -u
ROOT=/home/smp9898/A2SF
PY=$HOME/miniconda3/envs/A2SF/bin/python
RPY='export PATH=$HOME/miniconda3/envs/A2SF/bin:$PATH; cd ~/A2SF'
BUD=128; SINK=4
LOGD=$ROOT/logs/campaign
RES=$LOGD/nsink_results.csv
DRV=$LOGD/driver.log
[ -f "$RES" ] || echo "run_name,overall" > "$RES"

declare -A TRISTATS=( [llama3-1b]=llama3-1b_stats.pt [llama3-8b]=llama3-8b_synth_stats.pt [qwen2]=qwen2_synth_stats.pt [mistral-7b]=mistral-7b_synth_stats.pt )

args_for(){ # method model
  case "$1" in
    keydiff) echo "--method keydiff --window 32 --n_sink $SINK";;
    l2norm)  echo "--method l2norm --window 32 --n_sink $SINK";;
    triattention) echo "--method triattention --n_sink $SINK --triattention_stats runs/triattention_stats/${TRISTATS[$2]}";;
  esac; }

run_job(){ # host model method
  local host="$1" model="$2" meth="$3"
  local run="${meth}_sink_${model}_${BUD}"
  local log="$LOGD/${run}.log"; local ea; ea="$(args_for "$meth" "$model")"
  local cmd="longbench.py --model $model --budget $BUD --gpus_per_model 1 --run_name $run $ea"
  echo "[$(date +%H:%M:%S)] $host START $run" >> $DRV
  rm -rf "$ROOT/result_txt/pred/$BUD/$run"
  if [ "$host" = "local" ]; then ( cd $ROOT && HF_HUB_OFFLINE=1 $PY $cmd ) > "$log" 2>&1
  else ssh $host "rm -rf A2SF/result_txt/pred/$BUD/$run" 2>/dev/null
       ssh $host "$RPY; HF_HUB_OFFLINE=1 python $cmd" > "$log" 2>&1
       rsync -az $host:A2SF/result_txt/pred/$BUD/$run/ $ROOT/result_txt/pred/$BUD/$run/ >> "$log" 2>&1; fi
  local ov; ov=$( cd $ROOT && $PY longbench_eval.py result_txt/pred/$BUD/$run 2>>"$log" | grep "Overall Average" | grep -oE '[0-9]+\.[0-9]+' | tail -1 )
  echo "${run},${ov:-FAIL}" >> "$RES"
  echo "[$(date +%H:%M:%S)] $host DONE  $run -> ${ov:-FAIL}" >> $DRV
}

gate(){ until grep -q 'WINDOW16 RECOMPARE ALL COMPLETE' "$DRV" 2>/dev/null && grep -q 'NIAH SWEEP ALL COMPLETE' "$DRV" 2>/dev/null; do sleep 180; done; }
gate

( for m in keydiff l2norm triattention; do run_job local mistral-7b $m; done; echo "[$(date +%H:%M:%S)] SINK mistral DONE" >> $DRV ) &
( for m in keydiff l2norm triattention; do run_job eslab18 llama3-8b $m; done
  for m in keydiff l2norm triattention; do run_job eslab18 llama3-1b $m; done
  echo "[$(date +%H:%M:%S)] SINK 8b+1b DONE" >> $DRV ) &
( for m in keydiff l2norm triattention; do run_job eslab20 qwen2 $m; done; echo "[$(date +%H:%M:%S)] SINK qwen DONE" >> $DRV ) &
wait
echo "[$(date +%H:%M:%S)] NSINK RERUN ALL COMPLETE" >> $DRV
