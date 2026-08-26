#!/usr/bin/env bash
# Generate 1B 13-action LongBench at budget 64 (test tight-budget per-prompt headroom).
cd /home/smp9898/A2SF; PY=$HOME/miniconda3/envs/A2SF/bin/python
B=64; MODEL=llama3-1b
ACTIONS=( "00:0:1" "01:0.01:1" "02:0.01:16" "03:0.01:32" "04:0.01:128"
          "05:0.1:1" "06:0.1:16" "07:0.1:32" "08:0.1:128"
          "09:10:1" "10:10:16" "11:10:32" "12:10:128" )
run_one(){ local spec=$1 gpu=$2
  local aidx=$(echo $spec|cut -d: -f1) a=$(echo $spec|cut -d: -f2) b=$(echo $spec|cut -d: -f3)
  local rn=${MODEL}_action${aidx}_${B}
  local alldir="result_txt/backup/${MODEL}/${B}/ALL/${rn}"
  [ "$(ls "$alldir"/*.jsonl 2>/dev/null|wc -l)" -ge 16 ] && { echo "SKIP $rn"; return; }
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$gpu $PY longbench.py --model $MODEL --method sigmoid \
    --sigmoid_a $a --window $b --budget $B --gpus_per_model 1 --run_name $rn >/dev/null 2>&1
  mkdir -p "$alldir"; rsync -a result_txt/pred/${B}/${rn}/ "$alldir/" 2>/dev/null
  echo "DONE $rn ($(cat result_txt/pred/${B}/${rn}/*.jsonl 2>/dev/null|wc -l) lines)"
}
# 8 GPU, 13 actions -> 2 waves
i=0
for spec in "${ACTIONS[@]}"; do
  run_one "$spec" $((i%8)) & i=$((i+1))
  (( i%8==0 )) && wait
done
wait
echo "GEN_1B_B64_DONE"
$PY script/build_lb_index_from_all.py --all_dir "result_txt/backup/${MODEL}/${B}/ALL" --out "runs/fast_lb_eval/index_${MODEL}_${B}.pt" 2>&1 | tail -2
echo "INDEX_1B_B64_DONE"
