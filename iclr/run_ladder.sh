#!/usr/bin/env bash
# Feasibility ladder eval: datasets round-robined over GPUS (default 1-7; GPU 0는
# 타 사용자 작업 상주로 제외). Resumable: eval_lb skips done samples/tables.
set -u
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/A2SF/bin/python
ROW=${ROW:?set ROW=fixed|agent|oracle|shuffled}
N=${N:-50}
FIXED_A=${FIXED_A:-0.1}
FIXED_B=${FIXED_B:-32}
CKPT=${CKPT:-iclr/runs/agent_1b/agent_best.pt}
MODEL=${MODEL:-llama3-1b}
TAG=${TAG:-feas1b}
# 실행 직전 빈 GPU 동적 스캔 (기본): 타 사용자 작업과의 경합 방지.
# GPUS 환경변수를 주면 그 목록을 강제 사용.
if [ -z "${GPUS:-}" ]; then
  GPUS=($(nvidia-smi --query-gpu=index,utilization.gpu,memory.used           --format=csv,noheader,nounits | awk -F', ' '$2<10 && $3<1500 {print $1}'))
  [ ${#GPUS[@]} -eq 0 ] && { echo "빈 GPU 없음 — 중단"; exit 1; }
  echo "동적 GPU 배정: ${GPUS[*]}"
else
  GPUS=($GPUS)
fi
DATASETS=(narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique \
          gov_report qmsum multi_news trec triviaqa samsum \
          passage_count passage_retrieval_en lcc repobench-p)
mkdir -p logs/ladder
NG=${#GPUS[@]}
for gi in $(seq 0 $((NG-1))); do
  (
    g=${GPUS[$gi]}
    di=$gi
    while [ $di -lt ${#DATASETS[@]} ]; do
      ds=${DATASETS[$di]}
      extra=""
      [[ "$ROW" == "fixed" ]] && extra="--fixed_a $FIXED_A --fixed_b $FIXED_B"
      [[ "$ROW" == "oracle" ]] && extra=""
      [[ "$ROW" == "agent" ]] && extra="--agent_ckpt $CKPT"
      CUDA_VISIBLE_DEVICES=$g $PY iclr/eval_lb.py --model "$MODEL" \
        --dataset "$ds" --row "$ROW" --n_samples "$N" --run_tag "$TAG" $extra \
        >> "logs/ladder/${TAG}_${ROW}_gpu$g.log" 2>&1
      di=$((di+NG))
    done
  ) &
done
wait
echo "LADDER ROW $ROW DONE"
$PY longbench_eval.py "result_txt/pred/128/${TAG}_${ROW}" 2>/dev/null | tail -20
