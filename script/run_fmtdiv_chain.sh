#!/bin/bash
# fmtdiv scoring chain: 8B -> qwen -> (mistral if model dir ready) on GPUs 0-3.
set -u
cd /home/smp9898/A2SF
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in llama3-8b qwen2; do
  echo "=== [$(date +%H:%M)] scoring $m ==="
  HF_HUB_OFFLINE=1 python RL/dataset.py \
    --input datasets/training/raw/fmtdiv_v1/input.jsonl \
    --outdir datasets/training/raw/fmtdiv_v1_${m} \
    --model $m --budget 128 --gpus 0,1,2,3 --action_batch_size 5 \
    --actions "0:1,0.01:128,1:16,10:1,10:16"
done
# mistral: wait for the model rsync to finish (config.json present = complete enough to try)
until [ -f /home/smp9898/models/mistral-7b-v0.2/config.json ] && ! pgrep -f "rsync.*mistral-7b-v0.2" > /dev/null; do
  echo "[$(date +%H:%M)] waiting for mistral model rsync..."; sleep 120
done
echo "=== [$(date +%H:%M)] scoring mistral-7b ==="
HF_HUB_OFFLINE=1 python RL/dataset.py \
  --input datasets/training/raw/fmtdiv_v1/input.jsonl \
  --outdir datasets/training/raw/fmtdiv_v1_mistral-7b \
  --model mistral-7b --budget 128 --gpus 0,1,2,3 --action_batch_size 5 \
  --actions "0:1,0.01:128,1:16,10:1,10:16"
echo "=== CHAIN COMPLETE ==="
