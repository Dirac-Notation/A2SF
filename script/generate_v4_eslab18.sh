#!/bin/bash
# Generate sigmoid scores for v4 inputs on eslab18 (3090 × 8)
export PATH=$HOME/miniconda3/envs/A2SF/bin:$PATH
cd ~/A2SF

mkdir -p datasets/training/scored_v4/llama3-1b

python3 -u datasets/generate_sigmoid_dataset.py \
    --input  datasets/training/inputs_v4_eslab18.jsonl \
    --outdir datasets/training/scored_v4/llama3-1b \
    --model  llama3-1b \
    --budget 128 \
    --gpus   0,1,2,3,4,5,6,7 \
    --action_batch_size 4

echo "[eslab18] generation done"
