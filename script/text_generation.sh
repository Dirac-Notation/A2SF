#!/bin/bash

model_name=${1:-meta-llama/Llama-2-7b-hf}
cache_ratio=${2:-0.4}
penalty=${3:-0.2}

python -u run_text_generation.py \
    --model_name $model_name \
    --cache_ratio $cache_ratio \
    --penalty $penalty