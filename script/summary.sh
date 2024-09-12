#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python -u summary_test.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --cache_budget 250 \
    --forgetting_factor 0.4 \
    --data_path data/xsum-5shot.jsonl