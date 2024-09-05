#!/bin/bash

python -u summary_test.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --cache_budget 200 \
    --forgetting_factor 0.1