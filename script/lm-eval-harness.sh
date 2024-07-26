#!/bin/bash

CUDA=$1

# CUDA_VISIBLE_DEVICES=$CUDA python lm-eval-harness.py \
#     --task_list openbookqa piqa arc_challenge arc_easy mathqa \
#     --model_list meta-llama/Llama-2-7b-hf \
#     --fewshot_list 3 \
#     --ratio_list 0.2

CUDA_VISIBLE_DEVICES=$CUDA python lm-eval-harness.py \
    --task_list openbookqa \
    --model_list meta-llama/Llama-2-7b-hf \
    --fewshot_list 3 \
    --ratio_list 0.2

CUDA_VISIBLE_DEVICES=$CUDA python lm-eval-harness.py \
    --task_list piqa \
    --model_list meta-llama/Llama-2-7b-hf \
    --fewshot_list 3 \
    --ratio_list 0.2

CUDA_VISIBLE_DEVICES=$CUDA python lm-eval-harness.py \
    --task_list arc_challenge \
    --model_list meta-llama/Llama-2-7b-hf \
    --fewshot_list 3 \
    --ratio_list 0.2

CUDA_VISIBLE_DEVICES=$CUDA python lm-eval-harness.py \
    --task_list arc_easy \
    --model_list meta-llama/Llama-2-7b-hf \
    --fewshot_list 3 \
    --ratio_list 0.2

CUDA_VISIBLE_DEVICES=$CUDA python lm-eval-harness.py \
    --task_list mathqa \
    --model_list meta-llama/Llama-2-7b-hf \
    --fewshot_list 3 \
    --ratio_list 0.2

# Dataset
## openbookqa piqa arc_challenge arc_easy mathqa

# Models
## meta-llama/Llama-2-7b-hf huggyllama/llama-7b facebook/opt-6.7b facebook/opt-2.7b