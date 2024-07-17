#!/bin/bash

shots=$1

python -u generate_task_data.py --output-file data/openbookqa-${shots}.jsonl \
    --task-name openbookqa \
    --num-fewshot ${shots}

python -u generate_task_data.py --output-file data/piqa-${shots}.jsonl \
    --task-name piqa \
    --num-fewshot ${shots}

python -u generate_task_data.py --output-file data/arc_challenge-${shots}.jsonl \
    --task-name arc_challenge \
    --num-fewshot ${shots}

python -u generate_task_data.py --output-file data/arc_easy-${shots}.jsonl \
    --task-name arc_easy \
    --num-fewshot ${shots}

python -u generate_task_data.py --output-file data/mathqa-${shots}.jsonl \
    --task-name mathqa \
    --num-fewshot ${shots}