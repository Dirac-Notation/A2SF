#!/bin/bash

shots=$1

python -u generate_task_data.py --output-file lm/openbookqa-${shots}.jsonl \
    --task-name openbookqa \
    --num-fewshot ${shots}

python -u generate_task_data.py --output-file lm/piqa-${shots}.jsonl \
    --task-name piqa \
    --num-fewshot ${shots}

python -u generate_task_data.py --output-file lm/arc_challenge-${shots}.jsonl \
    --task-name arc_challenge \
    --num-fewshot ${shots}

python -u generate_task_data.py --output-file lm/arc_easy-${shots}.jsonl \
    --task-name arc_easy \
    --num-fewshot ${shots}

python -u generate_task_data.py --output-file lm/mathqa-${shots}.jsonl \
    --task-name mathqa \
    --num-fewshot ${shots}