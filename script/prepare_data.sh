#!/bin/bash

python -u generate_task_data.py \
    --task_list openbookqa piqa arc_challenge arc_easy mathqa \
    --fewshot_list 0 1 2 3 4 5 \