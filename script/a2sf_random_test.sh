python evaluate_llama_budgets.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --gpu 3 \
    --datasets fewshot_data/cnn_dailymail-3shot.jsonl \
    --select_budget 45 95 145 195 245 295 \
    --recent_budget 50 100 150 200 250 300 \
    --random_budget 5 \
    --streaming_budget 0 \
    --forgetting_factor 0.99 \
    --random_method random