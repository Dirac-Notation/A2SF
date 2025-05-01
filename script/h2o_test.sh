# python evaluate_llama_budgets.py \
#     --model_name meta-llama/Llama-2-7b-chat-hf \
#     --gpu 1 \
#     --datasets datasets/cnn_dailymail-3shot.jsonl \
#     --select_budget 50 100 150 200 250 300 350 400 450 500 \
#     --recent_budget 50 100 150 200 250 300 350 400 450 500 \
#     --random_budget 0 \
#     --streaming_budget 0 \
#     --forgetting_factor 1.00 \
#     --random_method att

python evaluate_llama_budgets.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --gpu 1 \
    --datasets datasets/xsum-3shot.jsonl \
    --select_budget 25 50 75 100 125 150\
    --recent_budget 25 50 75 100 125 150 \
    --random_budget 0 \
    --streaming_budget 0 \
    --forgetting_factor 1.00 \
    --random_method att