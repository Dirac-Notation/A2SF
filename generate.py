import torch
import argparse
from utils_real_drop.kv_llama import KVLlamaForCausalLM
from transformers import AutoTokenizer
from utils import get_prompt

# CLI arguments (only cache settings and device)
parser = argparse.ArgumentParser(description="Llama2 generation with RoPE cache settings via CLI")
parser.add_argument('--gpu', type=str, default="1", help="GPU device number to use")
parser.add_argument('--use_compression', action='store_true', default=False, help="Whether to use compression for cache")
parser.add_argument('--select_budget', type=int, default=50, help="select_budget for cache")
parser.add_argument('--recent_budget', type=int, default=50, help="recent_budget for cache")
parser.add_argument('--random_budget', type=int, default=0, help="random_budget for cache")
parser.add_argument('--streaming_budget', type=int, default=0, help="streaming_budget for cache")
parser.add_argument('--random_method', type=str, default="att", help="random_method for cache")
parser.add_argument('--forgetting_factor', type=float, default=1.0, help="forgetting_factor for cache")
args = parser.parse_args()

# assign CLI args
device = f"cuda:{args.gpu}"

# Hardcoded model name and example prompt (>200 tokens)
model_name = "meta-llama/Llama-2-7b-chat-hf"
prompt = get_prompt()

# 모델과 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = KVLlamaForCausalLM.from_pretrained(model_name).to(torch.bfloat16).to(device)

with torch.inference_mode():
    # 입력 프롬프트 준비
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # 캐시 초기화 (예시 버짓)
    model.init_cache(
        use_compression=args.use_compression,
        select_budget=args.select_budget,
        recent_budget=args.recent_budget,
        random_budget=args.random_budget,
        streaming_budget=args.streaming_budget,
        random_method=args.random_method,
        forgetting_factor=args.forgetting_factor
    )

    # 초기 forward pass
    position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
    outputs = model(input_ids, position_ids=position_ids)
    next_token_logits = outputs.logits[:, -1, :]

    # 프롬프트 출력
    print(prompt, end=" ", flush=True)

    # 토큰 생성 (fixed 128 tokens)
    for i in range(256):
        # 다음 토큰 예측
        next_token_scores = next_token_logits
        next_tokens = torch.argmax(next_token_scores, dim=-1)
        
        # 현재 생성된 토큰 디코딩 및 출력
        current_token = tokenizer.decode(next_tokens[0], skip_special_tokens=True)
        print(current_token, end=" ", flush=True)
        
        # 다음 forward pass를 위한 position_ids 계산
        current_position = torch.tensor([[input_ids.shape[1] + i]], dtype=torch.long, device=device)
        
        # 다음 forward pass
        outputs = model(next_tokens.unsqueeze(-1), position_ids=current_position)
        next_token_logits = outputs.logits[:, -1, :]

    print()
