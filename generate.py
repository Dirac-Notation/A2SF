import torch
import argparse
from utils_real_drop.kv_llama import LlamaForCausalLM
from transformers import AutoTokenizer

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
prompt_list = [
    "The Schrödinger equation is a partial differential equation that governs the wave function of a non-relativistic quantum-mechanical system.",
    "Its discovery was a significant landmark in the development of quantum mechanics.",
    "It is named after Erwin Schrödinger, an Austrian physicist, who postulated the equation in 1925 and published it in 1926, forming the basis for the work that resulted in his Nobel Prize in Physics in 1933.",
    "Conceptually, the Schrödinger equation is the quantum counterpart of Newton's second law in classical mechanics.",
    "Given a set of known initial conditions, Newton's second law makes a mathematical prediction as to what path a given physical system will take over time.",
    "The Schrödinger equation gives the evolution over time of the wave function, the quantum-mechanical characterization of an isolated physical system.",
    "The equation was postulated by Schrödinger based on a postulate of Louis de Broglie that all matter has an associated matter wave.",
    "The equation predicted bound states of the atom in agreement with experimental observations.",
    "The Schrödinger equation is not the only way to study quantum mechanical systems and make predictions.",
    "Other formulations of quantum mechanics include matrix mechanics, introduced by Werner Heisenberg, and the path integral formulation, developed chiefly by Richard Feynman.",
    "When these approaches are compared, the use of the Schrödinger equation is sometimes called \"wave mechanics\".",
    "The equation given by Schrödinger is nonrelativistic because it contains a first derivative in time and a second derivative in space, and therefore space and time are not on equal footing.",
    "Paul Dirac incorporated special relativity and quantum mechanics into a single formulation that simplifies to the Schrödinger equation in the non-relativistic limit.",
    "This is the Dirac equation, which contains a single derivative in both space and time.",
    "Another partial differential equation, the Klein–Gordon equation, led to a problem with probability density even though it was a relativistic wave equation.",
    "The probability density could be negative, which is physically unviable.",
    "This was fixed by Dirac by taking the so-called square root of the Klein–Gordon operator and in turn introducing Dirac matrices.",
    "In a modern context, the Klein–Gordon equation describes spin-less particles, while the Dirac equation describes spin-1/2 particles."
]
prompt = " ".join(prompt_list)

# 모델과 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to(torch.bfloat16).to(device)

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