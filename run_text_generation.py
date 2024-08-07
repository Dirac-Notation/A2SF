import argparse
import logging

import numpy as np
import torch
import json
import tqdm 
import copy

from rouge import Rouge

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils_hh.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default='huggyllama/llama-7b')
    parser.add_argument("--cache_ratio", type=float, default=0.2)
    parser.add_argument("--penalty", type=float, default=0.1)
    parser.add_argument("--length", type=int, default=64)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.warning(f"device: {args.device}")

    rogue = Rouge()

    # Change to your custom prompt text
    prompt_text = 'In a small, bustling cafe nestled in the heart of a vibrant city, a serendipitous event unfolded, leaving a lasting impression on all who witnessed it. As the patrons sat sipping their coffees and engaging in animated conversations, a talented street musician entered the cafe, carrying a weathered guitar and radiating an aura of creativity.'

    model_name = args.model_name
    config = AutoConfig.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.half().eval()
    
    check_point = copy.deepcopy(model.state_dict())

    ######## Generate with Full Cache
    # input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(model.device)
    input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(args.device)

    model.to(args.device)
    generate_ids = model.generate(input_ids, max_new_tokens=args.length)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    result = result.replace(prompt_text, "")
    print("################## Generated Context with Full Cache ###################")
    print(result)
    print()

    ######### Enable HH
    config.heavy_ratio = args.cache_ratio/2
    config.recent_ratio = args.cache_ratio/2
    config.penalty = 1.0
    
    model = convert_kvcache_llama_heavy_recent(model, config)
    model.load_state_dict(check_point)
    model.half().eval().to(args.device)

    generate_ids_hh = model.generate(input_ids, max_new_tokens=args.length)
    result_hh = tokenizer.batch_decode(generate_ids_hh, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    result_hh = result_hh.replace(prompt_text, "")
    print("################## Generated Context with Heavy Hitter Oracle ###################")
    print(result_hh)
    print()

    score = rogue.get_scores(result_hh, result, avg=True)

    print(score)
    print()

    ######### Enable Decay
    config.heavy_ratio = args.cache_ratio
    config.recent_ratio = 0.0
    config.penalty = args.penalty
    
    model = convert_kvcache_llama_heavy_recent(model, config)
    model.load_state_dict(check_point)
    model.half().eval().to(args.device)

    generate_ids_hh = model.generate(input_ids, max_new_tokens=args.length)
    result_hh = tokenizer.batch_decode(generate_ids_hh, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    result_hh = result_hh.replace(prompt_text, "")
    print("################## Generated Context with H2O Decay ###################")
    print(result_hh)
    print()

    score = rogue.get_scores(result_hh, result, avg=True)

    print(score)
    print()