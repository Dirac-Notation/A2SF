import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import json

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils_real_drop import Qwen2Tokenizer, Qwen2ForCausalLM

from utils import get_prompt

def make_a2sf_mask(attention_maps, prompt_length, input_ids, total_budget, compression_ratio, forgetting_factor, puntuation_ids):
    a2sf_maps = attention_maps.clone()
    
    recent = compression_ratio
    
    recent_budget = round(recent * total_budget)
    select_budget = round((1-recent) * total_budget)
    
    prompt_ids = input_ids[:, :PROMPT_LENGTH]
    orig_shape = prompt_ids.shape
    
    flattened_input_ids = prompt_ids.view(-1)
    num_all = flattened_input_ids.size(0)
    
    pos = torch.isin(flattened_input_ids, torch.tensor(puntuation_ids, device=input_ids.device)).nonzero(as_tuple=True)[0].tolist()
    num_t = len(pos)
    
    starts = [0] + [p + 1 for p in pos]
    ends   = pos + [num_all - 1]

    exponents = torch.empty_like(flattened_input_ids)
    for i, (s, e) in enumerate(zip(starts, ends)):
        exponents[s : e + 1] = num_t - i
    exponents = exponents.view(orig_shape[0], 1, orig_shape[1])

    forgetting = (forgetting_factor**exponents).view(1,1,-1,1)

    scores = (a2sf_maps[:,:,:prompt_length,:] * forgetting).sum(dim=2, keepdim=True)
    
    for i in range(attention_maps.size(2)-prompt_length-1):
        current_pos = prompt_length + i
        window_start = current_pos - recent_budget
        
        selected_scores = scores[:,:,:,:window_start].topk(k=select_budget, dim=3).indices
        
        mask = torch.zeros_like(scores, device=attention_maps.device)
        mask[:,:,:,window_start:] = 1
        mask.scatter_(3, selected_scores, 1)
        
        a2sf_maps[:,:,current_pos+1:,:] = a2sf_maps[:,:,current_pos+1:,:] * mask
        divider = a2sf_maps[:,:,current_pos+1,:].sum(dim=2, keepdim=True)
        a2sf_maps[:,:,current_pos+1,:] = a2sf_maps[:,:,current_pos+1,:] / divider
        
        scores = scores * mask
        scores = scores + a2sf_maps[:,:,current_pos+1,:].unsqueeze(2)
        
        if input_ids[0,current_pos].item() in puntuation_ids:
            scores *= forgetting_factor
            
    return a2sf_maps

def process_model(model_name, device, prompt_length, generation_length, total_budget, num_prompts):
    print(f"\nProcessing model: {model_name}")
    
    # Load model and tokenizer
    model2path = json.load(open("config/model2path.json", "r"))
    model_path = model2path[model_name]
    
    if "qwen" in model_name.lower():
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        model = Qwen2ForCausalLM.from_pretrained(model_path).to(torch.float16).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(torch.float16).to(device)

    punctuation_ids = [
        tokenizer.encode(".", add_special_tokens=False)[0],
        tokenizer.encode(" .", add_special_tokens=False)[0],
    ]

    # Get model configuration
    num_heads = model.config.num_attention_heads
    num_groups = model.config.num_key_value_heads if hasattr(model.config, 'num_key_value_heads') else num_heads
    
    print(f"Model config: num_attention_heads={num_heads}, num_key_value_heads={num_groups}")
    
    # Process prompts
    attention_map_buffer = []
    values_buffer = []
    
    for prompt_idx in range(num_prompts):
        prompt = get_prompt(prompt_idx)
        with torch.inference_mode():
            if "llama" in model_name.lower():
                prompt = f"[INST]{prompt}[/INST]"
            
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            input_ids = torch.cat([input_ids[:, :prompt_length//2], input_ids[:, -prompt_length//2:]], dim=1).to(device)
            
            outputs = model(input_ids, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            for i in tqdm(range(generation_length), desc="Token generation"):
                next_token_scores = next_token_logits
                next_tokens = torch.argmax(next_token_scores, dim=-1)
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=1)
                
                outputs = model(next_tokens.unsqueeze(-1), past_key_values=past_key_values, use_cache=True)
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

            outputs = model(input_ids, output_attentions=True)

            attn_shape = outputs.attentions[0].shape
            attention_maps = torch.cat(outputs.attentions, dim=0).view(-1, num_groups, num_heads // num_groups, *attn_shape[2:]).mean(dim=2).cpu()
            past_key_values = outputs.past_key_values
            
            attention_map_buffer.append(attention_maps)
            values_buffer.append(torch.cat([past_key_values[i][1] for i in range(num_heads)], dim=0).cpu())

            del outputs, next_token_logits, past_key_values
            torch.cuda.empty_cache()
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
    # Search space
    ratio_step = 0.05
    factor_step = 0.01
    # compression_ratio = [ratio_step*i for i in range(1, int(1/ratio_step))]
    a2sf_factors = [factor_step*i for i in range(int(1/factor_step)+1)]
    
    # Fixed
    compression_ratio = [0.5 for i in range(32)]
    # a2sf_factors = [1.00 for i in range(32)]
    
    conditions = []
    for ratio in compression_ratio:
        for a2sf_factor in a2sf_factors:
            conditions.append((ratio, a2sf_factor))
    
    conditions_results = [0 for _ in range(len(conditions))]
    
    for prompt_idx in range(num_prompts):
        attention_maps = attention_map_buffer[prompt_idx].to(device)
        values = values_buffer[prompt_idx].to(device)
        
        original_output = torch.matmul(attention_maps[:,:,prompt_length:,:], values)
        original_output = original_output.transpose(1, 2).contiguous()
        original_output = original_output.reshape(original_output.size(0), -1, original_output.size(3))
        
        for cond_idx, condition in enumerate(tqdm(conditions, desc=f"Processing conditions")):
            compression_ratio = condition[0]
            forgetting_factors = condition[1]
            
            condition_maps = make_a2sf_mask(attention_maps, prompt_length, input_ids, total_budget, compression_ratio, forgetting_factors, punctuation_ids)
            condition_output = torch.matmul(condition_maps[:,:,prompt_length:,:], values)
            condition_output = condition_output.transpose(1, 2).contiguous()
            condition_output = condition_output.reshape(condition_output.size(0), -1, condition_output.size(3))
            conditions_results[cond_idx] += F.cosine_similarity(original_output, condition_output, dim=2).mean(dim=1)
    
    conditions_results = torch.stack(conditions_results)
    selected = conditions_results.max(dim=0).indices
    selected_conditions = [conditions[i] for i in selected]
    
    budgets = [round(selected_conditions[i][0], 2) for i in range(len(selected_conditions))]
    a2sf_factors = [round(selected_conditions[i][1], 2) for i in range(len(selected_conditions))]
    
    return {
        "model": model_name,
        "compression_method": "a2sf",
        "compression_ratio": budgets,
        "forgetting_factors": a2sf_factors
    }

def save_config(results, output_file="config/config.jsonl"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Process each model
    results = []
    for model_name in args.models:
        result = process_model(
            model_name=model_name,
            device=device,
            prompt_length=args.prompt_length,
            generation_length=args.generation_length,
            total_budget=args.total_budget,
            num_prompts=args.num_prompts
        )
        results.append(result)
        
        # Print results for this model
        print("\nSearch Results Summary:")
        print("=" * 50)
        print(f"Model: {model_name}")
        print(f"Total Budget: {args.total_budget}")
        print(f"Compression Method: {result['compression_method']}")
        print(f"Compression Ratios: {result['compression_ratio']}")
        print(f"Forgetting Factors: {result['forgetting_factors']}")
        print("-" * 50)
    
    # Save all results to config file
    save_config(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--models", type=str, nargs='+', default=["llama2"], choices=["llama", "llama2", "llama3", "opt", "qwen2"])
    parser.add_argument("--prompt_length", type=int, default=950)
    parser.add_argument("--generation_length", type=int, default=50)
    parser.add_argument("--total_budget", type=int, default=200)
    parser.add_argument("--num_prompts", type=int, default=5)
    args = parser.parse_args()
    
    # Set global variables
    PROMPT_LENGTH = args.prompt_length
    GENERATION_LENGTH = args.generation_length
    TOTAL_BUDGET = args.total_budget
    NUM_PROMPTS = args.num_prompts
    
    main(args)