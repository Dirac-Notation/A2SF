import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import json
import random

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils_real_drop import Qwen2Tokenizer, Qwen2ForCausalLM

from utils import get_prompt

seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_sentence_exp(input_ids, puntuation_ids):
    orig_shape = input_ids.shape
    
    flattened_input_ids = input_ids.view(-1)
    num_all = flattened_input_ids.size(0)
    
    pos = torch.isin(flattened_input_ids, torch.tensor(puntuation_ids, device=input_ids.device)).nonzero(as_tuple=True)[0].tolist()
    num_t = len(pos)
    
    starts = [0] + [p + 1 for p in pos]
    ends   = pos + [num_all - 1]
    
    exponents = torch.empty_like(flattened_input_ids)
    for i, (s, e) in enumerate(zip(starts, ends)):
        exponents[s : e + 1] = num_t - i
    exponents = exponents.view(orig_shape[0], 1, orig_shape[1])
    
    return exponents

def make_layerwise_a2sf_mask(
    attention_maps,
    prompt_length,
    input_ids,
    total_budget,
    budget_ratio,
    a2sf_factor,
    local_ratio,
    sentence_exp,
    puntuation_ids
    ):
    a2sf_maps = attention_maps.clone()
    
    layer_cache_budget = int(total_budget * budget_ratio)
    layer_recent_budget = round(layer_cache_budget * local_ratio)
    layer_select_budget = round(layer_cache_budget * (1-local_ratio))
    
    forgetting = (a2sf_factor**sentence_exp).view(1,-1,1)
    layer_scores = (a2sf_maps[:,:prompt_length,:] * forgetting).sum(dim=1, keepdim=True)
    
    for i in range(attention_maps.size(2)-prompt_length):
        current_pos = prompt_length + i
        window_start = current_pos - layer_recent_budget
        selected_scores = layer_scores[:,:,:window_start].topk(k=layer_select_budget, dim=2).indices
        
        mask = torch.zeros_like(layer_scores, device=attention_maps.device)
        mask[:,:,window_start:] = 1
        mask.scatter_(2, selected_scores, 1)
        
        a2sf_maps[:,current_pos:,:] = a2sf_maps[:,current_pos:,:] * mask
        divider = a2sf_maps[:,current_pos,:].sum(dim=1, keepdim=True)
        a2sf_maps[:,current_pos,:] = a2sf_maps[:,current_pos,:] / divider
        
        layer_scores = layer_scores * mask
        layer_scores = layer_scores + a2sf_maps[:,current_pos,:].unsqueeze(1)
        
        if input_ids[0,current_pos].item() in puntuation_ids:
            layer_scores *= a2sf_factor
            
    return a2sf_maps

def load_model_and_tokenizer(model_name, model_path, device):
    """Load model and tokenizer based on model name."""
    if "qwen" in model_name.lower():
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        model = Qwen2ForCausalLM.from_pretrained(model_path).to(torch.float16).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(torch.float16).to(device)
    return model, tokenizer

def get_punctuation_ids(tokenizer):
    """Get punctuation token IDs from tokenizer."""
    return [
        tokenizer.encode(".", add_special_tokens=False)[0],
        tokenizer.encode(" .", add_special_tokens=False)[0],
    ]

def process_single_prompt(model, tokenizer, prompt, prompt_length, generation_length, model_name, puntuation_ids, device):
    """Process a single prompt and return attention maps and values."""
    with torch.inference_mode():
        if "llama" in model_name.lower():
            prompt = f"[INST]{prompt}[/INST]"
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = torch.cat([input_ids[:, :prompt_length//2], input_ids[:, -prompt_length//2:]], dim=1).to(device)
        
        sentence_exp = make_sentence_exp(input_ids, puntuation_ids)
        
        outputs = model(input_ids, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        
        for _ in range(generation_length):
            next_token_scores = next_token_logits
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=1)
            
            outputs = model(next_tokens.unsqueeze(-1), past_key_values=past_key_values, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        outputs = model(input_ids, output_attentions=True)

        num_heads = model.config.num_attention_heads
        num_groups = model.config.num_key_value_heads if hasattr(model.config, 'num_key_value_heads') else num_heads
        
        attn_shape = outputs.attentions[0].shape
        
        input_ids = input_ids.cpu()
        attention_maps = torch.cat(outputs.attentions, dim=0).view(-1, num_groups, num_heads // num_groups, *attn_shape[2:]).mean(dim=2).cpu()
        values = torch.cat([past_key_values[i][1] for i in range(num_heads)], dim=0).cpu()
        sentence_exp = sentence_exp.cpu()

        return attention_maps, values, input_ids, sentence_exp

def process_model(model_name, device, args):
    prompt_length = args.prompt_length
    generation_length = args.generation_length
    total_budget = args.total_budget
    num_prompts = args.num_prompts
    
    print(f"\nProcessing model: {model_name}")
    
    # Load model and tokenizer
    model2path = json.load(open("config/model2path.json", "r"))
    model_path = model2path[model_name]
    
    model, tokenizer = load_model_and_tokenizer(model_name, model_path, device)
    puntuation_ids = get_punctuation_ids(tokenizer)
    
    # Get model configuration
    num_heads = model.config.num_attention_heads
    num_groups = model.config.num_key_value_heads if hasattr(model.config, 'num_key_value_heads') else num_heads
    
    print(f"Model config: num_attention_heads={num_heads}, num_key_value_heads={num_groups}")
    
    # Process prompts
    attention_map_buffer = []
    values_buffer = []
    input_ids_buffer = []
    sentence_exp_buffer = []
    
    for prompt_idx in tqdm(range(num_prompts), desc="Processing prompts"):
        prompt = get_prompt(prompt_idx)
        attention_maps, values, input_ids, sentence_exp = process_single_prompt(model, tokenizer, prompt, prompt_length, generation_length, model_name, puntuation_ids, device)
        
        attention_map_buffer.append(attention_maps)
        values_buffer.append(values)
        input_ids_buffer.append(input_ids)
        sentence_exp_buffer.append(sentence_exp)
        
        torch.cuda.empty_cache()
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
    # Search space
    factor_step = 0.01
    local_ratio_step = 0.1
    local_ratios = [local_ratio_step*i for i in range(int(1/local_ratio_step)+1)]
    a2sf_factors = [factor_step*i for i in range(int(1/factor_step)+1)]
    
    layerwise_local_ratio = [0.5 for i in range(32)]
    layerwise_a2sf_factors = [1.0 for i in range(32)]
    layerwise_budget_ratio = [1.0 for i in range(32)]
    
    for prompt_idx in tqdm(range(num_prompts)):
        attention_maps = attention_map_buffer[prompt_idx].to(device)
        values = values_buffer[prompt_idx].to(device)
        input_ids = input_ids_buffer[prompt_idx].to(device)
        sentence_exp = sentence_exp_buffer[prompt_idx].to(device)
    
        original_output = torch.matmul(attention_maps[:,:,prompt_length:,:], values)
        original_output = original_output.transpose(1, 2).contiguous()
        original_output = original_output.reshape(original_output.size(0), original_output.size(1), -1)
        
        if args.local_search:
            a2sf_results = [0.0 for _ in range(len(local_ratios))]                     
            for layer_idx in tqdm(range(attention_maps.size(0))):
                a2sf_factor = layerwise_a2sf_factors[layer_idx]
                layer_ratio = layerwise_budget_ratio[layer_idx]

                for local_ratio_idx, local_ratio in enumerate(local_ratios):
                    condition_maps = make_layerwise_a2sf_mask(attention_maps[layer_idx], prompt_length, input_ids, total_budget, layer_ratio, a2sf_factor, local_ratio, sentence_exp, puntuation_ids)
                    condition_output = torch.matmul(condition_maps[:,prompt_length:,:], values[layer_idx])
                    condition_output = condition_output.transpose(0, 1).contiguous()
                    condition_output = condition_output.reshape(condition_output.size(0), -1)
                    a2sf_results[local_ratio_idx] += F.cosine_similarity(original_output[layer_idx], condition_output, dim=1).mean(dim=0).item()
                
                max_idx = a2sf_results.index(max(a2sf_results))
                layerwise_local_ratio[layer_idx] = local_ratios[max_idx]
        
        if args.factor_search:
            a2sf_results = [0.0 for _ in range(len(a2sf_factors))]                     
            for layer_idx in tqdm(range(attention_maps.size(0))):
                layer_ratio = layerwise_budget_ratio[layer_idx]
                local_ratio = layerwise_local_ratio[layer_idx]

                for a2sf_factor_idx, a2sf_factor in enumerate(a2sf_factors):
                    condition_maps = make_layerwise_a2sf_mask(attention_maps[layer_idx], prompt_length, input_ids, total_budget, layer_ratio, a2sf_factor, local_ratio, sentence_exp, puntuation_ids)
                    condition_output = torch.matmul(condition_maps[:,prompt_length:,:], values[layer_idx])
                    condition_output = condition_output.transpose(0, 1).contiguous()
                    condition_output = condition_output.reshape(condition_output.size(0), -1)
                    a2sf_results[a2sf_factor_idx] += F.cosine_similarity(original_output[layer_idx], condition_output, dim=1).mean(dim=0).item()
                
                max_idx = a2sf_results.index(max(a2sf_results))
                layerwise_a2sf_factors[layer_idx] = a2sf_factors[max_idx]
        
        if args.ratio_search:
            condition_maps = []
            for layer_idx in range(attention_maps.size(0)):
                layer_a2sf_factor = layerwise_a2sf_factors[layer_idx]
                layer_ratio = layerwise_budget_ratio[layer_idx]
                
                condition_maps.append(
                    make_layerwise_a2sf_mask(attention_maps[layer_idx], prompt_length, input_ids, total_budget, layer_ratio, layer_a2sf_factor, local_ratio, sentence_exp, puntuation_ids)
                )
            
            condition_maps = torch.stack(condition_maps, dim=0)
            condition_output = torch.matmul(condition_maps[:,:,prompt_length:,:], values)
            condition_output = condition_output.transpose(1, 2).contiguous()
            condition_output = condition_output.reshape(condition_output.size(0), condition_output.size(1), -1)
            sim_score = F.cosine_similarity(original_output, condition_output, dim=2).mean(dim=1)
            
            for _ in tqdm(range(100)):
                min_idx = sim_score.argmin()
                max_idx = sim_score.argmax()
                
                layerwise_budget_ratio[min_idx] += 0.01
                layerwise_budget_ratio[max_idx] -= 0.01
                
                condition_maps[min_idx] = make_layerwise_a2sf_mask(attention_maps[min_idx], prompt_length, input_ids, total_budget, layerwise_budget_ratio[min_idx], layerwise_a2sf_factors[min_idx], sentence_exp, puntuation_ids)
                condition_maps[max_idx] = make_layerwise_a2sf_mask(attention_maps[max_idx], prompt_length, input_ids, total_budget, layerwise_budget_ratio[max_idx], layerwise_a2sf_factors[max_idx], sentence_exp, puntuation_ids)
                
                condition_output = torch.matmul(condition_maps[:,:,prompt_length:,:], values)
                condition_output = condition_output.transpose(1, 2).contiguous()
                condition_output = condition_output.reshape(condition_output.size(0), condition_output.size(1), -1)
                sim_score = F.cosine_similarity(original_output, condition_output, dim=2).mean(dim=1)

        layerwise_local_ratio = [round(ratio, 2) for ratio in layerwise_local_ratio]
        layerwise_a2sf_factors = [round(factor, 2) for factor in layerwise_a2sf_factors]
        layerwise_budget_ratio = [round(ratio, 2) for ratio in layerwise_budget_ratio]
    
    return {
        "model": model_name,
        "layerwise_ratio": layerwise_budget_ratio,
        "forgetting_factors": layerwise_a2sf_factors,
        "local_ratios": layerwise_local_ratio
    }

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Process each model
    results = []
    for model_name in args.models:
        result = process_model(
            model_name=model_name,
            device=device,
            args=args,
        )
        results.append(result)
        
        # Print results for this model
        print("\nSearch Results Summary:")
        print("=" * 50)
        print(f"Model: {model_name}")
        print(f"Total Budget: {args.total_budget}")
        print(f'''
\"layerwise_ratio\": {result['layerwise_ratio']},
\"forgetting_factors\": {result['forgetting_factors']},
\"local_ratios\": {result['local_ratios']}
''')
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--models", type=str, nargs='+', default=["llama2"], choices=["llama", "llama2", "llama3", "opt", "qwen2"])
    parser.add_argument("--prompt_length", type=int, default=900)
    parser.add_argument("--generation_length", type=int, default=100)
    parser.add_argument("--num_prompts", type=int, default=5)
    parser.add_argument("--total_budget", type=int, default=100)
    parser.add_argument("--ratio_search", action="store_true", default=False)
    parser.add_argument("--factor_search", action="store_true", default=False)
    parser.add_argument("--local_search", action="store_true", default=False)
    args = parser.parse_args()
    
    main(args)