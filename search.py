import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import get_prompt

def make_a2sf_mask(attention_maps, prompt_length, total_budget, compression_ratio, forgetting_factor=1.00, b=1.0):
    a2sf_maps = attention_maps.clone()
    
    recent, select = compression_ratio
    
    recent_budget = round(recent * total_budget)
    select_budget = round(select * total_budget)
    
    exponent = (b * forgetting_factor**torch.arange(prompt_length-1,-1,-1, device=attention_maps.device)).view(1,1,-1,1)
    scores = (a2sf_maps[:,:,:prompt_length,:] * exponent).sum(dim=2, keepdim=True)
    
    for i in range(attention_maps.size(2)-prompt_length-1):
        current_pos = prompt_length + i
        window_start = current_pos - recent_budget
        
        # Select top-k tokens within the window
        selected_scores = scores[:,:,:,:window_start].topk(k=select_budget, dim=3).indices
        
        # Create and apply mask
        mask = torch.zeros_like(scores, device=attention_maps.device)
        mask[:,:,:,window_start:] = 1
        mask.scatter_(3, selected_scores, 1)
        
        # Apply mask and normalize
        a2sf_maps[:,:,current_pos+1:,:] = a2sf_maps[:,:,current_pos+1:,:] * mask
        divider = a2sf_maps[:,:,current_pos+1,:].sum(dim=2, keepdim=True)
        a2sf_maps[:,:,current_pos+1,:] = a2sf_maps[:,:,current_pos+1,:] / divider
        
        scores = scores * mask
        scores = scores + a2sf_maps[:,:,current_pos+1,:].unsqueeze(2)
    
    return a2sf_maps

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--prompt_length", type=int, default=300)
parser.add_argument("--generation_length", type=int, default=200)
parser.add_argument("--total_budget", type=int, default=100)
parser.add_argument("--num_prompts", type=int, default=100)
args = parser.parse_args()

PROMPT_LENGTH = args.prompt_length
GENERATION_LENGTH = args.generation_length
TOTAL_BUDGET = args.total_budget
NUM_PROMPTS = args.num_prompts

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model).to(torch.float16).to(device)

attention_map_buffer = []
values_buffer = []

# 각 프롬프트에 대해 처리
for prompt_idx in range(NUM_PROMPTS):
    prompt = get_prompt(prompt_idx)
    with torch.inference_mode():        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:,:PROMPT_LENGTH].to(device)
        
        past_key_values = None
        
        outputs = model(input_ids, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        
        for i in tqdm(range(GENERATION_LENGTH), desc="Token generation"):
            next_token_scores = next_token_logits
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=1)
            
            outputs = model(next_tokens.unsqueeze(-1), past_key_values=past_key_values, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        outputs = model(input_ids, output_attentions=True)
        attention_maps = torch.cat(outputs.attentions, dim=0).cpu().to(torch.float32)
        
        attention_map_buffer.append(attention_maps)
        values_buffer.append(torch.cat([past_key_values[i][1] for i in range(32)], dim=0).cpu().to(torch.float32))

        del outputs, next_token_logits, past_key_values
        torch.cuda.empty_cache()

del model, tokenizer
torch.cuda.empty_cache()

# Search Space
compression_ratio = [(0.05*i,0.05*(20-i)) for i in range(1, 20)]
a2sf_factors = [i/1000 for i in range(0,1001,2)]
bonus = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

conditions = []
for ratio in compression_ratio:
    for a2sf_factor in a2sf_factors:
        for b in bonus:
            conditions.append((ratio, a2sf_factor, b))  

# Store results for each total_budget
all_results = []

# Calculate original scores once for all prompts
conditions_results = [0 for _ in range(len(conditions))]

for prompt_idx in range(NUM_PROMPTS):
    attention_maps = attention_map_buffer[prompt_idx].to(device)

    values = values_buffer[prompt_idx].to(device)
    original_output = torch.matmul(attention_maps[:,:,PROMPT_LENGTH:,:], values)

    for cond_idx, condition in enumerate(tqdm(conditions, desc=f"Processing conditions")):
        compression_ratio = condition[0]
        forgetting_factors = condition[1]
        b = condition[2]
        
        condition_maps = make_a2sf_mask(attention_maps, PROMPT_LENGTH, TOTAL_BUDGET, compression_ratio, forgetting_factors, b)
        condition_output = torch.matmul(condition_maps[:,:,PROMPT_LENGTH:,:], values)
        conditions_results[cond_idx] += F.cosine_similarity(original_output, condition_output, dim=3).mean(dim=2)

conditions_results = torch.stack(conditions_results)

selected = conditions_results.mean(dim=2).max(dim=0).indices
selected_conditions = [conditions[i] for i in selected]

budgets = [selected_conditions[i][0] for i in range(len(selected_conditions))]
a2sf_factors = [selected_conditions[i][1] for i in range(len(selected_conditions))]

all_results.append({
    'total_budget': TOTAL_BUDGET,
    'budgets': budgets,
    'a2sf_factors': a2sf_factors
})

# Print all results at the end
print("\nSearch Results Summary:")
print("=" * 50)
for result in all_results:
    print(f"\nTotal Budget: {result['total_budget']}")
    formatted_budgets = [(round(r, 2), round(s, 2)) for r, s in result['budgets']]
    print(f"Budgets: {formatted_budgets}")
    print(f"A2SF Factors: {result['a2sf_factors']}")
    print("-" * 50)