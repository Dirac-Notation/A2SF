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

from utils import get_prompt, make_a2sf_mask, load_configs

PROMPT_LENGTH = 400
GENERATION_LENGTH = 50
TOTAL_BUDGET = 100
NUM_PROMPTS = 4

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(torch.float16).to(device)

attention_map_buffer = []
values_buffer = []

# Process each prompt
for prompt_idx in range(NUM_PROMPTS):
    prompt = get_prompt(prompt_idx)
    
    with torch.inference_mode():        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:,:PROMPT_LENGTH].to(device)
        
        past_key_values = None
        
        outputs = model(input_ids, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        
        for i in tqdm(range(GENERATION_LENGTH), desc=f"Token generation for prompt {prompt_idx}"):
            next_token_scores = next_token_logits
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=1)
            
            outputs = model(next_tokens.unsqueeze(-1), past_key_values=past_key_values, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        outputs = model(input_ids, output_attentions=True)
        attention_maps = torch.cat(outputs.attentions, dim=0).cpu()
        values = torch.cat([past_key_values[i][1] for i in range(len(past_key_values))], dim=0)

        del outputs, next_token_logits, past_key_values
        torch.cuda.empty_cache()
        
        attention_map_buffer.append(attention_maps)
        values_buffer.append(values)

del model, tokenizer
torch.cuda.empty_cache()

# Store results for each prompt
all_similarities = []

for prompt_idx in range(NUM_PROMPTS):
    attention_maps = attention_map_buffer[prompt_idx].to(device)
    values = values_buffer[prompt_idx].to(device)
    
    ratios = []
    for i in range(1,10):
        ratios.append([(i/10, 1-i/10) for _ in range(32)])
    factors = [1.0 for _ in range(32)]

    masks = [make_a2sf_mask(attention_maps, PROMPT_LENGTH, TOTAL_BUDGET, ratios[i], factors).cpu() for i in range(9)]

    # Calculate outputs for all masks
    original_output = torch.matmul(attention_maps, values)[:,:,-GENERATION_LENGTH:,:]
    outputs = [torch.matmul(masks[i].to(device), values).cpu()[:,:,-GENERATION_LENGTH:,:] for i in range(9)]

    # Calculate cosine similarities
    similarities = [F.cosine_similarity(original_output, outputs[i].to(device)).cpu().mean() for i in range(9)]
    all_similarities.append(similarities)

# Calculate and print average similarities
avg_similarities = torch.mean(torch.tensor(all_similarities), dim=0)
print("\nAverage Similarities across all prompts:")
print("=" * 50)
for i, sim in enumerate(avg_similarities):
    print(f"Ratio {i+1}/10: {sim:.4f}")
print("=" * 50)