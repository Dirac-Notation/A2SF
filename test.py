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
GENERATION_LENGTH = 100
TOTAL_BUDGET = 100
NUM_GROUP = 100

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(torch.float16).to(device)

attention_map_buffer = []

# 각 프롬프트에 대해 처리
prompt = get_prompt(0)

with torch.inference_mode():        
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:,:400].to(device)
    
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
    attention_maps = torch.cat(outputs.attentions, dim=0).cpu()

    values = torch.cat([past_key_values[i][1] for i in range(len(past_key_values))], dim=0)

    del outputs, next_token_logits, past_key_values
    torch.cuda.empty_cache()
    
    attention_map_buffer.append(attention_maps)

del model, tokenizer
torch.cuda.empty_cache()

configs_a = load_configs("Llama-2-7b-chat-hf", "h2o", TOTAL_BUDGET)
configs_b = load_configs("Llama-2-7b-chat-hf", "a2sf", TOTAL_BUDGET)

compression_ratio_a = configs_a.compression_ratio
forgetting_factors_a = configs_a.forgetting_factors

compression_ratio_b = configs_b.compression_ratio
forgetting_factors_b = configs_b.forgetting_factors

attention_maps = attention_map_buffer[0].to(device)

# Create both masks
mask_a = make_a2sf_mask(attention_maps, PROMPT_LENGTH, TOTAL_BUDGET, compression_ratio_a, forgetting_factors_a)
mask_b = make_a2sf_mask(attention_maps, PROMPT_LENGTH, TOTAL_BUDGET, compression_ratio_b, forgetting_factors_b)

# Calculate outputs for both masks
original_output = torch.matmul(attention_maps, values)
masked_output_a = torch.matmul(mask_a, values)
masked_output_b = torch.matmul(mask_b, values)

# Get last GENERATION_LENGTH tokens
original_output = original_output[:,:,-GENERATION_LENGTH:,:]
masked_output_a = masked_output_a[:,:,-GENERATION_LENGTH:,:]
masked_output_b = masked_output_b[:,:,-GENERATION_LENGTH:,:]

# Calculate cosine similarities
similarity_a = F.cosine_similarity(original_output, masked_output_a, dim=3)
similarity_b = F.cosine_similarity(original_output, masked_output_b, dim=3)

# Reshape to (num_layers, num_heads)
num_layers = similarity_a.size(0)
num_heads = similarity_a.size(1)
similarity_a = similarity_a.view(num_layers, num_heads, -1).mean(dim=-1)
similarity_b = similarity_b.view(num_layers, num_heads, -1).mean(dim=-1)

# Calculate difference
similarity_diff = similarity_b - similarity_a

# Create figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Plot Mask A
avg_similarity_a = similarity_a.mean().cpu().detach().item()
im1 = ax1.imshow(similarity_a.cpu().detach(), cmap='RdBu', aspect='auto', vmin=-1.0, vmax=1.0)
ax1.set_title(f'H2O Mask\nAvg: {avg_similarity_a:.2f}')
ax1.set_xlabel('Head')
ax1.set_ylabel('Layer')
fig.colorbar(im1, ax=ax1, label='Cosine Similarity')

# Plot Mask B
avg_similarity_b = similarity_b.mean().cpu().detach().item()
im2 = ax2.imshow(similarity_b.cpu().detach(), cmap='RdBu', aspect='auto', vmin=-1.0, vmax=1.0)
ax2.set_title(f'A2SF Mask\nAvg: {avg_similarity_b:.2f}')
ax2.set_xlabel('Head')
ax2.set_ylabel('Layer')
fig.colorbar(im2, ax=ax2, label='Cosine Similarity')

# Plot Difference
diff_max = max(abs(similarity_diff.min()), abs(similarity_diff.max())).cpu().detach().item()
avg_diff = similarity_diff.mean().cpu().detach().item()
im3 = ax3.imshow(similarity_diff.cpu().detach(), cmap='RdBu', aspect='auto', vmin=-1.0, vmax=1.0)
ax3.set_title(f'Difference (A2SF - H2O)\nAvg: {avg_diff:.2f}')
ax3.set_xlabel('Head')
ax3.set_ylabel('Layer')
fig.colorbar(im3, ax=ax3, label='Difference in Cosine Similarity')

plt.tight_layout()
plt.savefig('plots/similarity_heatmap_comparison.png')
plt.close()