import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import get_prompt

def get_attention_maps(model_name, device, prompt_idx=0, prompt_length=500):
    # Load model and tokenizer
    model2path = {"llama2": "meta-llama/Llama-2-7b-hf"}
    model_path = model2path[model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(torch.bfloat16).to(device)

    # Get model configuration
    num_heads = model.config.num_attention_heads
    num_groups = model.config.num_key_value_heads if hasattr(model.config, 'num_key_value_heads') else num_heads
    
    # Process prompt
    prompt = get_prompt(prompt_idx)
    prompt = f"[INST]{prompt}[/INST]"
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = torch.cat([input_ids[:, :prompt_length//2], input_ids[:, -prompt_length//2:]], dim=1).to(device)
    
    # Get attention maps
    outputs = model(input_ids, output_attentions=True)
    attn_shape = outputs.attentions[0].shape
    attention_maps = torch.cat(outputs.attentions, dim=0).view(-1, num_groups, num_heads // num_groups, *attn_shape[2:]).mean(dim=2)
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return attention_maps

def calculate_h2o_scores(attention_maps):
    h2o_scores = attention_maps[:,:,:-1,:].sum(dim=2)
    recent_scores = attention_maps[:,:,-51:-1,:].sum(dim=2)
    answer_scores = attention_maps[:,:,-1,:]

    return h2o_scores, recent_scores, answer_scores

def plot_attention_patterns(h2o_scores, recent_scores, answer_scores, layer_idx, head_idx, save_path):
    plt.figure(figsize=(15, 10))
    
    # Set font sizes
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 24,
        'axes.titlesize': 28,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })
    
    cpu_h2o_scores = h2o_scores.cpu().detach().to(torch.float16).numpy()
    cpu_recent_scores = recent_scores.cpu().detach().to(torch.float16).numpy()
    cpu_answer_scores = answer_scores.cpu().detach().to(torch.float16).numpy()
    
    # Plot H2O scores
    plt.plot(cpu_h2o_scores, label='H2O Score (0~499)', color='black')
    plt.plot(cpu_answer_scores, label='Answer Score (500)', color='green')
    
    # Plot recent scores at the bottom
    plt.fill_between(range(len(cpu_recent_scores)), 
                    0,  # Start from 0
                    cpu_recent_scores, 
                    alpha=0.3, color='blue', label='Recent Accumulative Score (449~499)')
    
    # Plot past scores on top of recent scores (derived from h2o_scores - recent_scores)
    past_scores = cpu_h2o_scores - cpu_recent_scores
    plt.fill_between(range(len(past_scores)), 
                    cpu_recent_scores,  # Start from recent scores
                    cpu_h2o_scores,     # End at H2O scores
                    alpha=0.3, color='red', label='Past Accumulative Score (0~449)')
    
    # Highlight Local Window (450-500)
    plt.axvspan(450, 500, alpha=0.1, color='gray', label='Local Window')
    
    # Get top 50 values and their indices for H2O scores (only from 0-450)
    h2o_scores_0_450 = cpu_h2o_scores[:450]
    top_h2o_indices = np.argsort(h2o_scores_0_450)[-50:]
    top_h2o_values = h2o_scores_0_450[top_h2o_indices]
    
    # Get top 50 values and their indices for answer scores
    answer_scores_0_450 = cpu_answer_scores[:450]
    top_answer_indices = np.argsort(answer_scores_0_450)[-50:]
    top_answer_values = answer_scores_0_450[top_answer_indices]
    
    # Plot top 50 H2O scores
    plt.scatter(top_h2o_indices, top_h2o_values, color='black', s=50, zorder=5)
    
    # Plot top 50 answer scores
    plt.scatter(top_answer_indices, top_answer_values, color='blue', s=50, zorder=5)
    
    plt.title(f'Attention Pattern Analysis (Layer {layer_idx}, Head {head_idx})')
    plt.xlabel('Token Position')
    plt.ylabel('Accumulative Attention Score')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.yscale('log')
    plt.ylim(top=1e3)  # Set y-axis maximum to 10^3
    
    # Apply tight layout
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index to use")
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get attention maps
    attention_maps = get_attention_maps("llama2", device)
    
    # Calculate scores
    h2o_scores, recent_scores, answer_scores = calculate_h2o_scores(attention_maps)
    
    # Create base directory
    base_dir = 'plots/motivation'
    os.makedirs(base_dir, exist_ok=True)
    
    # Plot for each head
    num_layers = attention_maps.size(0)
    num_heads = attention_maps.size(1)
    
    for layer_idx in range(num_layers):
        # Create layer directory
        layer_dir = os.path.join(base_dir, f'layer_{layer_idx}')
        os.makedirs(layer_dir, exist_ok=True)
        
        for head_idx in range(num_heads):
            save_path = os.path.join(layer_dir, f'head_{head_idx}.png')
            plot_attention_patterns(h2o_scores[layer_idx, head_idx], recent_scores[layer_idx, head_idx], answer_scores[layer_idx, head_idx], layer_idx, head_idx, save_path)

if __name__ == "__main__":
    main()
