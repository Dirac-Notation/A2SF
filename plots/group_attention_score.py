import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import make_optimal_mask, make_a2sf_mask, load_configs, generate

def load_datasets(
    dataset_path: str,
    tokenizer
):
    with open(dataset_path, "r") as f:
        datalines = f.readlines()
        articles = []
        
        for dataline in datalines:
            articles.append(json.loads(dataline))

    prompts = []
    answers = []
    output_indices = []

    for data in tqdm(articles, desc="Tokenizing"):
        input = data["article"]
        answer = data["summary_gt"]
        
        input_ids = tokenizer(input, return_tensors="pt").input_ids
        output_ids = tokenizer(answer, return_tensors="pt").input_ids

        prompts.append(input_ids)
        answers.append(answer)
        output_indices.append(output_ids)
    
    return prompts, answers, output_indices

def proportional_grouping_mean(attention_row, total_length, num_groups=100):
    # Calculate group sizes based on proportions
    group_sizes = [int(total_length * (i+1)/num_groups) - int(total_length * i/num_groups) for i in range(num_groups)]
    scores = []
    
    start_idx = 0
    for size in group_sizes:
        end_idx = start_idx + size
        scores.append(attention_row[:,:,start_idx:end_idx].mean(dim=2))
        start_idx = end_idx
    
    return torch.stack(scores, dim=2)

def proportional_grouping_sum(attention_row, total_length, num_groups=100):
    # Calculate group sizes based on proportions
    group_sizes = [int(total_length * (i+1)/num_groups) - int(total_length * i/num_groups) for i in range(num_groups)]
    scores = []
    
    start_idx = 0
    for size in group_sizes:
        end_idx = start_idx + size
        scores.append(attention_row[:,:,start_idx:end_idx].sum(dim=2))
        start_idx = end_idx
    
    return torch.stack(scores, dim=2)

def plot_average_attention_scores(original_ratio_scores, h2o_ratio_scores, a2sf_ratio_scores, h2o_small_ratio_scores, h2o_large_ratio_scores,
                                original_raw_scores, h2o_raw_scores, a2sf_raw_scores, h2o_small_raw_scores, h2o_large_raw_scores):
    # Calculate averages across all layers and heads
    avg_h2o_ratio = h2o_ratio_scores.mean(dim=(0,1))
    avg_a2sf_ratio = a2sf_ratio_scores.mean(dim=(0,1))
    
    avg_original_raw = original_raw_scores.mean(dim=(0,1))
    avg_h2o_raw = h2o_raw_scores.mean(dim=(0,1))
    avg_a2sf_raw = a2sf_raw_scores.mean(dim=(0,1))

    graph_x_size = avg_h2o_ratio.size(0)
    avg_seq_length = (sum(prompt_lengths) + sum(generation_lengths))/len(prompt_lengths)
    
    h2o_ratio = 0.5*args.total_budget/avg_seq_length
    a2sf_ratio = torch.tensor([i for i,_ in a2sf_configs.compression_ratio]).mean().item()*args.total_budget/avg_seq_length
    
    h2o_window = (1 - h2o_ratio) * graph_x_size
    a2sf_window = (1 - a2sf_ratio) * graph_x_size
    
    # Define section boundaries and colors (10 sections)
    sections = [
        (0, 10, '#FFB6C1'),    # Light Pink
        (10, 20, '#98FB98'),   # Pale Green
        (20, 30, '#87CEEB'),   # Sky Blue
        (30, 40, '#DDA0DD'),   # Plum
        (40, 50, '#F0E68C'),   # Khaki
        (50, 60, '#E6E6FA'),   # Lavender
        (60, 70, '#FFA07A'),   # Light Salmon
        (70, 80, '#B0E0E6'),   # Powder Blue
        (80, 90, '#FFDAB9'),   # Peach
        (90, 100, '#D8BFD8')   # Thistle
    ]
    
    # Function to add background colors
    def add_section_backgrounds(ax):
        for start, end, color in sections:
            start_idx = int(start * graph_x_size / 100)
            end_idx = int(end * graph_x_size / 100)
            ax.axvspan(start_idx, end_idx, color=color, alpha=0.3)
    
    # Create directory for saving plots
    os.makedirs('plots/group_attention_score', exist_ok=True)
    
    # First plot: Token Selection Ratio
    plt.figure(figsize=(15, 8))
    add_section_backgrounds(plt.gca())
    plt.plot(avg_h2o_ratio, color='#FF0000', label='H2O(0.5,0.5)', linewidth=2.5)  # Bright Red
    plt.plot(avg_a2sf_ratio, color='#0000FF', label='A2SF', linewidth=2.5)         # Bright Blue
    
    # Add vertical lines for local windows
    plt.axvline(x=h2o_window, color='#FF0000', linestyle='--', alpha=0.7)
    plt.text(h2o_window, plt.gca().get_ylim()[1], 'H2O\nWindow', rotation=90, va='top', ha='center', color='#FF0000', fontsize=20)
    plt.axvline(x=a2sf_window, color='#0000FF', linestyle='--', alpha=0.7)
    plt.text(a2sf_window, plt.gca().get_ylim()[1], 'A2SF\nWindow', rotation=90, va='top', ha='center', color='#0000FF', fontsize=20)
    
    plt.title('Average Token Selection Ratio Across All Layers and Heads', fontsize=24)
    plt.xlabel('Token Position', fontsize=22)
    plt.ylabel('Ratio of Selected Tokens', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=20, loc='upper center')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('plots/group_attention_score/token_selection_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Second plot: Attention Scores
    plt.figure(figsize=(15, 8))
    add_section_backgrounds(plt.gca())
    plt.plot(avg_original_raw, color='#000000', label='Original', linewidth=2.5, linestyle='--')  # Black
    plt.plot(avg_h2o_raw, color='#FF0000', label='H2O(0.5,0.5)', linewidth=2.5)                  # Bright Red
    plt.plot(avg_a2sf_raw, color='#0000FF', label='A2SF', linewidth=2.5)                         # Bright Blue
    
    # Add vertical lines for local windows
    plt.axvline(x=h2o_window, color='#FF0000', linestyle='--', alpha=0.7)
    plt.text(h2o_window, plt.gca().get_ylim()[1], 'H2O\nWindow', rotation=90, va='top', ha='center', color='#FF0000', fontsize=20)
    plt.axvline(x=a2sf_window, color='#0000FF', linestyle='--', alpha=0.7)
    plt.text(a2sf_window, plt.gca().get_ylim()[1], 'A2SF\nWindow', rotation=90, va='top', ha='center', color='#0000FF', fontsize=20)
    
    plt.title('Average Attention Scores Across All Layers and Heads', fontsize=24)
    plt.xlabel('Token Position', fontsize=22)
    plt.ylabel('Attention Score', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=20, loc='upper center')
    plt.yscale('log')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('plots/group_attention_score/attention_scores.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_attention_scores(original_ratio_scores, h2o_ratio_scores, a2sf_ratio_scores, h2o_small_ratio_scores, h2o_large_ratio_scores,
                                 original_raw_scores, h2o_raw_scores, a2sf_raw_scores, h2o_small_raw_scores, h2o_large_raw_scores):
    # First plot the average attention scores
    plot_average_attention_scores(original_ratio_scores, h2o_ratio_scores, a2sf_ratio_scores, h2o_small_ratio_scores, h2o_large_ratio_scores,
                                original_raw_scores, h2o_raw_scores, a2sf_raw_scores, h2o_small_raw_scores, h2o_large_raw_scores)
    
#     num_layers = original_ratio_scores.shape[0]
#     num_heads = original_ratio_scores.shape[1]
    
#     # 디렉토리 생성
#     os.makedirs('plots/group_attention_score', exist_ok=True)
    
#     # 각 레이어와 헤드별로 그래프 생성
#     for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
#         layer_dir = f'plots/group_attention_score/layer_{layer_idx}'
#         os.makedirs(layer_dir, exist_ok=True)

#         # h2o_window = ((PROMPT_LENGTH + GENERATION_LENGTH) - int(TOTAL_BUDGET * h2o_configs.compression_ratio[layer_idx][0]))//5
#         # a2sf_window = ((PROMPT_LENGTH + GENERATION_LENGTH) - int(TOTAL_BUDGET * a2sf_configs.compression_ratio[layer_idx][0]))//5
#         # h2o_small_window = ((PROMPT_LENGTH + GENERATION_LENGTH) - int(TOTAL_BUDGET * h2o_small_configs.compression_ratio[layer_idx][0]))//5
#         # h2o_large_window = ((PROMPT_LENGTH + GENERATION_LENGTH) - int(TOTAL_BUDGET * h2o_large_configs.compression_ratio[layer_idx][0]))//5

#         for head_idx in tqdm(range(num_heads), desc=f"Processing heads for layer {layer_idx}", leave=False):
#             # 2x2 서브플롯을 가진 그래프 생성
#             fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
#             # 첫 번째 서브플롯: H2O variants ratio comparison
#             ax1.plot(h2o_ratio_scores[layer_idx, head_idx], color='red', label='H2O(0.5,0.5)', linewidth=2)
#             ax1.plot(h2o_small_ratio_scores[layer_idx, head_idx], color='green', label='H2O(0.25,0.75)', linewidth=2)
#             ax1.plot(h2o_large_ratio_scores[layer_idx, head_idx], color='blue', label='H2O(0.75,0.25)', linewidth=2)
            
#             ax1.set_title(f'Layer {layer_idx}, Head {head_idx} - H2O Variants Token Selection Ratio')
#             ax1.set_xlabel(f'Token Position')
#             ax1.set_ylabel('Ratio of Selected Tokens')
#             ax1.grid(True, linestyle='--', alpha=0.7)
#             ax1.legend()

#             # Add vertical lines for H2O variants windows
#             # ax1.axvline(x=h2o_window, color='red', linestyle='--', alpha=0.5)
#             # ax1.text(h2o_window, ax1.get_ylim()[1], 'H2O\nWindow', rotation=90, va='top', ha='center', color='red')
#             # ax1.axvline(x=h2o_small_window, color='green', linestyle='--', alpha=0.5)
#             # ax1.text(h2o_small_window, ax1.get_ylim()[1], 'H2O Small\nWindow', rotation=90, va='top', ha='center', color='green')
#             # ax1.axvline(x=h2o_large_window, color='blue', linestyle='--', alpha=0.5)
#             # ax1.text(h2o_large_window, ax1.get_ylim()[1], 'H2O Large\nWindow', rotation=90, va='top', ha='center', color='blue')
            
#             # 두 번째 서브플롯: Original + H2O variants raw scores
#             ax2.plot(original_raw_scores[layer_idx, head_idx], color='black', label='Original', linewidth=2, linestyle='--')
#             ax2.plot(h2o_raw_scores[layer_idx, head_idx], color='red', label='H2O(0.5,0.5)', linewidth=2)
#             ax2.plot(h2o_small_raw_scores[layer_idx, head_idx], color='green', label='H2O(0.25,0.75)', linewidth=2)
#             ax2.plot(h2o_large_raw_scores[layer_idx, head_idx], color='blue', label='H2O(0.75,0.25)', linewidth=2)
            
#             ax2.set_title(f'Layer {layer_idx}, Head {head_idx} - Original + H2O Variants Attention Scores')
#             ax2.set_xlabel(f'Token Position')
#             ax2.set_ylabel('Attention Score')
#             ax2.grid(True, linestyle='--', alpha=0.7)
#             ax2.legend()
#             ax2.set_yscale('log')

#             # Add vertical lines for H2O variants windows
#             # ax2.axvline(x=h2o_window, color='red', linestyle='--', alpha=0.5)
#             # ax2.text(h2o_window, ax2.get_ylim()[1], 'H2O\nWindow', rotation=90, va='top', ha='center', color='red')
#             # ax2.axvline(x=h2o_small_window, color='green', linestyle='--', alpha=0.5)
#             # ax2.text(h2o_small_window, ax2.get_ylim()[1], 'H2O Small\nWindow', rotation=90, va='top', ha='center', color='green')
#             # ax2.axvline(x=h2o_large_window, color='blue', linestyle='--', alpha=0.5)
#             # ax2.text(h2o_large_window, ax2.get_ylim()[1], 'H2O Large\nWindow', rotation=90, va='top', ha='center', color='blue')
            
#             # 세 번째 서브플롯: H2O vs A2SF ratio comparison
#             ax3.plot(h2o_ratio_scores[layer_idx, head_idx], color='red', label='H2O', linewidth=2)
#             ax3.plot(a2sf_ratio_scores[layer_idx, head_idx], color='orange', label='A2SF', linewidth=2)
            
#             ax3.set_title(f'Layer {layer_idx}, Head {head_idx} - H2O vs A2SF Token Selection Ratio')
#             ax3.set_xlabel(f'Token Position')
#             ax3.set_ylabel('Ratio of Selected Tokens')
#             ax3.grid(True, linestyle='--', alpha=0.7)
#             ax3.legend()

#             # Add vertical lines for H2O and A2SF windows
#             # ax3.axvline(x=h2o_window, color='red', linestyle='--', alpha=0.5)
#             # ax3.text(h2o_window, ax3.get_ylim()[1], 'H2O\nWindow', rotation=90, va='top', ha='center', color='red')
#             # ax3.axvline(x=a2sf_window, color='orange', linestyle='--', alpha=0.5)
#             # ax3.text(a2sf_window, ax3.get_ylim()[1], 'A2SF\nWindow', rotation=90, va='top', ha='center', color='orange')
            
#             # 네 번째 서브플롯: Original + H2O + A2SF raw scores
#             ax4.plot(original_raw_scores[layer_idx, head_idx], color='black', label='Original', linewidth=2, linestyle='--')
#             ax4.plot(h2o_raw_scores[layer_idx, head_idx], color='red', label='H2O', linewidth=2)
#             ax4.plot(a2sf_raw_scores[layer_idx, head_idx], color='orange', label='A2SF', linewidth=2)
            
#             ax4.set_title(f'Layer {layer_idx}, Head {head_idx} - Original + H2O + A2SF Attention Scores')
#             ax4.set_xlabel(f'Token Position')
#             ax4.set_ylabel('Attention Score')
#             ax4.grid(True, linestyle='--', alpha=0.7)
#             ax4.legend()
#             ax4.set_yscale('log')

#             # Add vertical lines for H2O and A2SF windows
#             # ax4.axvline(x=h2o_window, color='red', linestyle='--', alpha=0.5)
#             # ax4.text(h2o_window, ax4.get_ylim()[1], 'H2O\nWindow', rotation=90, va='top', ha='center', color='red')
#             # ax4.axvline(x=a2sf_window, color='orange', linestyle='--', alpha=0.5)
#             # ax4.text(a2sf_window, ax4.get_ylim()[1], 'A2SF\nWindow', rotation=90, va='top', ha='center', color='orange')
            
#             plt.tight_layout()
            
#             # 파일 저장
#             plt.savefig(f'{layer_dir}/head_{head_idx}.png', dpi=300, bbox_inches='tight')
#             plt.close()

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--total_budget", type=int, default=100)
parser.add_argument("--num_prompt", type=int, default=100)
parser.add_argument("--dataset", type=str, default="datasets/cnn_dailymail-0shot.jsonl")

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(torch.float16).to(device)

input_ids, answers, output_indices = load_datasets(args.dataset, tokenizer)

input_ids_buffer = []
attention_map_buffer = []
prompt_lengths = []
generation_lengths = []

# 각 프롬프트에 대해 처리
for inputs, outputs in tqdm(zip(input_ids[:10], output_indices[:10]), desc="Processing prompts..."):
    prompt_length = inputs.shape[1]
    
    inputs = inputs.to(device)
    
    with torch.inference_mode():
        outputs = model.generate(inputs, max_new_tokens=outputs.shape[1], eos_token_id=tokenizer.eos_token_id, do_sample=False)
        generation_length = outputs.shape[1] - prompt_length
        outputs = model(outputs, output_attentions=True)
    
    attention_maps = torch.cat(outputs.attentions, dim=0)
    
    input_ids_buffer.append(inputs)
    prompt_lengths.append(prompt_length)
    generation_lengths.append(generation_length)
    attention_map_buffer.append(attention_maps.cpu())
    
    del outputs, attention_maps
    torch.cuda.empty_cache()

del model
torch.cuda.empty_cache()

# 결과물 준비
all_original_ratio_scores = []
all_h2o_ratio_scores = []
all_a2sf_ratio_scores = []
all_h2o_small_ratio_scores = []
all_h2o_large_ratio_scores = []
all_original_raw_scores = []
all_h2o_raw_scores = []
all_a2sf_raw_scores = []
all_h2o_small_raw_scores = []
all_h2o_large_raw_scores = []

h2o_configs = load_configs("llama2", "h2o", args.total_budget)
a2sf_configs = load_configs("llama2", "a2sf", args.total_budget)
h2o_small_configs = load_configs("llama2", "h2o_2", args.total_budget)
h2o_large_configs = load_configs("llama2", "h2o_8", args.total_budget)

for prompt_idx in tqdm(range(len(prompt_lengths)), desc="Masking attention maps..."):
    prompt_length = prompt_lengths[prompt_idx]
    generation_length = generation_lengths[prompt_idx]
    
    attention_maps = (attention_map_buffer[prompt_idx].to(device))
    
    # 마스크와 점수 계산
    optimal_map = make_optimal_mask(attention_maps, prompt_length, args.total_budget)
    h2o_map = make_a2sf_mask(attention_maps, prompt_length, args.total_budget, h2o_configs.compression_ratio, h2o_configs.forgetting_factors)
    a2sf_map = make_a2sf_mask(attention_maps, prompt_length, args.total_budget, a2sf_configs.compression_ratio, a2sf_configs.forgetting_factors, method="a2sf", input_ids=input_ids_buffer[prompt_idx], punctuation_ids=[tokenizer.encode(".", add_special_tokens=False)[0], tokenizer.encode(" .", add_special_tokens=False)[0]])
    h2o_small_map = make_a2sf_mask(attention_maps, prompt_length, args.total_budget, h2o_small_configs.compression_ratio, h2o_small_configs.forgetting_factors)
    h2o_large_map = make_a2sf_mask(attention_maps, prompt_length, args.total_budget, h2o_large_configs.compression_ratio, h2o_large_configs.forgetting_factors)

    # Ratio scores (binary masks)
    original_ratio = (attention_maps > 0.0).to(torch.float16)
    optimal_ratio = (optimal_map > 0.0).to(torch.float16)
    h2o_ratio = (h2o_map > 0.0).to(torch.float16)
    a2sf_ratio = (a2sf_map > 0.0).to(torch.float16)
    h2o_small_ratio = (h2o_small_map > 0.0).to(torch.float16)
    h2o_large_ratio = (h2o_large_map > 0.0).to(torch.float16)

    # Raw attention scores
    original_raw = attention_maps.to(torch.float16)
    optimal_raw = optimal_map.to(torch.float16)
    h2o_raw = h2o_map.to(torch.float16)
    a2sf_raw = a2sf_map.to(torch.float16)
    h2o_small_raw = h2o_small_map.to(torch.float16)
    h2o_large_raw = h2o_large_map.to(torch.float16)

    for i in range(prompt_length, prompt_length+generation_length):
        # Calculate ratio scores using proportional grouping
        ratio_scores = proportional_grouping_mean(original_ratio[:,:,i,:], i+1).cpu()
        optimal_ratio_scores = proportional_grouping_mean(optimal_ratio[:,:,i,:], i+1).cpu()
        h2o_ratio_scores = proportional_grouping_mean(h2o_ratio[:,:,i,:], i+1).cpu()
        a2sf_ratio_scores = proportional_grouping_mean(a2sf_ratio[:,:,i,:], i+1).cpu()
        h2o_small_ratio_scores = proportional_grouping_mean(h2o_small_ratio[:,:,i,:], i+1).cpu()
        h2o_large_ratio_scores = proportional_grouping_mean(h2o_large_ratio[:,:,i,:], i+1).cpu()

        # Calculate raw attention scores using proportional grouping
        raw_scores = proportional_grouping_sum(original_raw[:,:,i,:], i+1).cpu()
        optimal_raw_scores = proportional_grouping_sum(optimal_raw[:,:,i,:], i+1).cpu()
        h2o_raw_scores = proportional_grouping_sum(h2o_raw[:,:,i,:], i+1).cpu()
        a2sf_raw_scores = proportional_grouping_sum(a2sf_raw[:,:,i,:], i+1).cpu()
        h2o_small_raw_scores = proportional_grouping_sum(h2o_small_raw[:,:,i,:], i+1).cpu()
        h2o_large_raw_scores = proportional_grouping_sum(h2o_large_raw[:,:,i,:], i+1).cpu()

        all_original_ratio_scores.append(ratio_scores)
        all_h2o_ratio_scores.append(h2o_ratio_scores)
        all_a2sf_ratio_scores.append(a2sf_ratio_scores)
        all_h2o_small_ratio_scores.append(h2o_small_ratio_scores)
        all_h2o_large_ratio_scores.append(h2o_large_ratio_scores)

        all_original_raw_scores.append(raw_scores)
        all_h2o_raw_scores.append(h2o_raw_scores)
        all_a2sf_raw_scores.append(a2sf_raw_scores)
        all_h2o_small_raw_scores.append(h2o_small_raw_scores)
        all_h2o_large_raw_scores.append(h2o_large_raw_scores)

    del attention_maps, optimal_map, h2o_map, a2sf_map, h2o_small_map, h2o_large_map
    del original_ratio, optimal_ratio, h2o_ratio, a2sf_ratio, h2o_small_ratio, h2o_large_ratio
    del original_raw, optimal_raw, h2o_raw, a2sf_raw, h2o_small_raw, h2o_large_raw
    del ratio_scores, optimal_ratio_scores, h2o_ratio_scores, a2sf_ratio_scores, h2o_small_ratio_scores, h2o_large_ratio_scores
    del raw_scores, optimal_raw_scores, h2o_raw_scores, a2sf_raw_scores, h2o_small_raw_scores, h2o_large_raw_scores
    torch.cuda.empty_cache()

all_original_ratio_scores = torch.stack(all_original_ratio_scores).mean(dim=0) + 1e-5
all_h2o_ratio_scores = torch.stack(all_h2o_ratio_scores).mean(dim=0) + 1e-5
all_a2sf_ratio_scores = torch.stack(all_a2sf_ratio_scores).mean(dim=0) + 1e-5
all_h2o_small_ratio_scores = torch.stack(all_h2o_small_ratio_scores).mean(dim=0) + 1e-5
all_h2o_large_ratio_scores = torch.stack(all_h2o_large_ratio_scores).mean(dim=0) + 1e-5

all_original_raw_scores = torch.stack(all_original_raw_scores).mean(dim=0) + 1e-5
all_h2o_raw_scores = torch.stack(all_h2o_raw_scores).mean(dim=0) + 1e-5
all_a2sf_raw_scores = torch.stack(all_a2sf_raw_scores).mean(dim=0) + 1e-5
all_h2o_small_raw_scores = torch.stack(all_h2o_small_raw_scores).mean(dim=0) + 1e-5
all_h2o_large_raw_scores = torch.stack(all_h2o_large_raw_scores).mean(dim=0) + 1e-5

# 모든 프롬프트의 결과를 사용하여 그래프 생성
plot_combined_attention_scores(all_original_ratio_scores, all_h2o_ratio_scores, all_a2sf_ratio_scores, 
                             all_h2o_small_ratio_scores, all_h2o_large_ratio_scores,
                             all_original_raw_scores, all_h2o_raw_scores, all_a2sf_raw_scores,
                             all_h2o_small_raw_scores, all_h2o_large_raw_scores)