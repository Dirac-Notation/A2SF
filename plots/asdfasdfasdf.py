import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import get_prompt, make_optimal_mask, make_a2sf_mask, load_configs, generate

def diag(attn_map: torch.Tensor):
    return torch.stack([attn_map.diagonal(offset=i, dim1=2, dim2=3).mean(dim=2) for i in range(-(GENERATION_LENGTH-1), PROMPT_LENGTH+1)], dim=-1)

def plot_combined_attention_scores(original_scores, h2o_scores, a2sf_scores, h2o_small_scores, h2o_large_scores):
    num_layers = original_scores.shape[1]
    num_heads = original_scores.shape[2]
    
    # 디렉토리 생성
    os.makedirs('plots/group_attention_score', exist_ok=True)
    
    # grouped_prompt_length = PROMPT_LENGTH // 5
    
    # 각 레이어와 헤드별로 그래프 생성
    for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
        layer_dir = f'plots/group_attention_score/layer_{layer_idx}'
        os.makedirs(layer_dir, exist_ok=True)

        h2o_window = ((PROMPT_LENGTH + GENERATION_LENGTH) - int(TOTAL_BUDGET * h2o_configs.compression_ratio[layer_idx][0]))//5
        a2sf_window = ((PROMPT_LENGTH + GENERATION_LENGTH) - int(TOTAL_BUDGET * a2sf_configs.compression_ratio[layer_idx][0]))//5
        h2o_small_window = ((PROMPT_LENGTH + GENERATION_LENGTH) - int(TOTAL_BUDGET * h2o_small_configs.compression_ratio[layer_idx][0]))//5
        h2o_large_window = ((PROMPT_LENGTH + GENERATION_LENGTH) - int(TOTAL_BUDGET * h2o_large_configs.compression_ratio[layer_idx][0]))//5

        for head_idx in tqdm(range(num_heads), desc=f"Processing heads for layer {layer_idx}", leave=False):
            # 2x2 서브플롯을 가진 그래프 생성
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 첫 번째 서브플롯: 각 프롬프트의 오리지널들만 표시
            for prompt_idx in range(NUM_PROMPT):
                ax1.plot(original_scores[prompt_idx, layer_idx, head_idx])
            
            ax1.set_title(f'Layer {layer_idx}, Head {head_idx} - Original Scores by Prompt')
            ax1.set_xlabel(f'Token Position')
            ax1.set_ylabel('Attention Score')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_yscale('log')
            
            # 두 번째 서브플롯: 오리지널+옵티멀
            ax2.plot(original_scores.mean(dim=0)[layer_idx, head_idx], color='black', label=f'Original', linewidth=2, linestyle='--')
            ax2.plot(h2o_small_scores.mean(dim=0)[layer_idx, head_idx], color='green', label='H2O(0.25,0.75)', linewidth=2)
            ax2.plot(h2o_scores.mean(dim=0)[layer_idx, head_idx], color='red', label='H2O(0.5,0.5)', linewidth=2)
            ax2.plot(h2o_large_scores.mean(dim=0)[layer_idx, head_idx], color='blue', label='H2O(0.75,0.25)', linewidth=2)
            
            ax2.set_title(f'Layer {layer_idx}, Head {head_idx} - Original + H2O Small + H2O + H2O Large')
            ax2.set_xlabel(f'Token Position')
            ax2.set_ylabel('Attention Score')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()
            ax2.set_yscale('log')
            
            ax2.axvline(x=h2o_small_window, color='green', linestyle='--', alpha=0.5)
            ax2.text(h2o_small_window, ax2.get_ylim()[1], 'H2O Small\nWindow', rotation=90, va='top', ha='center', color='green')
            ax2.axvline(x=h2o_window, color='red', linestyle='--', alpha=0.5)
            ax2.text(h2o_window, ax2.get_ylim()[1], 'H2O\nWindow', rotation=90, va='top', ha='center', color='red')
            ax2.axvline(x=h2o_large_window, color='blue', linestyle='--', alpha=0.5)
            ax2.text(h2o_large_window, ax2.get_ylim()[1], 'H2O Large\nWindow', rotation=90, va='top', ha='center', color='blue')
            
            # 세 번째 서브플롯: 오리지널+H2O(100,100)+A2SF(0.99,100,100)
            ax3.plot(original_scores.mean(dim=0)[layer_idx, head_idx], color='black', label=f'Original', linewidth=2, linestyle='--')
            ax3.plot(h2o_scores.mean(dim=0)[layer_idx, head_idx], color='green', label='H2O', linewidth=2)
            ax3.plot(a2sf_scores.mean(dim=0)[layer_idx, head_idx], color='orange', label='A2SF', linewidth=2)
            
            ax3.set_title(f'Layer {layer_idx}, Head {head_idx} - Comparison')
            ax3.set_xlabel(f'Token Position')
            ax3.set_ylabel('Attention Score')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.legend()
            ax3.set_yscale('log')
            
            # Add vertical lines for boundaries with labels
            ax3.axvline(x=h2o_window, color='blue', linestyle='--', alpha=0.5)
            ax3.text(h2o_window, ax3.get_ylim()[1], 'H2O\nWindow', rotation=90, va='top', ha='center', color='blue')
            ax3.axvline(x=a2sf_window, color='orange', linestyle='--', alpha=0.5)
            ax3.text(a2sf_window, ax3.get_ylim()[1], 'A2SF\nWindow', rotation=90, va='top', ha='center', color='orange')
            
            # 네 번째 서버플롯: (오리지널-H2O) + (오리지널-A2SF)
            ax4.plot(torch.abs(original_scores.mean(dim=0)[layer_idx, head_idx] - h2o_scores.mean(dim=0)[layer_idx, head_idx]), color='blue', label='Original - H2O', linewidth=2)
            ax4.plot(torch.abs(original_scores.mean(dim=0)[layer_idx, head_idx] - a2sf_scores.mean(dim=0)[layer_idx, head_idx]), color='orange', label='Original - A2SF', linewidth=2)
            
            ax4.set_title(f'Layer {layer_idx}, Head {head_idx} - Original - H2O + Original - A2SF')
            ax4.set_xlabel(f'Token Position')
            ax4.set_ylabel('Attention Score')
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.legend()
            ax4.set_yscale('log')
            
            ax4.axvline(x=h2o_window, color='blue', linestyle='--', alpha=0.5)
            ax4.text(h2o_window, ax4.get_ylim()[1], 'H2O\nWindow', rotation=90, va='top', ha='center', color='blue')
            ax4.axvline(x=a2sf_window, color='orange', linestyle='--', alpha=0.5)
            ax4.text(a2sf_window, ax4.get_ylim()[1], 'A2SF\nWindow', rotation=90, va='top', ha='center', color='orange')
            
            plt.tight_layout()
            
            # 파일 저장
            plt.savefig(f'{layer_dir}/head_{head_idx}.png', dpi=300, bbox_inches='tight')
            plt.close()

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--prompt_length", type=int, default=300)
parser.add_argument("--generation_length", type=int, default=200)
parser.add_argument("--total_budget", type=int, default=100)
parser.add_argument("--num_prompt", type=int, default=100)

args = parser.parse_args()

PROMPT_LENGTH = args.prompt_length
GENERATION_LENGTH = args.generation_length
TOTAL_BUDGET = args.total_budget
NUM_PROMPT = args.num_prompt

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(torch.float16).to(device)

attention_map_buffer = []

# 각 프롬프트에 대해 처리
for prompt_idx in tqdm(range(NUM_PROMPT), desc="Processing prompts..."):
    prompt = get_prompt(prompt_idx)
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:,:PROMPT_LENGTH].to(device)
    
    _, attention_maps, _ = generate(model, input_ids, GENERATION_LENGTH)

    attention_map_buffer.append(attention_maps.cpu())

del model, tokenizer
torch.cuda.empty_cache()

# 결과물 준비
all_original_scores = []
all_h2o_scores = []
all_a2sf_scores = []
all_h2o_small_scores = []
all_h2o_large_scores = []

h2o_configs = load_configs("Llama-2-7b-chat-hf", "h2o", TOTAL_BUDGET)
a2sf_configs = load_configs("Llama-2-7b-chat-hf", "a2sf", TOTAL_BUDGET)
h2o_small_configs = load_configs("Llama-2-7b-chat-hf", "h2o_small", TOTAL_BUDGET)
h2o_large_configs = load_configs("Llama-2-7b-chat-hf", "h2o_large", TOTAL_BUDGET)

for prompt_idx in tqdm(range(NUM_PROMPT), desc="Masking attention maps..."):
    attention_maps = attention_map_buffer[prompt_idx].to(device)
    
    # 마스크와 점수 계산
    optimal_map = make_optimal_mask(attention_maps, PROMPT_LENGTH, TOTAL_BUDGET)
    h2o_map = make_a2sf_mask(attention_maps, PROMPT_LENGTH, TOTAL_BUDGET, h2o_configs.compression_ratio, h2o_configs.forgetting_factors)
    a2sf_map = make_a2sf_mask(attention_maps, PROMPT_LENGTH, TOTAL_BUDGET, a2sf_configs.compression_ratio, a2sf_configs.forgetting_factors)   
    h2o_small_map = make_a2sf_mask(attention_maps, PROMPT_LENGTH, TOTAL_BUDGET, h2o_small_configs.compression_ratio, h2o_small_configs.forgetting_factors)
    h2o_large_map = make_a2sf_mask(attention_maps, PROMPT_LENGTH, TOTAL_BUDGET, h2o_large_configs.compression_ratio, h2o_large_configs.forgetting_factors)

    # 정규화
    scores = (diag(attention_maps[:,:,PROMPT_LENGTH:,:]) + 1e-5).cpu()
    optimal_scores = (diag(optimal_map[:,:,PROMPT_LENGTH:,:]) + 1e-5).cpu()
    h2o_scores = (diag(h2o_map[:,:,PROMPT_LENGTH:,:]) + 1e-5).cpu()
    a2sf_scores = (diag(a2sf_map[:,:,PROMPT_LENGTH:,:]) + 1e-5).cpu()
    h2o_small_scores = (diag(h2o_small_map[:,:,PROMPT_LENGTH:,:]) + 1e-5).cpu()
    h2o_large_scores = (diag(h2o_large_map[:,:,PROMPT_LENGTH:,:]) + 1e-5).cpu()

    # 각 프롬프트의 결과를 리스트에 저장
    all_original_scores.append(scores)
    all_h2o_scores.append(h2o_scores)
    all_a2sf_scores.append(a2sf_scores)
    all_h2o_small_scores.append(h2o_small_scores)
    all_h2o_large_scores.append(h2o_large_scores)

    del attention_maps, optimal_map, h2o_map, a2sf_map, h2o_small_map, h2o_large_map, scores, optimal_scores, h2o_scores, a2sf_scores, h2o_small_scores, h2o_large_scores
    torch.cuda.empty_cache()

all_original_scores = torch.stack(all_original_scores)
all_h2o_scores = torch.stack(all_h2o_scores)
all_a2sf_scores = torch.stack(all_a2sf_scores)
all_h2o_small_scores = torch.stack(all_h2o_small_scores)
all_h2o_large_scores = torch.stack(all_h2o_large_scores)

all_original_scores = all_original_scores.view(*all_original_scores.shape[:3],-1,5).mean(dim=-1)
all_h2o_scores = all_h2o_scores.view(*all_h2o_scores.shape[:3],-1,5).mean(dim=-1)
all_a2sf_scores = all_a2sf_scores.view(*all_a2sf_scores.shape[:3],-1,5).mean(dim=-1)
all_h2o_small_scores = all_h2o_small_scores.view(*all_h2o_small_scores.shape[:3],-1,5).mean(dim=-1)
all_h2o_large_scores = all_h2o_large_scores.view(*all_h2o_large_scores.shape[:3],-1,5).mean(dim=-1)

# 모든 프롬프트의 결과를 사용하여 그래프 생성
plot_combined_attention_scores(all_original_scores, all_h2o_scores, all_a2sf_scores, all_h2o_small_scores, all_h2o_large_scores)