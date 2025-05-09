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

def partioning(attention_row, total_length):
    points = [int(i/NUM_GROUP * total_length) for i in range(NUM_GROUP+1)]
    scores = []

    for i in range(NUM_GROUP):
        scores.append(attention_row[:,:,points[i]:points[i+1]].sum(dim=2))
    
    return torch.stack(scores, dim=2)

def plot_combined_attention_scores(original_scores, optimal_scores, h2o_scores, a2sf_scores):
    num_layers = optimal_scores.shape[1]
    num_heads = optimal_scores.shape[2]
    
    # 디렉토리 생성
    os.makedirs('plots/group_attention_score', exist_ok=True)
    
    # Calculate boundaries
    prompt_boundary = int(PROMPT_LENGTH / (PROMPT_LENGTH + GENERATION_LENGTH) * NUM_GROUP)
    
    # 각 레이어와 헤드별로 그래프 생성
    for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
        layer_dir = f'plots/group_attention_score/layer_{layer_idx}'
        os.makedirs(layer_dir, exist_ok=True)

        h2o_window = NUM_GROUP - int(TOTAL_BUDGET * h2o_configs.compression_ratio[layer_idx][0] / (PROMPT_LENGTH + GENERATION_LENGTH) * NUM_GROUP)
        a2sf_window = NUM_GROUP - int(TOTAL_BUDGET * a2sf_configs.compression_ratio[layer_idx][0] / (PROMPT_LENGTH + GENERATION_LENGTH) * NUM_GROUP)

        for head_idx in tqdm(range(num_heads), desc=f"Processing heads for layer {layer_idx}", leave=False):
            # 2x2 서브플롯을 가진 그래프 생성
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 첫 번째 서브플롯: 각 프롬프트의 오리지널들만 표시
            for prompt_idx in range(4):
                color = ['black', 'blue', 'green', 'red'][prompt_idx]
                ax1.plot(original_scores[prompt_idx, layer_idx, head_idx],
                        color=color, label=f'Prompt {prompt_idx}', linewidth=2)
            
            ax1.set_title(f'Layer {layer_idx}, Head {head_idx} - Original Scores by Prompt')
            ax1.set_xlabel(f'Position Group({NUM_GROUP} groups)')
            ax1.set_ylabel('Attention Score')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()
            ax1.set_yscale('log')
            
            # Add vertical lines for boundaries with labels
            ax1.axvline(x=prompt_boundary, color='gray', linestyle='--', alpha=0.5)
            ax1.text(prompt_boundary, ax1.get_ylim()[1], 'Prompt\nBoundary', 
                    rotation=90, va='top', ha='center', color='gray')
            
            # 두 번째 서브플롯: H2O Recent Budget + 오리지널
            ax2.plot(original_scores.mean(dim=0)[layer_idx, head_idx], color='black', label=f'Original', linewidth=2, linestyle='--')
            ax2.plot(h2o_scores.mean(dim=0)[layer_idx, head_idx], color="blue", label=f'H2O', linewidth=2)
            
            ax2.set_title(f'Layer {layer_idx}, Head {head_idx} - H2O Variants')
            ax2.set_xlabel(f'Position Group({NUM_GROUP} groups)')
            ax2.set_ylabel('Attention Score')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()
            ax2.set_yscale('log')
            
            # Add vertical lines for boundaries with labels
            ax2.axvline(x=prompt_boundary, color='gray', linestyle='--', alpha=0.5)
            ax2.text(prompt_boundary, ax2.get_ylim()[1], 'Prompt\nBoundary', 
                    rotation=90, va='top', ha='center', color='gray')
            ax2.axvline(x=h2o_window, color='blue', linestyle='--', alpha=0.5)
            ax2.text(h2o_window, ax2.get_ylim()[1], 'H2O\nWindow', 
                    rotation=90, va='top', ha='center', color='blue')
            
            # 세 번째 서브플롯: A2SF Factor별 + 오리지널
            ax3.plot(original_scores.mean(dim=0)[layer_idx, head_idx], color='black', label=f'Original', linewidth=2, linestyle='--')
            ax3.plot(a2sf_scores.mean(dim=0)[layer_idx, head_idx], color='orange', label=f'A2SF', linewidth=2)
            
            ax3.set_title(f'Layer {layer_idx}, Head {head_idx} - A2SF Variants')
            ax3.set_xlabel(f'Position Group({NUM_GROUP} groups)')
            ax3.set_ylabel('Attention Score')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.legend()
            ax3.set_yscale('log')
            
            # Add vertical lines for boundaries with labels
            ax3.axvline(x=prompt_boundary, color='gray', linestyle='--', alpha=0.5)
            ax3.text(prompt_boundary, ax3.get_ylim()[1], 'Prompt\nBoundary', 
                    rotation=90, va='top', ha='center', color='gray')
            ax3.axvline(x=a2sf_window, color='orange', linestyle='--', alpha=0.5)
            ax3.text(a2sf_window, ax3.get_ylim()[1], 'A2SF\nWindow', 
                    rotation=90, va='top', ha='center', color='orange')
            
            # 네 번째 서브플롯: 오리지널+옵티멀+H2O(100,100)+A2SF(0.99,100,100)
            ax4.plot(original_scores.mean(dim=0)[layer_idx, head_idx], color='black', label=f'Original', linewidth=2, linestyle='--')
            ax4.plot(optimal_scores.mean(dim=0)[layer_idx, head_idx], color='red', label='Optimal', linewidth=2)
            ax4.plot(h2o_scores.mean(dim=0)[layer_idx, head_idx], color='green', label='H2O', linewidth=2)
            ax4.plot(a2sf_scores.mean(dim=0)[layer_idx, head_idx], color='orange', label='A2SF', linewidth=2)
            
            ax4.set_title(f'Layer {layer_idx}, Head {head_idx} - Comparison')
            ax4.set_xlabel(f'Position Group({NUM_GROUP} groups)')
            ax4.set_ylabel('Attention Score')
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.legend()
            ax4.set_yscale('log')
            
            # Add vertical lines for boundaries with labels
            ax4.axvline(x=prompt_boundary, color='gray', linestyle='--', alpha=0.5)
            ax4.text(prompt_boundary, ax4.get_ylim()[1], 'Prompt\nBoundary', 
                    rotation=90, va='top', ha='center', color='gray')
            ax4.axvline(x=h2o_window, color='blue', linestyle='--', alpha=0.5)
            ax4.text(h2o_window, ax4.get_ylim()[1], 'H2O\nWindow', 
                    rotation=90, va='top', ha='center', color='blue')
            ax4.axvline(x=a2sf_window, color='orange', linestyle='--', alpha=0.5)
            ax4.text(a2sf_window, ax4.get_ylim()[1], 'A2SF\nWindow', 
                    rotation=90, va='top', ha='center', color='orange')
            
            plt.tight_layout()
            
            # 파일 저장
            plt.savefig(f'{layer_dir}/head_{head_idx}.png', dpi=300, bbox_inches='tight')
            plt.close()

PROMPT_LENGTH = 500
GENERATION_LENGTH = 500
TOTAL_BUDGET = 100
NUM_GROUP = 100

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(torch.float16).to(device)

attention_map_buffer = []
prompt_length_list = []

# 각 프롬프트에 대해 처리
for prompt_idx in range(4):
    prompt = get_prompt(prompt_idx)
    
    with torch.inference_mode():        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:,:PROMPT_LENGTH].to(device)
        prompt_length_list.append(input_ids.size(1))
        
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
        attention_maps = torch.stack(outputs.attentions).squeeze(1).cpu()

        del outputs, next_token_logits, past_key_values
        torch.cuda.empty_cache()
        
        attention_map_buffer.append(attention_maps)

del model, tokenizer
torch.cuda.empty_cache()

# 결과물 준비
all_original_scores = []
all_optimal_scores = []
all_h2o_scores = []
all_a2sf_scores = []

h2o_configs = load_configs("Llama-2-7b-chat-hf", "h2o", TOTAL_BUDGET)
a2sf_configs = load_configs("Llama-2-7b-chat-hf", "a2sf", TOTAL_BUDGET)

for prompt_idx in range(4):
    attention_maps = attention_map_buffer[prompt_idx].to(device)
    prompt_length = prompt_length_list[prompt_idx]
    
    # 마스크와 점수 계산
    optimal_map = make_optimal_mask(attention_maps, prompt_length, TOTAL_BUDGET)
    
    h2o_map = make_a2sf_mask(attention_maps, prompt_length, TOTAL_BUDGET, h2o_configs.compression_ratio, h2o_configs.forgetting_factors)
    
    a2sf_map = make_a2sf_mask(attention_maps, prompt_length, TOTAL_BUDGET, a2sf_configs.compression_ratio, a2sf_configs.forgetting_factors)   
    
    # 각 프롬프트에 대한 점수 계산
    scores = torch.zeros(attention_maps.shape[0], attention_maps.shape[1], NUM_GROUP, device=device)
    optimal_scores = torch.zeros(attention_maps.shape[0], attention_maps.shape[1], NUM_GROUP, device=device)
    h2o_scores = torch.zeros(attention_maps.shape[0], attention_maps.shape[1], NUM_GROUP, device=device)
    a2sf_scores = torch.zeros(attention_maps.shape[0], attention_maps.shape[1], NUM_GROUP, device=device)
    
    for i in range(prompt_length, prompt_length+GENERATION_LENGTH):
        scores += partioning(attention_maps[:,:,i,:], i+1)
        optimal_scores += partioning(optimal_map[:,:,i,:], i+1)
        
        # H2O 변형 계산
        h2o_scores += partioning(h2o_map[:,:,i,:], i+1)
        
        # A2SF 변형 계산
        a2sf_scores += partioning(a2sf_map[:,:,i,:], i+1)
    
    # 정규화
    scores = (scores/GENERATION_LENGTH).cpu() + 1e-5
    optimal_scores = (optimal_scores/GENERATION_LENGTH).cpu() + 1e-5
    
    # H2O 변형 정규화
    h2o_scores = (h2o_scores/GENERATION_LENGTH).cpu() + 1e-5
    
    # A2SF 변형 정규화
    a2sf_scores = (a2sf_scores/GENERATION_LENGTH).cpu() + 1e-5
    
    # 각 프롬프트의 결과를 리스트에 저장
    all_original_scores.append(scores)
    all_optimal_scores.append(optimal_scores)
    all_h2o_scores.append(h2o_scores)
    all_a2sf_scores.append(a2sf_scores)
    
    del attention_maps, optimal_map, h2o_map, a2sf_map, scores, optimal_scores, h2o_scores, a2sf_scores
    torch.cuda.empty_cache()

all_original_scores = torch.stack(all_original_scores)
all_optimal_scores = torch.stack(all_optimal_scores)
all_h2o_scores = torch.stack(all_h2o_scores)
all_a2sf_scores = torch.stack(all_a2sf_scores)

# 모든 프롬프트의 결과를 사용하여 그래프 생성
plot_combined_attention_scores(all_original_scores, all_optimal_scores, all_h2o_scores, all_a2sf_scores)