import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import get_prompt, make_optimal_mask, make_a2sf_mask

def partioning(attention_row, total_length):
    points = [int(i/NUM_GROUP * total_length) for i in range(NUM_GROUP+1)]
    scores = []

    for i in range(NUM_GROUP):
        # scores.append((attention_row[:,:,points[i]:points[i+1]]>0).sum(dim=2))
        scores.append(attention_row[:,:,points[i]:points[i+1]].sum(dim=2))
    
    return torch.stack(scores, dim=2)

def plot_attention_scores(scores, plot_type="default"):
    num_layers = scores.shape[0]
    num_heads = scores.shape[1]
    
    # 그룹 레이블 (0.1부터 1.0까지)
    group_labels = [f"{i/NUM_GROUP:.2f}" for i in range(1, NUM_GROUP+1)]
    
    # 서브그래프 레이아웃 설정 (8x4 그리드)
    fig, axes = plt.subplots(8, 4, figsize=(20, 30))
    fig.suptitle('Attention Scores by Layer and Head', fontsize=16)
    
    # 각 레이어에 대한 서브그래프
    for layer_idx in range(num_layers):
        row = layer_idx // 4
        col = layer_idx % 4
        
        ax = axes[row, col]
        
        # 각 헤드에 대한 선 그래프
        for head_idx in range(num_heads):
            # 헤드별로 다른 색상 사용 (색상맵 사용)
            color = plt.cm.viridis(head_idx / num_heads)
            ax.plot(group_labels, scores[layer_idx, head_idx].cpu().numpy(), 
                    color=color, alpha=0.7, linewidth=1.5, 
                    label=f'Head {head_idx}' if head_idx % 5 == 0 else None)
        
        ax.set_title(f'Layer {layer_idx}')
        ax.set_xlabel('Position Group')
        ax.set_ylabel('Attention Score')
        ax.grid(True, linestyle='--', alpha=0.7)
        # ax.set_ylim(0, 1.0)
        ax.set_yscale('log')  # y축을 로그 스케일로 변경
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f'plots/attention_scores_by_layer_{plot_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 평균 점수를 보여주는 추가 그래프
    plt.figure(figsize=(12, 8))
    mean_scores = scores.mean(dim=1).cpu().numpy()  # 레이어별, 그룹별 평균
    
    for layer_idx in range(num_layers):
        plt.plot(group_labels, mean_scores[layer_idx], 
                label=f'Layer {layer_idx}', linewidth=2)
    
    plt.title('Average Attention Scores by Layer')
    plt.xlabel('Position Group')
    plt.ylabel('Average Attention Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'plots/average_attention_scores_{plot_type}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_attention_scores(original_scores, optimal_scores, h2o_variants, a2sf_scores):
    num_layers = optimal_scores.shape[0]
    num_heads = optimal_scores.shape[1]
    
    # 그룹 레이블 (0.1부터 1.0까지)
    group_labels = [f"{i/NUM_GROUP:.2f}" for i in range(1, NUM_GROUP+1)]
    
    # 디렉토리 생성
    os.makedirs('plots/group_attention_score', exist_ok=True)
    
    # 각 레이어와 헤드별로 그래프 생성
    for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
        layer_dir = f'plots/group_attention_score/layer_{layer_idx}'
        os.makedirs(layer_dir, exist_ok=True)
        
        for head_idx in tqdm(range(num_heads), desc=f"Processing heads for layer {layer_idx}", leave=False):
            # 두 개의 서브플롯을 가진 그래프 생성
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # 왼쪽 서브플롯: 현재 비교 (original, optimal, 세 가지 H2O 변형)
            ax1.plot(group_labels, original_scores[layer_idx, head_idx].cpu().numpy(), 
                    color='black', label='Original', linewidth=2)
            ax1.plot(group_labels, optimal_scores[layer_idx, head_idx].cpu().numpy(), 
                    color='red', label='Optimal', linewidth=2)
            
            # H2O 변형 추가
            colors = ['blue', 'green', 'purple']
            for i, (recent, select) in enumerate([(50,150), (100,100), (150,50)]):
                ax1.plot(group_labels, h2o_variants[i][layer_idx, head_idx].cpu().numpy(), 
                        color=colors[i], label=f'H2O ({recent},{select})', linewidth=2)
            
            ax1.set_title(f'Layer {layer_idx}, Head {head_idx} - H2O Variants')
            ax1.set_xlabel('Position Group')
            ax1.set_ylabel('Attention Score')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_yscale('log')
            ax1.legend()
            
            # 오른쪽 서브플롯: original, optimal, H2O(100,100), A2SF 비교
            ax2.plot(group_labels, original_scores[layer_idx, head_idx].cpu().numpy(), 
                    color='black', label='Original', linewidth=2)
            ax2.plot(group_labels, optimal_scores[layer_idx, head_idx].cpu().numpy(), 
                    color='red', label='Optimal', linewidth=2)
            ax2.plot(group_labels, h2o_variants[1][layer_idx, head_idx].cpu().numpy(), 
                    color='green', label='H2O (100,100)', linewidth=2)
            ax2.plot(group_labels, a2sf_scores[layer_idx, head_idx].cpu().numpy(), 
                    color='orange', label='A2SF', linewidth=2)
            
            ax2.set_title(f'Layer {layer_idx}, Head {head_idx} - Comparison')
            ax2.set_xlabel('Position Group')
            ax2.set_ylabel('Attention Score')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.set_yscale('log')
            ax2.legend()
            
            plt.tight_layout()
            
            # 파일 저장
            plt.savefig(f'{layer_dir}/head_{head_idx}.png', dpi=300, bbox_inches='tight')
            plt.close()

GENERATION_LENGTH = 200
RECENT_BUDGET = 50
SELECT_BUDGET = 50
NUM_GROUP = 10

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(torch.float16).to(device)

prompt = get_prompt()

with torch.inference_mode():
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:,:500].to(device)
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
    attention_maps = torch.stack(outputs.attentions).squeeze(1)#.to(torch.float32)
    
    del model, tokenizer, outputs, past_key_values
    torch.cuda.empty_cache()

optimal_map = make_optimal_mask(attention_maps, 500, RECENT_BUDGET+SELECT_BUDGET)
a2sf_map = make_a2sf_mask(attention_maps, 500, RECENT_BUDGET, SELECT_BUDGET, forgetting_factor=0.995)

# H2O 변형 추가
h2o_variant1_map = make_a2sf_mask(attention_maps, 500, int(RECENT_BUDGET*0.5), int(SELECT_BUDGET*1.5), forgetting_factor=1.00)
h2o_variant2_map = make_a2sf_mask(attention_maps, 500, RECENT_BUDGET, SELECT_BUDGET, forgetting_factor=1.00)
h2o_variant3_map = make_a2sf_mask(attention_maps, 500, int(RECENT_BUDGET*1.5), int(SELECT_BUDGET*0.5), forgetting_factor=1.00)

scores = torch.zeros(attention_maps.shape[0], attention_maps.shape[1], NUM_GROUP, device=device)
optimal_scores = torch.zeros(attention_maps.shape[0], attention_maps.shape[1], NUM_GROUP, device=device)
a2sf_scores = torch.zeros(attention_maps.shape[0], attention_maps.shape[1], NUM_GROUP, device=device)

# H2O 변형을 위한 텐서 추가
h2o_variants = [
    torch.zeros(attention_maps.shape[0], attention_maps.shape[1], NUM_GROUP, device=device),
    torch.zeros(attention_maps.shape[0], attention_maps.shape[1], NUM_GROUP, device=device),
    torch.zeros(attention_maps.shape[0], attention_maps.shape[1], NUM_GROUP, device=device)
]

for i in range(500, 500+GENERATION_LENGTH):
    scores += partioning(attention_maps[:,:,i,:], i+1)
    optimal_scores += partioning(optimal_map[:,:,i,:], i+1)
    a2sf_scores += partioning(a2sf_map[:,:,i,:], i+1)
    
    # H2O 변형 계산
    h2o_variants[0] += partioning(h2o_variant1_map[:,:,i,:], i+1)
    h2o_variants[1] += partioning(h2o_variant2_map[:,:,i,:], i+1)
    h2o_variants[2] += partioning(h2o_variant3_map[:,:,i,:], i+1)

scores /= GENERATION_LENGTH
optimal_scores /= GENERATION_LENGTH
a2sf_scores /= GENERATION_LENGTH

# H2O 변형 정규화
for i in range(3):
    h2o_variants[i] /= GENERATION_LENGTH

# 새로운 방식의 그래프 생성
plot_combined_attention_scores(scores, optimal_scores, h2o_variants, a2sf_scores)