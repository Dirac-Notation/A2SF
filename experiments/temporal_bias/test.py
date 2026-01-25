import torch
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rcParams
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# utils 모듈이 없다면 이 부분을 주석 처리하고 CompressionConfig 등을 직접 정의해야 할 수 있습니다.
# from utils import load_model, CompressionConfig 

workpath = os.path.dirname(os.path.abspath(__file__))


def layer_wise_analysis(answer_indices, window_indices):
    num_layers = answer_indices.size(0)
    num_heads = answer_indices.size(1)
    
    similarities = []
    for layer in range(num_layers):
        jaccard_similarity = 0
        for head in range(num_heads):
            answer_set = set(answer_indices[layer, head].tolist())
            window_set = set(window_indices[layer, head].tolist())
            intersection = answer_set & window_set
            union = answer_set | window_set
            jaccard_similarity += len(intersection) / len(union) if len(union) > 0 else 0
        similarities.append(jaccard_similarity / num_heads)
    return similarities


def analyze_version1(prefill_attention_maps, answer_indices, window_steps, block_size, token_budget):
    """
    Version 1: Sum of all previous blocks
    Returns: data_to_plot (Windows, Layers)
    """
    window_sim = []
    
    for window in window_steps:
        # Lookback window calculation - Version 1: Sum of all previous blocks
        window_score = prefill_attention_maps[:,:,-window:,:].sum(dim=2)
        window_indices = window_score.topk(token_budget, dim=2).indices
        window_sim.append(layer_wise_analysis(answer_indices, window_indices))

    # Convert to numpy: (Windows, Layers)
    return np.array(window_sim)


def analyze_version2(prefill_attention_maps, answer_indices, window_steps, block_size, token_budget):
    """
    Version 2: Each block separately
    Returns: data_to_plot (Windows, Layers)
    """
    window_sim = []
    
    for window in window_steps:
        # Lookback window calculation - Version 2: Each block separately
        if window == block_size:
            window_score = prefill_attention_maps[:,:,-window:,:].sum(dim=2)
        else:
            window_score = prefill_attention_maps[:,:,-window:-window+block_size,:].sum(dim=2)
        window_indices = window_score.topk(token_budget, dim=2).indices
        window_sim.append(layer_wise_analysis(answer_indices, window_indices))

    # Convert to numpy: (Windows, Layers)
    return np.array(window_sim)


# ---------------------------------------------------------
# 1. Global Style Settings
# ---------------------------------------------------------
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "figure.figsize": (16, 9),
    "figure.dpi": 200,
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 22,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.3
})

# ---------------------------------------------------------
# 2. 데이터 준비 및 프롬프트 구성
# ---------------------------------------------------------
model_name = "llama3"
# config/model2path.json 파일이 존재해야 합니다.
model2path = json.load(open("config/model2path.json", "r"))
model_path = model2path[model_name]

tokenizer = AutoTokenizer.from_pretrained(model_path)
attention_model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    attn_implementation="eager"
).eval()

# prefill_attention_list를 저장할 리스트
prefill_attention_list = []

def get_attn_hook(module, input, output):
    attention_map = output[1]
    if attention_map is not None and attention_map.size(2) != 1:
        prefill_attention_list.append(attention_map.detach().to("cpu"))
        return (output[0], None, output[2])
    return output

# 모든 레이어에 hook 등록
for layer in attention_model.model.layers:
    layer.self_attn.register_forward_hook(get_attn_hook)

prompt_path = os.path.join(workpath, "prompt.txt")
prompts = open(prompt_path, "r").readlines()

for idx, prompt in enumerate(prompts):
    print(f">>> Processing prompt {idx+1} of {len(prompts)}")
    prompt_with_format = f"[INST]{prompt}[/INST]"

    input_enc = tokenizer(prompt_with_format, return_tensors="pt", return_offsets_mapping=True)

    input_ids = input_enc.input_ids.to(attention_model.device)
    attention_mask = input_enc.attention_mask.to(attention_model.device)
    
    if input_ids.size(1) > 8192:
        input_ids = torch.cat([input_ids[:, :3750], input_ids[:, -3750:]], dim=1)
        attention_mask = torch.cat([attention_mask[:, :3750], attention_mask[:, -3750:]], dim=1)

    seq_len = input_ids.size(1)
    max_new_tokens = 513
    token_budget = int(0.1*seq_len) 

    prefill_attention_list.clear()

    print(">>> Generating tokens...")
    with torch.no_grad():
        output = attention_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_attentions=True
        )
    
    # Extract attention weights
    prefill_attention_maps = torch.stack(prefill_attention_list, dim=0).squeeze(1)
    decoding_attention_maps = [torch.stack(output.attentions[i], dim=0).squeeze(1).to("cpu") for i in range(1, len(output.attentions))]

    generation_chunk_size = 64

    for generated_token_length in range(generation_chunk_size, max_new_tokens, generation_chunk_size):
        first_decoding_attention_map = decoding_attention_maps[0]
        answer_score = torch.zeros(
            (*first_decoding_attention_map.shape[:3], seq_len),
            dtype=first_decoding_attention_map.dtype,
            device=first_decoding_attention_map.device
        )

        for atmaps in decoding_attention_maps[generated_token_length-generation_chunk_size:generated_token_length]:
            answer_score += atmaps[:,:,:,:seq_len]
            
        answer_score.squeeze_(dim=2)
        answer_indices = answer_score.topk(token_budget, dim=2).indices
        
        block_size = 16
        window_steps = list(range(block_size, 17*block_size, block_size))
        
        # 두 버전의 데이터 계산
        data_v1 = analyze_version1(prefill_attention_maps, answer_indices, window_steps, block_size, token_budget)
        data_v2 = analyze_version2(prefill_attention_maps, answer_indices, window_steps, block_size, token_budget)
        
        # Subplot visualization with both versions
        print(f">>> Plotting both versions for {generated_token_length} tokens")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
        
        # Get number of layers
        num_layers = data_v1.shape[1]
        
        # Version 2: Left subplot (each block)
        colors_v2 = plt.cm.tab20(np.linspace(0, 1, num_layers))
        for layer_idx in range(num_layers):
            ax1.plot(window_steps, data_v2[:, layer_idx], 
                    alpha=0.9, linewidth=2.5, linestyle='--',
                    color=colors_v2[layer_idx])
        ax1.set_title(f"Hit rate of each block (All Layers)", pad=20, fontweight='bold')
        ax1.set_xlabel("Block offset from end", labelpad=15, fontweight='bold')
        ax1.set_ylabel("Jaccard Similarity", labelpad=15, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
        ax1.set_ylim(0, 0.8)
        
        # Version 1: Right subplot (sum of previous blocks)
        colors_v1 = plt.cm.tab20(np.linspace(0.5, 1, num_layers))
        for layer_idx in range(num_layers):
            ax2.plot(window_steps, data_v1[:, layer_idx], 
                    alpha=0.9, linewidth=2.5, linestyle='-',
                    color=colors_v1[layer_idx])
        ax2.set_title(f"Hit rate of the sum of previous every blocks (All Layers)", pad=20, fontweight='bold')
        ax2.set_xlabel("Window size (Number of queries)", labelpad=15, fontweight='bold')
        ax2.set_ylabel("Jaccard Similarity", labelpad=15, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
        ax2.set_ylim(0, 0.8)
        ax2.yaxis.set_ticklabels([])  # Remove y-axis labels on right subplot to avoid overlap
        
        # Overall figure title
        fig.suptitle(f"Temporal Bias Analysis - {generated_token_length} tokens", fontsize=24, fontweight='bold', y=1.02)
        
        # 저장
        os.makedirs(os.path.join(workpath, f"plots/{idx}"), exist_ok=True)
        plt.tight_layout()
        save_path = os.path.join(workpath, f"plots/{idx}/temporal_bias_line_{generated_token_length}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close()

    print(">>> All analysis and visualization completed")