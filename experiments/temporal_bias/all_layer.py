import torch
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import glob

from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

workpath = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def plot_group_results(group_name, group_data, avg_prefill, avg_gen, workpath, sentences_info=None):
    """
    Plot temporal bias analysis results for a group
    Args:
        group_name: Name of the group
        group_data: Dictionary with item indices as keys and (data_v1_mean, data_v2_mean, seq_len, gen_len) as values
        avg_prefill: Average prefill length
        avg_gen: Average generation length
        workpath: Path to save the plot
        sentences_info: Dictionary with item indices as keys and sentence information dicts as values
    """
    print(f">>> Plotting group: {group_name} (Avg Prefill: {avg_prefill:.1f}, Avg Gen: {avg_gen:.1f})")
    
    # Apply style settings explicitly
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 22,
        "axes.linewidth": 1.2,
    })
    
    plt.rcParams.update({"legend.fontsize": 18})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    num_items = len([k for k in group_data.keys() if isinstance(k, int)])
    block_size = 8
    window_steps = list(range(block_size, 33*block_size, block_size))
    
    # Get colors for different items
    colors = plt.cm.tab20(np.linspace(0, 1, num_items))
    
    # Version 2: Left subplot (each block)
    item_indices = sorted([k for k in group_data.keys() if isinstance(k, int)])
    for item_idx, color in zip(item_indices, colors):
        data_v2_mean = group_data[item_idx][1]
        ax1.plot(window_steps, data_v2_mean, 
                alpha=0.9, linewidth=2.5, linestyle='--',
                color=color)
    ax1.set_title(f"Hit rate of each query", fontsize=22)
    ax1.set_xlabel("Query Index", fontsize=22)
    ax1.set_ylabel("Hit rate", fontsize=22)
    ax1.tick_params(labelsize=22)
    ax1.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    # ax1.set_ylim(0, 0.8)
    ax1.text(0.98, 0.98, f"Avg Prefill: {avg_prefill:.1f}\nAvg Gen: {avg_gen:.1f}", 
             transform=ax1.transAxes, fontsize=16, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Version 1: Right subplot (sum of previous blocks)
    for item_idx, color in zip(item_indices, colors):
        data_v1_mean = group_data[item_idx][0]
        ax2.plot(window_steps, data_v1_mean, 
                alpha=0.9, linewidth=2.5, linestyle='-',
                color=color)
    ax2.set_title(f"Hit rate of the sum of previous queries", fontsize=22)
    ax2.set_xlabel("The number of accumulated queries", fontsize=22)
    ax2.set_ylabel("Hit rate", fontsize=22)
    ax2.tick_params(labelsize=22)
    ax2.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    # ax2.set_ylim(0, 0.8)
    ax2.text(0.98, 0.98, f"Avg Prefill: {avg_prefill:.1f}\nAvg Gen: {avg_gen:.1f}", 
             transform=ax2.transAxes, fontsize=16, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 저장
    os.makedirs(os.path.join(workpath, "plots"), exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Overall figure title (after tight_layout to position correctly)
    fig.suptitle(f"{group_name}", fontsize=24, fontweight='bold', y=0.95)
    save_path = os.path.join(workpath, f"plots/temporal_bias_{group_name.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()

# ---------------------------------------------------------
# 2. 데이터 준비 및 프롬프트 구성
# ---------------------------------------------------------
# Load max generation lengths
dataset2maxlen_path = os.path.join(root_path, "config", "dataset2maxlen.json")
dataset2maxlen = json.load(open(dataset2maxlen_path, "r"))

# Load data from data.jsonl
data_jsonl_path = os.path.join(workpath, "data.jsonl")
selected_data = {}  # {group_name: [selected_items]}

with open(data_jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        group_name = item.get("group_name")
        if group_name not in selected_data:
            selected_data[group_name] = []
        selected_data[group_name].append((item.get("dataset"), item))

model_name = "llama3"
model2path = json.load(open(os.path.join(root_path, "config", "model2path.json"), "r"))
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

# Process each group
for group_name, selected_items in selected_data.items():
    print(f"\n>>> Processing group: {group_name} ({len(selected_items)} items)")
    
    # Store data for this group: {item_idx: (data_v1_mean, data_v2_mean, seq_len, gen_len)}
    group_data = {}
    # Store sentence information for this group: {item_idx: {dataset, input_prompt, seq_len, gen_len}}
    sentences_info = {}
    prefill_lengths = []
    gen_lengths = []
    
    for idx, (dataset_name, item) in enumerate(selected_items):
        print(f">>> Processing item {idx+1}/{len(selected_items)} from {dataset_name}")
        prompt = item["input_prompt"]
        prompt_with_format = f"[INST]{prompt}[/INST]"

        input_enc = tokenizer(prompt_with_format, return_tensors="pt", return_offsets_mapping=True)

        input_ids = input_enc.input_ids.to(attention_model.device)
        attention_mask = input_enc.attention_mask.to(attention_model.device)
        
        if input_ids.size(1) > 7500:
            input_ids = torch.cat([input_ids[:, :3750], input_ids[:, -3750:]], dim=1)
            attention_mask = torch.cat([attention_mask[:, :3750], attention_mask[:, -3750:]], dim=1)

        seq_len = input_ids.size(1)
        max_new_tokens = dataset2maxlen.get(dataset_name, 512)
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
        
        # Use all generated tokens (no chunking)
        generated_token_length = len(decoding_attention_maps)
        prefill_lengths.append(seq_len)
        gen_lengths.append(generated_token_length)
        
        first_decoding_attention_map = decoding_attention_maps[0]
        answer_score = torch.zeros(
            (*first_decoding_attention_map.shape[:3], seq_len),
            dtype=first_decoding_attention_map.dtype,
            device=first_decoding_attention_map.device
        )

        # Sum all decoding attention maps
        for atmaps in decoding_attention_maps:
            answer_score += atmaps[:,:,:,:seq_len]
            
        answer_score.squeeze_(dim=2)
        answer_indices = answer_score.topk(token_budget, dim=2).indices
        
        block_size = 8
        window_steps = list(range(block_size, 33*block_size, block_size))
        
        # 두 버전의 데이터 계산
        data_v1 = analyze_version1(prefill_attention_maps, answer_indices, window_steps, block_size, token_budget)
        data_v2 = analyze_version2(prefill_attention_maps, answer_indices, window_steps, block_size, token_budget)
        
        # Calculate layer-wise average (average across all layers)
        data_v1_mean = data_v1.mean(axis=1)  # (Windows,)
        data_v2_mean = data_v2.mean(axis=1)  # (Windows,)
        
        # Store data for this item
        group_data[idx] = (data_v1_mean, data_v2_mean, seq_len, generated_token_length)
        
        # Store sentence information for this item
        sentences_info[idx] = {
            "index": idx,
            "group_name": group_name,
            "dataset": dataset_name,
            "input_prompt": item.get("input_prompt", ""),
            "seq_len": int(seq_len),
            "gen_len": int(generated_token_length)
        }
    
    # Calculate averages for the group
    avg_prefill = np.mean(prefill_lengths)
    avg_gen = np.mean(gen_lengths)
    
    # Save sentences information to jsonl file
    os.makedirs(os.path.join(workpath, "sentences"), exist_ok=True)
    sentences_file = os.path.join(workpath, f"sentences/{group_name.replace(' ', '_')}_sentences.jsonl")
    with open(sentences_file, 'w', encoding='utf-8') as f:
        for idx in sorted(sentences_info.keys()):
            f.write(json.dumps(sentences_info[idx], ensure_ascii=False) + "\n")
    print(f"Saved sentences info to {sentences_file}")
    
    # Plot immediately after processing the group
    plot_group_results(group_name, group_data, avg_prefill, avg_gen, workpath, sentences_info)

print(">>> All analysis and visualization completed")