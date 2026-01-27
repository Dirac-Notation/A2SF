import torch
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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


def analyze_block_hit_rate(prefill_attention_maps, answer_indices, max_window, token_budget, block_size=8):
    """
    Calculate hit rate for each block (8 positions) from the end
    Returns: data_to_plot (Blocks, Layers)
    """
    seq_len = prefill_attention_maps.size(2)
    num_blocks = (min(max_window, seq_len) + block_size - 1) // block_size
    
    block_sim = []
    
    for block_idx in range(num_blocks):
        start_offset = block_idx * block_size + 1
        end_offset = min((block_idx + 1) * block_size + 1, max_window + 1, seq_len + 1)
        block_positions = list(range(start_offset, end_offset))
        
        if len(block_positions) == 0:
            break
        
        # Sum attention scores for all positions in this block
        block_score = torch.zeros(
            prefill_attention_maps.size(0),
            prefill_attention_maps.size(1),
            seq_len,
            dtype=torch.float32,
            device=prefill_attention_maps.device
        )
        
        for pos_offset in block_positions:
            pos_idx = -pos_offset
            block_score += prefill_attention_maps[:, :, pos_idx, :]
        
        block_indices = block_score.topk(token_budget, dim=2).indices
        block_sim.append(layer_wise_analysis(answer_indices, block_indices))
    
    # Convert to numpy: (Blocks, Layers)
    return np.array(block_sim)


def analyze_optimal_contribution(prefill_attention_maps, answer_indices, max_window, token_budget, block_size=8):
    """
    Find optimal contribution coefficients for each block (8 positions) from the end
    Returns: (optimal_coefficients, hit_rates) - both are lists for each block from -1~-8, -9~-16, ...
    """
    num_layers = prefill_attention_maps.size(0)
    num_heads = prefill_attention_maps.size(1)
    seq_len = prefill_attention_maps.size(2)
    
    # Test coefficients from 0.0 to 1.0 in 0.1 steps
    test_coefficients = np.arange(0.0, 1.1, 0.1)
    
    optimal_coefficients = []
    hit_rates = []
    accumulated_score = torch.zeros(
        num_layers, num_heads, seq_len,
        dtype=torch.float32,
        device=prefill_attention_maps.device
    )
    
    # Calculate number of blocks
    num_blocks = (min(max_window, seq_len) + block_size - 1) // block_size
    
    # Process in blocks of block_size
    for block_idx in range(num_blocks):
        start_offset = block_idx * block_size + 1
        end_offset = min((block_idx + 1) * block_size + 1, max_window + 1, seq_len + 1)
        block_positions = list(range(start_offset, end_offset))
        
        if len(block_positions) == 0:
            break
        
        # Get the range of positions for this block (from end)
        # For example, block 0: -1~-16, block 1: -17~-32, etc.
        # block_positions = [1, 2, ..., 16] means positions -1, -2, ..., -16 from the end
        
        # First block: fix coefficient to 1.0
        if block_idx == 0:
            best_coeff = 1.0
            print(f"  >>> Block {block_idx+1}/{num_blocks} (positions {start_offset}~{end_offset-1} from end) - fixed to 1.0", end="", flush=True)
            
            # Update accumulated score with coefficient 1.0 for all positions in first block
            for pos_offset in block_positions:
                pos_idx = -pos_offset
                accumulated_score += best_coeff * prefill_attention_maps[:, :, pos_idx, :]
            
            # Calculate hit rate for first block
            temp_indices = accumulated_score.topk(token_budget, dim=2).indices
            similarities = layer_wise_analysis(answer_indices, temp_indices)
            best_hit_rate = np.mean(similarities)
            
            optimal_coefficients.append(best_coeff)
            hit_rates.append(best_hit_rate)
            print(f" Done (coeff={best_coeff:.1f}, hit_rate={best_hit_rate:.4f})")
        else:
            # Other blocks: find optimal coefficient
            print(f"  >>> Finding optimal coefficient for block {block_idx+1}/{num_blocks} (positions {start_offset}~{end_offset-1} from end)", end="", flush=True)
            
            best_coeff = 0.0
            best_hit_rate = -1.0
            
            # Try each coefficient
            for coeff_idx, coeff in enumerate(test_coefficients):
                # Create temporary score with current coefficient applied to all positions in this block
                temp_score = accumulated_score.clone()
                # Apply coefficient to all positions in this block
                for pos_offset in block_positions:
                    pos_idx = -pos_offset
                    temp_score += coeff * prefill_attention_maps[:, :, pos_idx, :]
                
                # Calculate hit rate
                temp_indices = temp_score.topk(token_budget, dim=2).indices
                similarities = layer_wise_analysis(answer_indices, temp_indices)
                avg_hit_rate = np.mean(similarities)
                
                # Update best if this is better, or if equal and coefficient is larger
                if avg_hit_rate > best_hit_rate or (avg_hit_rate == best_hit_rate and coeff > best_coeff):
                    best_hit_rate = avg_hit_rate
                    best_coeff = coeff
                
                # Show progress for coefficient testing
                if (coeff_idx + 1) % 3 == 0 or coeff_idx == len(test_coefficients) - 1:
                    print(".", end="", flush=True)
            
            # Update accumulated score with best coefficient for all positions in this block
            for pos_offset in block_positions:
                pos_idx = -pos_offset
                accumulated_score += best_coeff * prefill_attention_maps[:, :, pos_idx, :]
            
            optimal_coefficients.append(best_coeff)
            hit_rates.append(best_hit_rate)
            
            print(f" Done (coeff={best_coeff:.1f}, hit_rate={best_hit_rate:.4f})")
    
    return optimal_coefficients, hit_rates


def plot_item_result(group_name, item_idx, block_hit_rates, optimal_coefficients, hit_rates, 
                     seq_len, gen_len, workpath, dataset_name=None):
    """
    Plot temporal bias analysis results for a single item (prompt)
    Args:
        group_name: Name of the group
        item_idx: Index of the item
        block_hit_rates: Hit rates for each block
        optimal_coefficients: Optimal coefficients for each block
        hit_rates: Hit rates for optimal coefficients
        seq_len: Sequence length
        gen_len: Generation length
        workpath: Path to save the plot
        dataset_name: Name of the dataset (optional)
    """
    print(f">>> Plotting item {item_idx+1} from group: {group_name} (Prefill: {seq_len}, Gen: {gen_len})")
    
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
    
    color = plt.cm.tab20(0)  # Use first color for single item
    
    # Left subplot: Block-wise hit rate
    block_indices = list(range(1, len(block_hit_rates) + 1))  # Block 1, 2, 3, ... (each block = 8 positions)
    ax1.plot(block_indices, block_hit_rates, 
            alpha=0.9, linewidth=2.5, linestyle='--',
            color=color, marker='o', markersize=4)
    ax1.set_title(f"Hit rate of each block", fontsize=22)
    ax1.set_xlabel("Block index (each block = 8 positions)", fontsize=22)
    ax1.set_ylabel("Hit rate", fontsize=22)
    ax1.tick_params(labelsize=22)
    ax1.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.text(0.98, 0.50, f"Prefill: {seq_len}\nGen: {gen_len}", 
             transform=ax1.transAxes, fontsize=16, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right subplot: Optimal contribution coefficients with hit rates (dual y-axis)
    ax2_twin = ax2.twinx()  # Create secondary y-axis
    
    block_indices = list(range(1, len(optimal_coefficients) + 1))  # Block 1, 2, 3, ... (each block = 8 positions)
    
    # Plot optimal coefficients on primary y-axis
    ax2.plot(block_indices, optimal_coefficients, 
            alpha=0.9, linewidth=2.5, linestyle='-',
            color=color, marker='o', markersize=4, label='Coefficient')
    
    # Plot hit rates on secondary y-axis
    ax2_twin.plot(block_indices, hit_rates, 
                 alpha=0.7, linewidth=2.0, linestyle='--',
                 color=color, marker='s', markersize=3, label='Hit rate')
    
    ax2.set_title(f"Optimal contribution coefficients & Hit rates", fontsize=22)
    ax2.set_xlabel("Block index (each block = 8 positions)", fontsize=22)
    ax2.set_ylabel("Optimal coefficient", fontsize=22, color='black')
    ax2.tick_params(labelsize=22, axis='y', labelcolor='black')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    
    ax2_twin.set_ylabel("Hit rate", fontsize=22, color='gray')
    ax2_twin.set_ylim(0, 0.8)
    ax2_twin.tick_params(labelsize=22, axis='y', labelcolor='gray')
    
    ax2.text(0.98, 0.50, f"Prefill: {seq_len}\nGen: {gen_len}", 
             transform=ax2.transAxes, fontsize=16, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Create group folder and save
    group_folder = os.path.join(workpath, "plots", group_name.replace(' ', '_'))
    os.makedirs(group_folder, exist_ok=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Overall figure title (after tight_layout to position correctly)
    title = f"{group_name} - Item {item_idx+1}"
    if dataset_name:
        title += f" ({dataset_name})"
    fig.suptitle(title, fontsize=24, fontweight='bold', y=0.95)
    
    filename = f"item_{item_idx+1}"
    if dataset_name:
        filename += f"_{dataset_name.replace(' ', '_')}"
    save_path = os.path.join(group_folder, f"{filename}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()

# ---------------------------------------------------------
# 2. 데이터 준비 및 프롬프트 구성
# ---------------------------------------------------------
# Load max generation lengths
dataset2maxlen_path = os.path.join(root_path, "config", "dataset2maxlen.json")
with open(dataset2maxlen_path, "r") as f:
    dataset2maxlen = json.load(f)

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
model2path_path = os.path.join(root_path, "config", "model2path.json")
with open(model2path_path, "r") as f:
    model2path = json.load(f)
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
        prefill_attention_list.append(attention_map.detach().to("cpu", dtype=torch.float32))
        return (output[0], None, output[2])
    return output

# 모든 레이어에 hook 등록
for layer in attention_model.model.layers:
    layer.self_attn.register_forward_hook(get_attn_hook)

# Process each group
for group_name, selected_items in selected_data.items():
    print(f"\n>>> Processing group: {group_name} ({len(selected_items)} items)")
    
    # Store sentence information for this group: {item_idx: {dataset, input_prompt, seq_len, gen_len}}
    sentences_info = {}
    
    for idx, (dataset_name, item) in enumerate(selected_items):
        print(f">>> Processing item {idx+1}/{len(selected_items)} from {dataset_name}")
        prompt = item["input_prompt"]
        prompt_with_format = f"[INST]{prompt}[/INST]"

        input_enc = tokenizer(prompt_with_format, return_tensors="pt")

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
        
        # Extract attention weights and convert to float32
        prefill_attention_maps = torch.stack(prefill_attention_list, dim=0).squeeze(1).float()
        decoding_attention_maps = [torch.stack(output.attentions[i], dim=0).squeeze(1).to("cpu", dtype=torch.float32) for i in range(1, len(output.attentions))]
        
        # Use all generated tokens (no chunking)
        generated_token_length = len(decoding_attention_maps)
        
        first_decoding_attention_map = decoding_attention_maps[0]
        answer_score = torch.zeros(
            (*first_decoding_attention_map.shape[:3], seq_len),
            dtype=torch.float32,
            device=first_decoding_attention_map.device
        )

        # Sum all decoding attention maps
        for atmaps in decoding_attention_maps:
            answer_score += atmaps[:,:,:,:seq_len]
            
        answer_score.squeeze_(dim=2)
        answer_indices = answer_score.topk(token_budget, dim=2).indices
        
        # Limit window size for reasonable computation
        max_window = min(seq_len, 128)
        block_size = 8
        
        # Calculate block-wise hit rate
        block_hit_rates_data = analyze_block_hit_rate(
            prefill_attention_maps, answer_indices, max_window, token_budget, block_size
        )
        block_hit_rates_mean = block_hit_rates_data.mean(axis=1)  # (Blocks,)
        
        # Calculate optimal contribution coefficients and hit rates
        optimal_coefficients, hit_rates = analyze_optimal_contribution(
            prefill_attention_maps, answer_indices, max_window, token_budget, block_size
        )
        
        # Store sentence information for this item
        sentences_info[idx] = {
            "index": idx,
            "group_name": group_name,
            "dataset": dataset_name,
            "input_prompt": item.get("input_prompt", ""),
            "seq_len": int(seq_len),
            "gen_len": int(generated_token_length)
        }
        
        # Plot immediately after processing each item
        plot_item_result(
            group_name, idx, block_hit_rates_mean, optimal_coefficients, hit_rates,
            seq_len, generated_token_length, workpath, dataset_name
        )
    
    # Save sentences information to jsonl file
    os.makedirs(os.path.join(workpath, "sentences"), exist_ok=True)
    sentences_file = os.path.join(workpath, f"sentences/{group_name.replace(' ', '_')}_sentences.jsonl")
    with open(sentences_file, 'w', encoding='utf-8') as f:
        for idx in sorted(sentences_info.keys()):
            f.write(json.dumps(sentences_info[idx], ensure_ascii=False) + "\n")
    print(f"Saved sentences info to {sentences_file}")

print(">>> All analysis and visualization completed")