import torch
import json
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

workpath = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    return sum(similarities) / num_layers


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
        
        block_score = prefill_attention_maps[:, :, -end_offset:-start_offset, :].sum(dim=2)
        
        block_indices = block_score.topk(token_budget, dim=2).indices
        block_sim.append(layer_wise_analysis(answer_indices, block_indices))
    
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
        device=prefill_attention_maps.device
    )
    
    local_budget = int(token_budget * 0.125)
    selective_budget = token_budget - local_budget

    # Calculate number of blocks
    num_blocks = (min(max_window, seq_len) + block_size - 1) // block_size
    
    # Process in blocks of block_size
    for block_idx in range(num_blocks):
        start_offset = block_idx * block_size + 1
        end_offset = min((block_idx + 1) * block_size + 1, max_window + 1, seq_len + 1)
        
        # First block: fix coefficient to 1.0
        if block_idx == 0:
            best_coeff = 1.0
            print(f"  >>> Block {block_idx+1}/{num_blocks} (positions {start_offset}~{end_offset-1} from end) - fixed to 1.0", end="", flush=True)
            
            # Update accumulated score with coefficient 1.0 for all positions in first block
            accumulated_score += prefill_attention_maps[:, :, -end_offset:-start_offset, :].sum(dim=2)
            
            # Calculate hit rate for first block
            temp_indices = accumulated_score.topk(token_budget, dim=2).indices
            similarities = layer_wise_analysis(answer_indices, temp_indices)
            best_hit_rate = similarities
            
            optimal_coefficients.append(best_coeff)
            hit_rates.append(best_hit_rate)
            print(f" Done (coeff={best_coeff:.1f}, hit_rate={best_hit_rate:.4f})")
        else:
            # Other blocks: find optimal coefficient
            print(f"  >>> Finding optimal coefficient for block {block_idx+1}/{num_blocks} (positions {start_offset}~{end_offset-1} from end)", end="", flush=True)
            
            best_coeff = 0.0
            
            # Try each coefficient
            for coeff_idx, coeff in enumerate(test_coefficients):
                # Create temporary score with current coefficient applied to all positions in this block
                temp_score = accumulated_score.clone()
                # Apply coefficient to all positions in this block
                temp_score += coeff * prefill_attention_maps[:, :, -end_offset:-start_offset, :].sum(dim=2)
                
                # Calculate hit rate
                temp_score[:,:,-local_budget:] = temp_score.max()
                temp_indices = temp_score.topk(selective_budget, dim=2).indices
                similarities = layer_wise_analysis(answer_indices, temp_indices)
                
                # Update best if this is better, or if equal and coefficient is larger
                if similarities >= best_hit_rate:
                    best_hit_rate = similarities
                    best_coeff = coeff
                
                # Show progress for coefficient testing
                if (coeff_idx + 1) % 3 == 0 or coeff_idx == len(test_coefficients) - 1:
                    print(".", end="", flush=True)
            
            # Update accumulated score with best coefficient for all positions in this block
            accumulated_score += best_coeff * prefill_attention_maps[:, :, -end_offset:-start_offset, :].sum(dim=2)
            
            optimal_coefficients.append(best_coeff)
            hit_rates.append(best_hit_rate)
            
            print(f" Done (coeff={best_coeff:.1f}, hit_rate={best_hit_rate:.4f})")
    
    return optimal_coefficients, hit_rates


def plot_averaged_result(group_name, all_block_hit_rates, all_optimal_coefficients, all_hit_rates, 
                         workpath, avg_seq_len, avg_gen_len):
    """
    Plot averaged temporal bias analysis results for all items in a task group
    Args:
        group_name: Name of the group (task)
        all_block_hit_rates: List of block_hit_rates arrays for all items
        all_optimal_coefficients: List of optimal_coefficients lists for all items
        all_hit_rates: List of hit_rates lists for all items
        workpath: Path to save the plot
        avg_seq_len: Average prefill sequence length
        avg_gen_len: Average generation length
    """
    print(f">>> Plotting averaged results for group: {group_name} (Avg Prefill: {avg_seq_len:.0f}, Avg Gen: {avg_gen_len:.0f})")
    
    # Find maximum number of blocks across all items
    max_blocks = max(len(bhr) for bhr in all_block_hit_rates)
    
    # Pad all arrays/lists to the same length (max_blocks)
    padded_block_hit_rates = []
    padded_optimal_coefficients = []
    padded_hit_rates = []
    
    for i in range(len(all_block_hit_rates)):
        # Pad block_hit_rates (numpy array)
        bhr = all_block_hit_rates[i]
        if len(bhr) < max_blocks:
            padded = np.pad(bhr, (0, max_blocks - len(bhr)), mode='constant', constant_values=np.nan)
        else:
            padded = bhr
        padded_block_hit_rates.append(padded)
        
        # Pad optimal_coefficients (list)
        oc = all_optimal_coefficients[i]
        if len(oc) < max_blocks:
            padded_oc = oc + [np.nan] * (max_blocks - len(oc))
        else:
            padded_oc = oc[:max_blocks]
        padded_optimal_coefficients.append(padded_oc)
        
        # Pad hit_rates (list)
        hr = all_hit_rates[i]
        if len(hr) < max_blocks:
            padded_hr = hr + [np.nan] * (max_blocks - len(hr))
        else:
            padded_hr = hr[:max_blocks]
        padded_hit_rates.append(padded_hr)
    
    # Convert to numpy arrays and calculate averages (ignoring NaN values)
    padded_block_hit_rates = np.array(padded_block_hit_rates)
    padded_optimal_coefficients = np.array(padded_optimal_coefficients)
    padded_hit_rates = np.array(padded_hit_rates)
    
    avg_block_hit_rates = np.nanmean(padded_block_hit_rates, axis=0)
    avg_optimal_coefficients = np.nanmean(padded_optimal_coefficients, axis=0)
    avg_hit_rates = np.nanmean(padded_hit_rates, axis=0)
    
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
    
    color = plt.cm.tab20(0)  # Use first color for averaged result
    
    # Left subplot: Block-wise hit rate
    block_indices = list(range(1, max_blocks + 1))  # Block 1, 2, 3, ...
    ax1.plot(block_indices, avg_block_hit_rates, 
            alpha=0.9, linewidth=2.5, linestyle='--',
            color=color, marker='o', markersize=4)
    ax1.set_title(f"Hit rate of each block", fontsize=22)
    ax1.set_xlabel("Block index", fontsize=22)
    ax1.set_ylabel("Hit rate", fontsize=22)
    ax1.tick_params(labelsize=22)
    ax1.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.text(0.98, 0.50, f"Prefill: {avg_seq_len:.0f}\nGen: {avg_gen_len:.0f}", 
             transform=ax1.transAxes, fontsize=16, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right subplot: Optimal contribution coefficients with hit rates (dual y-axis)
    ax2_twin = ax2.twinx()  # Create secondary y-axis
    
    block_indices = list(range(1, max_blocks + 1))  # Block 1, 2, 3, ...
    
    # Plot optimal coefficients on primary y-axis
    ax2.plot(block_indices, avg_optimal_coefficients, 
            alpha=0.9, linewidth=2.5, linestyle='-',
            color=color, marker='o', markersize=4, label='Coefficient')
    
    # Plot hit rates on secondary y-axis
    ax2_twin.plot(block_indices, avg_hit_rates, 
                 alpha=0.7, linewidth=2.0, linestyle='--',
                 color=color, marker='s', markersize=3, label='Hit rate')
    
    ax2.set_title(f"Optimal contribution coefficients & Hit rates", fontsize=22)
    ax2.set_xlabel("Block index", fontsize=22)
    ax2.set_ylabel("Optimal coefficient", fontsize=22, color='black')
    ax2.tick_params(labelsize=22, axis='y', labelcolor='black')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    
    ax2_twin.set_ylabel("Hit rate", fontsize=22, color='black')
    ax2_twin.set_ylim(0, 0.8)
    ax2_twin.tick_params(labelsize=22, axis='y', labelcolor='black')
    
    ax2.text(0.98, 0.50, f"Prefill: {avg_seq_len:.0f}\nGen: {avg_gen_len:.0f}", 
             transform=ax2.transAxes, fontsize=16, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Create group folder and save
    group_folder = os.path.join(workpath, "plots", group_name.replace(' ', '_'))
    os.makedirs(group_folder, exist_ok=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Overall figure title (after tight_layout to position correctly)
    title = f"{group_name}"
    fig.suptitle(title, fontsize=24, fontweight='bold', y=0.95)
    
    filename = f"averaged_result"
    save_path = os.path.join(group_folder, f"{filename}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved averaged plot to {save_path}")
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

longbench_folder_path = os.path.join(root_path, "datasets", "longbench")
dataset_list = os.listdir(longbench_folder_path)
dataset_prompts = {}
for dataset in dataset_list:
    dataset_path = os.path.join(longbench_folder_path, dataset)
    
    with open(dataset_path, "r") as f:
        data_lines = f.readlines()
    
    for line in data_lines:
        item = json.loads(line)
        input_prompt = item.get("input_prompt")
        dataset_name = item.get("dataset")
        length = item.get("length")

        if length < 2000 or length > 6000:
            continue

        if dataset_name not in dataset_prompts:
            dataset_prompts[dataset_name] = []
        dataset_prompts[dataset_name].append((dataset_name, input_prompt))

task_group = {
    "Code Complete": ["repobench-p", "lcc"],
    "Few Shot": ["trec", "triviaqa", "samsum", "lsht"],
    "Single-doc QA": ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
    "Multi-doc QA": ["hotpotqa", "2wikimqa", "musique", "dureader"],
    "Summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "Passage Retrieval": ["passage_retrieval_en", "passage_retrieval_zh", "passage_count"],
}

selected_data = {}
for task_name, dataset_name in task_group.items():
    all_prompts = []
    for dataset_name in dataset_name:
        if dataset_name in dataset_prompts:
            all_prompts.extend(dataset_prompts[dataset_name])
    if len(all_prompts) > 5:
        selected_data[task_name] = random.sample(all_prompts, 5)

# Load tokenizer and attention model
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
        prefill_attention_list.append(attention_map.detach().to("cpu"))
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
    
    # Store results for averaging
    all_block_hit_rates = []
    all_optimal_coefficients = []
    all_hit_rates = []
    all_seq_lens = []
    all_gen_lens = []
    
    for idx, (dataset_name, prompt) in enumerate(selected_items):
        print(f">>> Processing item {idx+1}/{len(selected_items)} from {dataset_name}")
        prompt_with_format = f"[INST]{prompt}[/INST]"

        input_enc = tokenizer(prompt_with_format, return_tensors="pt")

        input_ids = input_enc.input_ids.to(attention_model.device)
        attention_mask = input_enc.attention_mask.to(attention_model.device)
        
        if input_ids.size(1) > 7500:
            input_ids = torch.cat([input_ids[:, :3750], input_ids[:, -3750:]], dim=1)
            attention_mask = torch.cat([attention_mask[:, :3750], attention_mask[:, -3750:]], dim=1)

        seq_len = input_ids.size(1)
        max_new_tokens = dataset2maxlen.get(dataset_name, 512)
        token_budget = 128

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
        
        first_decoding_attention_map = decoding_attention_maps[0]
        answer_score = torch.zeros(
            (*first_decoding_attention_map.shape[:2], seq_len),
            device=first_decoding_attention_map.device
        )

        # Sum all decoding attention maps
        for atmaps in decoding_attention_maps:
            answer_score += atmaps[:,:,0,:seq_len]
            
        answer_indices = answer_score.topk(token_budget, dim=2).indices
        
        # Limit window size for reasonable computation
        max_window = 128
        block_size = 4
        
        # Calculate block-wise hit rate
        block_hit_rates_data = analyze_block_hit_rate(
            prefill_attention_maps, answer_indices, max_window, token_budget, block_size
        )
        
        # Calculate optimal contribution coefficients and hit rates
        optimal_coefficients, hit_rates = analyze_optimal_contribution(
            prefill_attention_maps, answer_indices, max_window, token_budget, block_size
        )
        
        # Store results for averaging
        all_block_hit_rates.append(block_hit_rates_data)
        all_optimal_coefficients.append(optimal_coefficients)
        all_hit_rates.append(hit_rates)
        all_seq_lens.append(seq_len)
        all_gen_lens.append(generated_token_length)
        
        # Store sentence information for this item
        sentences_info[idx] = {
            "index": idx,
            "group_name": group_name,
            "dataset": dataset_name,
            "input_prompt": prompt,
            "seq_len": int(seq_len),
            "gen_len": int(generated_token_length)
        }
    
    # Calculate average sequence and generation lengths
    avg_seq_len = np.mean(all_seq_lens)
    avg_gen_len = np.mean(all_gen_lens)
    
    # Plot averaged results for this group
    plot_averaged_result(
        group_name, all_block_hit_rates, all_optimal_coefficients, all_hit_rates,
        workpath, avg_seq_len, avg_gen_len
    )
    
    # Save sentences information to jsonl file
    os.makedirs(os.path.join(workpath, "sentences"), exist_ok=True)
    sentences_file = os.path.join(workpath, f"sentences/{group_name.replace(' ', '_')}_sentences.jsonl")
    with open(sentences_file, 'w', encoding='utf-8') as f:
        for idx in sorted(sentences_info.keys()):
            f.write(json.dumps(sentences_info[idx], ensure_ascii=False) + "\n")
    print(f"Saved sentences info to {sentences_file}")

print(">>> All analysis and visualization completed")