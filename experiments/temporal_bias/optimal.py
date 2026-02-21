import torch
import json
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
                if similarities > best_hit_rate:
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


def plot_single_result(group_name, dataset_name, prompt_idx, block_hit_rates, optimal_coefficients, hit_rates,
                       workpath, seq_len, gen_len):
    """
    Plot temporal bias analysis result for a single prompt.
    Saves to task_name/block_hit/ and task_name/coeff_hit/ with filename dataset_name_{idx}.png
    """
    max_blocks = len(block_hit_rates)
    block_indices = list(range(max_blocks))

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
    color = plt.cm.tab20(0)

    base_folder = os.path.join(workpath, "plots", group_name.replace(' ', '_'))
    block_hit_folder = os.path.join(base_folder, "block_hit")
    coeff_hit_folder = os.path.join(base_folder, "coeff_hit")
    os.makedirs(block_hit_folder, exist_ok=True)
    os.makedirs(coeff_hit_folder, exist_ok=True)

    file_suffix = f"{dataset_name.replace(' ', '_')}_{prompt_idx}"

    # 첫 번째 그래프: 블록별 Hit rate → task_name/block_hit/
    fig1, ax1 = plt.subplots(1, 1, figsize=(7.5, 5))
    ax1.plot(
        block_indices,
        block_hit_rates,
        alpha=0.9,
        linewidth=2.5,
        linestyle='--',
        color=color,
        marker='o',
        markersize=4,
    )
    ax1.set_xlabel("Query Block index", fontsize=22)
    ax1.set_ylabel("Hit rate", fontsize=22)
    ax1.tick_params(labelsize=22)
    ax1.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.text(
        0.98,
        0.50,
        f"Prefill: {seq_len:.0f}tokens\nGen: {gen_len:.0f}tokens",
        transform=ax1.transAxes,
        fontsize=16,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )
    ax1.set_title(f"{dataset_name} (prompt {prompt_idx})", fontsize=24, fontweight='bold')
    plt.tight_layout()
    save_path_block = os.path.join(block_hit_folder, f"{file_suffix}.png")
    plt.savefig(save_path_block, dpi=300, bbox_inches='tight')
    print(f"Saved block hit rate plot to {save_path_block}")
    plt.close(fig1)

    # 두 번째 그래프: Optimal coefficient & Hit rate → task_name/coeff_hit/
    fig2, ax2 = plt.subplots(1, 1, figsize=(7.5, 5))
    ax2_twin = ax2.twinx()
    x_vals = np.array(block_indices, dtype=np.float64)
    y_vals = np.array(optimal_coefficients, dtype=np.float64)

    ax2.plot(
        x_vals,
        y_vals,
        alpha=0.9,
        linewidth=2.5,
        linestyle='-',
        color=color,
        marker='o',
        markersize=4,
        label='Coefficient',
    )

    def sigmoid_func(x, a, b):
        return 1/(1+np.exp(a*(x-b)))
    
    popt1, _ = curve_fit(
        sigmoid_func,
        x_vals,
        y_vals,
        p0=[1.0, 16.0],
        bounds=((0, 0), (np.inf, np.inf)),
    )
    
    sigmoid_fit_curve = sigmoid_func(x_vals, *popt1)
    
    ax2.plot(
        block_indices,
        sigmoid_fit_curve,
        alpha=0.8,
        linewidth=2.0,
        linestyle=':',
        color='red',
        marker=None,
        label='Sigmoid fit',
    )
    ax2.text(
        0.98,
        0.65,
        f"a = {popt1[0]:.2f}, b = {popt1[1]:.2f}",
        transform=ax2.transAxes,
        fontsize=16,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
    )

    ax2_twin.plot(
        block_indices,
        hit_rates,
        alpha=0.7,
        linewidth=2.0,
        linestyle='--',
        color=color,
        # marker='s',
        # markersize=3,
        label='Hit rate',
    )
    ax2.set_xlabel("QueryBlock index", fontsize=22)
    ax2.set_ylabel("Optimal coefficient", fontsize=22, color='black')
    ax2.tick_params(labelsize=22, axis='y', labelcolor='black')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    ax2_twin.set_ylabel("Hit rate", fontsize=22, color='black')
    ax2_twin.tick_params(labelsize=22, axis='y', labelcolor='black')
    ax2.text(
        0.98,
        0.50,
        f"Prefill: {seq_len:.0f}tokens\nGen: {gen_len:.0f}tokens",
        transform=ax2.transAxes,
        fontsize=16,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )
    ax2.set_title(f"{dataset_name} (prompt {prompt_idx})", fontsize=24, fontweight='bold')
    plt.tight_layout()
    save_path_coeff = os.path.join(coeff_hit_folder, f"{file_suffix}.png")
    plt.savefig(save_path_coeff, dpi=300, bbox_inches='tight')
    print(f"Saved coefficient & hit rate plot to {save_path_coeff}")
    plt.close(fig2)


# ---------------------------------------------------------
# 2. 데이터 준비 및 프롬프트 구성 (Main Execution)
# ---------------------------------------------------------

dataset2maxlen_path = os.path.join(root_path, "config", "dataset2maxlen.json")
with open(dataset2maxlen_path, "r") as f:
    dataset2maxlen = json.load(f)

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
    "Few Shot": ["samsum"],
    "Single-doc QA": ["qasper"],
    "Multi-doc QA": ["hotpotqa"],
    "Summarization": ["gov_report"],
}

# 각 데이터셋별로 num_items개씩 문장을 뽑되,
# 어떤 task에 속한 데이터셋인지 정보는 함께 유지한다.
num_items = 10
dataset_selected_data = {}
dataset2task = {}
for task_name, datasets in task_group.items():
    for d_name in datasets:
        dataset2task[d_name] = task_name
        if d_name not in dataset_prompts:
            continue
        prompts = dataset_prompts[d_name]
        if len(prompts) > num_items:
            sampled = random.sample(prompts, num_items)
        else:
            sampled = prompts
        dataset_selected_data[d_name] = sampled

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

prefill_attention_list = []

def get_attn_hook(module, input, output):
    attention_map = output[1]
    if attention_map is not None and attention_map.size(2) != 1:
        prefill_attention_list.append(attention_map.detach().to("cpu"))
        return (output[0], None, output[2])
    return output

for layer in attention_model.model.layers:
    layer.self_attn.register_forward_hook(get_attn_hook)

for dataset_name, selected_items in dataset_selected_data.items():
    task_name = dataset2task.get(dataset_name, "UnknownTask")
    print(f"\n>>> Processing dataset: {dataset_name} (task: {task_name}, {len(selected_items)} items)")
    sentences_info = {}

    for idx, (ds_name, prompt) in enumerate(selected_items):
        print(f">>> Processing item {idx+1}/{len(selected_items)} from {ds_name}")
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

        prefill_attention_maps = torch.stack(prefill_attention_list, dim=0).squeeze(1)
        decoding_attention_maps = [torch.stack(output.attentions[i], dim=0).squeeze(1).to("cpu") for i in range(1, len(output.attentions))]
        generated_token_length = len(decoding_attention_maps)

        first_decoding_attention_map = decoding_attention_maps[0]
        answer_score = torch.zeros((*first_decoding_attention_map.shape[:2], seq_len), device=first_decoding_attention_map.device)

        for atmaps in decoding_attention_maps:
            answer_score += atmaps[:,:,0,:seq_len]

        answer_indices = answer_score.topk(token_budget, dim=2).indices
        max_window, block_size = 128, 4

        block_hit_rates_data = analyze_block_hit_rate(prefill_attention_maps, answer_indices, max_window, token_budget, block_size)
        optimal_coefficients, hit_rates = analyze_optimal_contribution(prefill_attention_maps, answer_indices, max_window, token_budget, block_size)

        plot_single_result(
            task_name, dataset_name, idx,
            block_hit_rates_data, optimal_coefficients, hit_rates,
            workpath, seq_len, generated_token_length
        )

        sentences_info[idx] = {
            "index": idx, "dataset": dataset_name,
            "input_prompt": prompt, "seq_len": int(seq_len), "gen_len": int(generated_token_length)
        }
    
    # 문장 정보도 Task 폴더 안에 저장
    sentences_group_folder = os.path.join(workpath, "sentences", task_name.replace(' ', '_'))
    os.makedirs(sentences_group_folder, exist_ok=True)
    sentences_file = os.path.join(
        sentences_group_folder,
        f"{dataset_name.replace(' ', '_')}_sentences.jsonl"
    )
    with open(sentences_file, 'w', encoding='utf-8') as f:
        for idx in sorted(sentences_info.keys()):
            f.write(json.dumps(sentences_info[idx], ensure_ascii=False) + "\n")

print(">>> All analysis and visualization completed")