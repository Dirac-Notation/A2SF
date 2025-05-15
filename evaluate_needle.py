import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from utils_real_drop import KVLlamaForCausalLM, KVOPTForCausalLM, OptimalLlamaForCausalLM
from utils import load_configs

def calculate_lcs_ratio(expected, predicted):
    """Calculate the ratio of longest common subsequence length to expected answer length."""
    if not expected or not predicted:
        return 0.0
    
    # Create a matrix for LCS calculation
    m, n = len(expected), len(predicted)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if expected[i-1] == predicted[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Get the LCS length
    lcs_length = dp[m][n]
    
    # Calculate ratio
    return lcs_length / len(expected)

def load_dataset(file_path):
    """Load the needle-in-haystack dataset from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def evaluate_model(model, tokenizer, dataset, device, method, init_cache_fn=None, cache_params=None):
    """Evaluate the model on the needle-in-haystack task with budget settings."""
    results = defaultdict(list)
    
    # Create directory for saving results
    os.makedirs("result_txt/needle", exist_ok=True)
    result_file = f"result_txt/needle/{method}.jsonl"
    
    # Open file in write mode to overwrite any existing content
    with open(result_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(dataset, desc="Evaluating samples"):
            # Get the prompt and expected answer
            prompt = sample["prompt"]
            expected_answer = sample["answer"]
            needle_position = sample["needle_position"]
            total_tokens = sample["total_tokens"]
            
            # Add [INST] tags for Llama models
            if "llama" in method.lower():
                prompt = f"[INST]{prompt}[/INST]"
            
            # Initialize cache with budget settings if provided
            if init_cache_fn and cache_params:
                init_cache_fn(cache_params)

            # Tokenize the input
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate response
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.0, do_sample=False)

            # Decode the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the model's answer (everything after the prompt)
            model_answer = response[len(prompt):].strip()

            # Calculate LCS ratio
            lcs_ratio = calculate_lcs_ratio(expected_answer, model_answer)
            
            # Record the result
            results[(total_tokens, needle_position)].append({
                "expected": expected_answer,
                "model_answer": model_answer,
                "lcs_ratio": lcs_ratio
            })
            
            # Save to JSONL file
            result_entry = {
                "sentence": model_answer,
                "position": needle_position,
                "length": total_tokens,
                "expected_answer": expected_answer,
                "lcs_ratio": lcs_ratio
            }
            f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
    
    return results

def calculate_metrics(results):
    """Calculate accuracy metrics for each needle position and context length."""
    metrics = {}
    
    for (total_tokens, position), samples in results.items():
        # Calculate average LCS ratio
        total_ratio = sum(sample["lcs_ratio"] for sample in samples)
        total_count = len(samples)
        avg_ratio = total_ratio / total_count if total_count > 0 else 0
        
        # Count correct predictions (LCS ratio > 0.5)
        correct_count = sum(sample["lcs_ratio"] for sample in samples)
        
        metrics[(total_tokens, position)] = {
            "accuracy": avg_ratio,
            "correct_count": correct_count,
            "total_count": total_count
        }
    
    return metrics

def create_heatmap(metrics, output_file):
    """Create a heatmap visualization of the results."""
    # Extract unique context lengths and positions
    context_lengths = sorted(list(set(k[0] for k in metrics.keys())))
    positions = sorted(list(set(k[1] for k in metrics.keys())))
    
    # Create a 2D array for the heatmap
    heatmap_data = np.zeros((len(positions), len(context_lengths)))
    
    # Fill the heatmap data
    for i, position in enumerate(positions):
        for j, length in enumerate(context_lengths):
            if (length, position) in metrics:
                heatmap_data[i, j] = metrics[(length, position)]["accuracy"]
    
    # Create the heatmap using matplotlib
    plt.figure(figsize=(15, 10))
    
    # Create the heatmap
    im = plt.imshow(heatmap_data, cmap='RdYlGn', vmin=0.0, vmax=1.0)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Accuracy', fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    
    # Set axis labels and title
    plt.xlabel('Context Length (tokens)', fontsize=20)
    plt.ylabel('Needle Position (%)', fontsize=20)
    plt.title('Needle-in-Haystack Performance with Budget Settings', fontsize=22, pad=20)
    
    # Set tick labels
    plt.xticks(np.arange(len(context_lengths)), context_lengths, fontsize=16)
    plt.yticks(np.arange(len(positions)), positions, fontsize=16)
    
    # Add text annotations
    for i in range(len(positions)):
        for j in range(len(context_lengths)):
            if (context_lengths[j], positions[i]) in metrics:
                text = plt.text(j, i, f"{heatmap_data[i, j]:.2f}", 
                               ha="center", va="center", color="black", fontsize=16)
    
    plt.tight_layout()
    
    # Save the heatmap with high DPI for better quality
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_file}")

def main(args):
    # Load model paths
    model2path = json.load(open("config/model2path.json", "r"))
    model_path = model2path[args.model]
    
    # Initialize budget and hyperparameter lists
    datasets = args.dataset
    budget_list = args.budget
    methods = args.method

    # Check and extend list lengths
    max_len = len(datasets) * len(budget_list) * len(methods)
    
    # Prepare device, model, and tokenizer
    device = f"cuda:{args.gpu}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load appropriate model based on model name
    if "llama" in args.model.lower():
        model = (KVLlamaForCausalLM.from_pretrained(model_path).to(torch.float16).to(device))
    elif "opt" in args.model.lower():
        model = (KVOPTForCausalLM.from_pretrained(model_path).to(torch.float16).to(device))
    else:
        raise ValueError(f"Unsupported model: {args.model}. Only Llama and OPT models are supported.")

    cur_idx = 0
    for dataset in datasets:
        dataset_name = os.path.basename(dataset).split('.')[0]
        
        # Load dataset
        dataset = load_dataset(dataset)

        # Nested loops: budget → method
        for cur_budget in budget_list:
            for cur_method in methods:
                cur_idx += 1
                config = load_configs(args.model, cur_method, cur_budget)

                # Evaluate with current configuration
                results = evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    device=device,
                    method=cur_method,
                    init_cache_fn=model.init_cache,
                    cache_params=config
                )

                # Calculate metrics
                metrics = calculate_metrics(results)
                
                # Create heatmap
                if cur_method == "full":
                    output_file = f"plots/needle/needle_heatmap_full.png"
                else:
                    output_file = f"plots/needle/needle_heatmap_{cur_method}_budget{cur_budget}.png"
                os.makedirs("plots/needle", exist_ok=True)
                create_heatmap(metrics, output_file)
                
                # Print summary
                print(f"\nConfig {cur_idx}/{max_len} | dataset={dataset_name}, method={cur_method}, budget={cur_budget}")
                print("Position | Accuracy | Correct/Total")
                print("-" * 40)
                for position in sorted(set(k[1] for k in metrics.keys())):
                    # Calculate average accuracy across all context lengths for this position
                    position_samples = [(k[0], v) for k, v in metrics.items() if k[1] == position]
                    total_correct = sum(sample[1]["correct_count"] for sample in position_samples)
                    total_samples = sum(sample[1]["total_count"] for sample in position_samples)
                    avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
                    print(f"{position:3.2f} | {avg_accuracy:.2%} | {total_correct}/{total_samples}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Llama predictions with various budgets on needle-in-haystack task.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, default="llama2", choices=["llama", "llama2", "llama3", "opt"])
    parser.add_argument("--dataset", type=str, nargs='+', default=["datasets/needle_dataset.jsonl"])
    parser.add_argument("--budget", type=int, nargs='+', default=[100])
    parser.add_argument("--method", type=str, nargs='+', default=["h2o"])
    args = parser.parse_args()
    main(args)
