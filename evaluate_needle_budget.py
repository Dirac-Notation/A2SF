import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from utils_real_drop.kv_llama import LlamaForCausalLM
from utils_real_drop.kv_opt import OPTForCausalLM

def load_dataset(file_path):
    """Load the needle-in-haystack dataset from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def evaluate_model(model, tokenizer, dataset, device, init_cache_fn=None, cache_params=None):
    """Evaluate the model on the needle-in-haystack task with budget settings."""
    results = defaultdict(list)
    
    for sample in tqdm(dataset, desc="Evaluating samples"):
        # Get the prompt and expected answer
        prompt = sample["prompt"]
        expected_answer = sample["answer"]
        needle_position = sample["needle_position"]
        total_tokens = sample["total_tokens"]
        
        # Initialize cache with budget settings if provided
        if init_cache_fn and cache_params:
            init_cache_fn(**cache_params)
        
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.0,
                do_sample=False,
            )

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the model's answer (everything after the prompt)
        model_answer = response[len(prompt):].strip()

        # Check if the model found the correct password
        is_correct = expected_answer in model_answer
        
        # Record the result
        results[(total_tokens, needle_position)].append({
            "expected": expected_answer,
            "model_answer": model_answer,
            "is_correct": is_correct
        })
    
    return results

def calculate_metrics(results):
    """Calculate accuracy metrics for each needle position and context length."""
    metrics = {}
    
    for (total_tokens, position), samples in results.items():
        correct_count = sum(1 for sample in samples if sample["is_correct"])
        total_count = len(samples)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        metrics[(total_tokens, position)] = {
            "accuracy": accuracy,
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
    plt.figure(figsize=(12, 8))
    
    # Create the heatmap
    im = plt.imshow(heatmap_data, cmap='RdYlGn', vmin=0.0, vmax=1.0)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Accuracy')
    
    # Set axis labels and title
    plt.xlabel('Context Length (tokens)')
    plt.ylabel('Needle Position (%)')
    plt.title('Needle-in-Haystack Performance with Budget Settings')
    
    # Set tick labels
    plt.xticks(np.arange(len(context_lengths)), context_lengths)
    plt.yticks(np.arange(len(positions)), positions)
    
    # Add text annotations
    for i in range(len(positions)):
        for j in range(len(context_lengths)):
            if (context_lengths[j], positions[i]) in metrics:
                text = plt.text(j, i, f"{heatmap_data[i, j]:.2f}", 
                               ha="center", va="center", color="black")
    
    plt.tight_layout()
    
    # Save the heatmap
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf", 
                        help="Name of the model to use")
    parser.add_argument("--dataset_path", type=str, default="datasets/needle_dataset_Llama-2-7b-hf.jsonl",
                        help="Path to the dataset file")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU number to use (default: 0)")
    parser.add_argument("--select_budget", type=int, nargs='+', default=[100],
                        help="Select budget values")
    parser.add_argument("--recent_budget", type=int, nargs='+', default=[100],
                        help="Recent budget values")
    parser.add_argument("--random_budget", type=int, nargs='+', default=[0],
                        help="Random budget values")
    parser.add_argument("--streaming_budget", type=int, nargs='+', default=[0],
                        help="Streaming budget values")
    parser.add_argument("--forgetting_factor", type=float, nargs='+', default=[1.0],
                        help="Forgetting factor values")
    parser.add_argument("--random_method", type=str, nargs='+', default=["att"],
                        help="Random method values")
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize budget and hyperparameter lists
    sb_list = args.select_budget
    rb_list = args.recent_budget
    randb_list = args.random_budget
    sbud_list = args.streaming_budget
    ff_list = args.forgetting_factor
    rm_list = args.random_method

    # Check and extend list lengths
    lengths = [len(sb_list), len(rb_list), len(randb_list), len(sbud_list)]
    max_len = max(lengths)
    cfg_len = max_len * len(ff_list) * len(rm_list)
    for l in lengths:
        if l != 1 and l != max_len:
            raise ValueError("All budget lists (including streaming) must have the same length or contain only one element.")
    if len(sb_list) == 1: sb_list *= max_len
    if len(rb_list) == 1: rb_list *= max_len
    if len(randb_list) == 1: randb_list *= max_len
    if len(sbud_list) == 1: sbud_list *= max_len
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load appropriate model based on model name
    print(f"Loading model: {args.model_name}")
    if "llama" in args.model_name.lower():
        model = (LlamaForCausalLM.from_pretrained(args.model_name).to(torch.bfloat16).to(device))
    elif "opt" in args.model_name.lower():
        model = (OPTForCausalLM.from_pretrained(args.model_name).to(torch.bfloat16).to(device))
    else:
        raise ValueError(f"Unsupported model: {args.model_name}. Only Llama and OPT models are supported.")
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} samples")
    
    # Prepare output directory
    os.makedirs("plots", exist_ok=True)
    
    # Nested loops: forgetting_factor → random_method → budget configs
    for ff in ff_list:
        for rm in rm_list:
            for idx in range(max_len):
                cur_sb = sb_list[idx]
                cur_rb = rb_list[idx]
                cur_rand = randb_list[idx]
                cur_sbud = sbud_list[idx]
                
                # Warm-up
                print(f"Warming up with config: sel={cur_sb}, rec={cur_rb}, ran={cur_rand}, str={cur_sbud}, ff={ff}, rm={rm}")
                for sample in dataset[:10]:
                    model.init_cache(
                        use_compression=False,
                        select_budget=cur_sb,
                        recent_budget=cur_rb,
                        random_budget=cur_rand,
                        streaming_budget=cur_sbud,
                        random_method=rm,
                        forgetting_factor=ff
                    )
                    prompt = sample["prompt"]
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        _ = model.generate(
                            **inputs,
                            max_new_tokens=16,
                            temperature=0.0,
                            do_sample=False,
                        )
                
                # Evaluate with current configuration
                print(f"Evaluating with config: sel={cur_sb}, rec={cur_rb}, ran={cur_rand}, str={cur_sbud}, ff={ff}, rm={rm}")
                results = evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    device=device,
                    init_cache_fn=model.init_cache,
                    cache_params={
                        'use_compression': True,
                        'select_budget': cur_sb,
                        'recent_budget': cur_rb,
                        'random_budget': cur_rand,
                        'streaming_budget': cur_sbud,
                        'random_method': rm,
                        'forgetting_factor': ff
                    }
                )
                
                # Calculate metrics
                metrics = calculate_metrics(results)
                
                # Create heatmap
                output_file = f"plots/needle_heatmap_budget_s{cur_sb}_r{cur_rb}_ra{cur_rand}_st{cur_sbud}_ff{ff}_rm{rm}.png"
                print(f"Creating heatmap visualization: {output_file}")
                create_heatmap(metrics, output_file)
                
                # Print summary
                print("\nSummary of results:")
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
    main()
