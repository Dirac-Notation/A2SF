import json
import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import datetime
import sys

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_model, set_seed, CompressionConfig


def check_exact_match(expected, predicted):
    """Check if the expected answer appears exactly in the predicted text."""
    if not expected or not predicted:
        return False
    
    # Strip whitespace
    expected_clean = expected.strip()
    predicted_clean = predicted.strip()
    
    # Check exact match
    if expected_clean == predicted_clean:
        return True
    
    # Check if expected is contained in predicted (for cases where model adds extra text)
    if expected_clean in predicted_clean:
        return True
    
    # Extract numbers from predicted text and check if expected number is present
    import re
    numbers_in_pred = re.findall(r'\d+', predicted_clean)
    if expected_clean in numbers_in_pred:
        return True
    
    return False

def load_dataset(file_path):
    """Load the needle-in-haystack dataset from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def evaluate_model(model, tokenizer, dataset, device, method, config=None, model_name=None, window=None, budget=None):
    """Evaluate the model on the needle-in-haystack task with budget settings."""
    results = defaultdict(list)
    
    # Create directory for saving results
    os.makedirs("result_txt/needle", exist_ok=True)
    if method == "full":
        result_file = f"result_txt/needle/{model_name}_{method}.jsonl"
    else:
        result_file = f"result_txt/needle/{model_name}_{method}_window{window}_budget{budget}.jsonl"
    
    # Open file in write mode to overwrite any existing content
    with open(result_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(dataset, desc="Evaluating samples"):
            # Get the prompt and expected answer
            prompt = sample["prompt"]
            expected_answer = sample["answer"]
            needle_position = sample["needle_position"]
            total_tokens = sample["total_tokens"]
            
            # Add [INST] tags for Llama models
            if "llama" in model_name.lower():
                prompt = f"[INST]{prompt}[/INST]"
            
            # Initialize cache with budget settings if provided
            if config:
                model.init_cache(config)

            # Tokenize the input
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(torch.bfloat16).to(model.device)
            
            context_length = input_ids.shape[-1]
            
            # Generate response
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=64,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]

            # Decode the response
            model_answer = tokenizer.decode(outputs[context_length:], skip_special_tokens=True).strip()

            # Check exact match
            exact_match = check_exact_match(expected_answer, model_answer)
            
            # Record the result
            result_entry = {
                "expected": expected_answer,
                "model_answer": model_answer,
                "exact_match": exact_match
            }
            
            results[(total_tokens, needle_position)].append(result_entry)
            
            # Save to JSONL file
            jsonl_entry = {
                "sentence": model_answer,
                "position": needle_position,
                "length": total_tokens,
                "expected_answer": expected_answer,
                "exact_match": exact_match
            }
            
            f.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')
    
    return results

def calculate_metrics(results):
    """Calculate accuracy metrics for each needle position and context length."""
    metrics = {}
    
    for (total_tokens, position), samples in results.items():
        total_count = len(samples)
        
        # Count exact matches
        exact_match_count = sum(1 for sample in samples if sample.get("exact_match", False))
        
        # Calculate accuracy as exact match rate
        accuracy = exact_match_count / total_count if total_count > 0 else 0.0
        
        metrics[(total_tokens, position)] = {
            "accuracy": accuracy,
            "correct_count": exact_match_count,
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
    
    # Set axis labels with larger font size
    plt.xlabel('Context Length (tokens)', fontsize=30)
    plt.ylabel('Needle Position (%)', fontsize=30)
    
    # Set tick labels with larger font size and rotate x-axis labels
    plt.xticks(np.arange(len(context_lengths)), context_lengths, fontsize=26, rotation=45)
    plt.yticks(np.arange(len(positions)), positions, fontsize=26)
    
    plt.tight_layout()
    
    # Save the heatmap with high DPI for better quality
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_file}")

def main(args):
    set_seed(42)
    
    # Set GPU environment
    gpus = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    
    # Initialize budget and hyperparameter lists
    datasets = args.dataset
    budget_list = args.budget
    methods = args.method
    models = args.model
    window_list = args.window
    
    # Check and extend list lengths
    max_len = len(datasets) * len(budget_list) * len(methods) * len(models) * len(window_list)
    
    # Create directory for saving results
    os.makedirs("result_json/needle", exist_ok=True)
    os.makedirs("plots/needle", exist_ok=True)
    
    # Dictionary to store all results
    all_results = {
        "experiment_info": {
            "models": models,
            "datasets": datasets,
            "budgets": budget_list,
            "methods": methods,
            "windows": window_list
        },
        "results": {}
    }
    
    cur_idx = 0
    for model_name in models:
        # Normalize model name
        model_name = model_name.split("_")[0].lower()
        
        # Prepare device, model, and tokenizer
        device = f"cuda:{args.gpus[0]}"  # Use first GPU for device reference
        
        # Load model and tokenizer using the utility function
        print(f"Loading model: {model_name}")
        model, tokenizer = load_model(model_name, args.gpus)
        print("Model loaded successfully!")

        for dataset in datasets:
            dataset_name = os.path.basename(dataset).split('.')[0]
            
            # Load dataset
            dataset_data = load_dataset(dataset)

            # Nested loops: window → budget → method
            for cur_window in window_list:
                for cur_budget in budget_list:
                    for cur_method in methods:
                        cur_idx += 1
                        
                        # Create compression config
                        config = CompressionConfig()
                        config["compression_method"] = cur_method
                        config["observation_window"] = cur_window
                        config["total_budget"] = cur_budget
                        config["a"] = 10
                        config["b"] = cur_window

                        # Evaluate with current configuration
                        results = evaluate_model(
                            model=model,
                            tokenizer=tokenizer,
                            dataset=dataset_data,
                            device=device,
                            method=cur_method,
                            config=config,
                            model_name=model_name,
                            window=cur_window,
                            budget=cur_budget
                        )

                        # Calculate metrics
                        metrics = calculate_metrics(results)
                        
                        # Create heatmap
                        if cur_method == "full":
                            output_file = f"plots/needle/needle_heatmap_{model_name}_full.png"
                        else:
                            output_file = f"plots/needle/needle_heatmap_{model_name}_{cur_method}_window{cur_window}_budget{cur_budget}.png"
                        
                        create_heatmap(metrics, output_file)
                        
                        # Store results in the dictionary
                        result_key = f"{model_name}_{dataset_name}_{cur_method}_window{cur_window}_budget{cur_budget}"
                        
                        all_results["results"][result_key] = {
                            "model": model_name,
                            "dataset": dataset_name,
                            "method": cur_method,
                            "window": cur_window,
                            "budget": cur_budget,
                            "metrics": {
                                f"{length}_{position}": {
                                    "accuracy": metrics[(length, position)]["accuracy"],
                                    "correct_count": metrics[(length, position)]["correct_count"],
                                    "total_count": metrics[(length, position)]["total_count"]
                                }
                                for length, position in metrics.keys()
                            }
                        }
                        
                        # Print summary
                        print(f"\nConfig {cur_idx}/{max_len} | model={model_name}, dataset={dataset_name}, method={cur_method}, window={cur_window}, budget={cur_budget}")
                        print("Position | Accuracy | Correct/Total")
                        print("-" * 40)
                        for position in sorted(set(k[1] for k in metrics.keys())):
                            # Calculate average accuracy across all context lengths for this position
                            position_samples = [(k[0], v) for k, v in metrics.items() if k[1] == position]
                            total_correct = sum(sample[1]["correct_count"] for sample in position_samples)
                            total_samples = sum(sample[1]["total_count"] for sample in position_samples)
                            avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
                            print(f"{position:3.2f} | {avg_accuracy:.2%} | {total_correct}/{total_samples}")
        
        # Clear GPU memory after processing each model
        del model
        torch.cuda.empty_cache()
    
    # Save all results to a JSON file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"result_json/needle/needle_results_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {result_file}")

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Evaluate model predictions with various budgets on needle-in-haystack task.")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument("--model", type=str, nargs='+', default=["llama2"], choices=["llama", "llama2", "llama3", "opt", "qwen2"])
    parser.add_argument("--dataset", type=str, nargs='+', default=["datasets/needle_dataset.jsonl"])
    parser.add_argument("--budget", type=int, nargs='+', default=[128], help="Total budget for compression")
    parser.add_argument("--method", type=str, nargs='+', default=["snap"], help="Compression method (full, a2sf, h2o, snap, sigmoid)")
    parser.add_argument("--window", type=int, nargs='+', default=[16], help="Observation window size")
    
    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
