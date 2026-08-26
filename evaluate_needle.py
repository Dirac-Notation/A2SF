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
import re

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_model, set_seed, CompressionConfig
from longbench_eval import dataset2metric

# Must match keys used when training NeuralUCBPolicy (RL/trainer.py).
METRIC_HEADS = sorted({fn.__name__ for fn in dataset2metric.values()})

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

def evaluate_model(
    model,
    tokenizer,
    dataset,
    device,
    method,
    config=None,
    model_name=None,
    window=None,
    budget=None,
):
    """Evaluate the model on the needle-in-haystack task with budget settings."""
    results = defaultdict(list)
    
    # Create directory for saving results
    os.makedirs("result_txt/needle", exist_ok=True)
    if method == "full":
        result_file = f"result_txt/needle/{model_name}_{method}.jsonl"
    elif method == "waits" and config is not None:
        abtag = f"a{config['a']}_b{config['b']}".replace(".", "_")
        result_file = f"result_txt/needle/{model_name}_{method}_{abtag}_budget{budget}.jsonl"
    else:
        result_file = f"result_txt/needle/{model_name}_{method}_window{window}_budget{budget}.jsonl"
    
    # Open file in write mode to overwrite any existing content
    with open(result_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(dataset, desc="Evaluating samples"):
            # Get the prompt and expected answer (supports new password/distractor schema)
            prompt = sample["prompt"]
            expected_answer = sample.get("answer") or sample.get("password")
            needle_position = float(sample.get("needle_position", sample.get("position_pct")))
            total_tokens = int(float(sample.get("total_tokens", sample.get("actual_tokens"))))

            # [INST] only if not already baked into the prompt (new dataset bakes it)
            if "llama" in model_name.lower() and "[INST]" not in prompt:
                prompt = f"[INST]{prompt}[/INST]"

            run_config = config

            # Initialize cache with budget settings if provided
            if run_config:
                model.init_cache(run_config)

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
                    num_logits_to_keep=1,
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
        exact_match_count = sum(1 for sample in samples if sample.get("exact_match", False))
        accuracy = exact_match_count / total_count if total_count > 0 else 0.0
        
        metrics[(total_tokens, position)] = {
            "accuracy": accuracy,
            "correct_count": exact_match_count,
            "total_count": total_count
        }
    
    return metrics

def create_heatmap(metrics, output_file):
    """Create a heatmap visualization of the results."""
    context_lengths = sorted(list(set(k[0] for k in metrics.keys())))
    positions = sorted(list(set(k[1] for k in metrics.keys())))
    
    heatmap_data = np.zeros((len(positions), len(context_lengths)))
    
    for i, position in enumerate(positions):
        for j, length in enumerate(context_lengths):
            if (length, position) in metrics:
                heatmap_data[i, j] = metrics[(length, position)]["accuracy"]
    
    plt.figure(figsize=(15, 10))
    im = plt.imshow(heatmap_data, cmap='RdYlGn', vmin=0.0, vmax=1.0)
    
    plt.xlabel('Context Length (tokens)', fontsize=30)
    plt.ylabel('Needle Position (%)', fontsize=30)
    
    plt.xticks(np.arange(len(context_lengths)), context_lengths, fontsize=26, rotation=45)
    plt.yticks(np.arange(len(positions)), positions, fontsize=26)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close() # Close to free memory
    print(f"Heatmap saved to {output_file}")

def main(args):
    set_seed(42)
    
    datasets = args.dataset
    budget_list = args.budget
    methods = args.method
    models = args.model
    window_list = args.window
    
    max_len = len(datasets) * len(budget_list) * len(methods) * len(models) * len(window_list)
    
    os.makedirs("result_json/needle", exist_ok=True)
    os.makedirs("plots/needle", exist_ok=True)
    
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
    for model_name_raw in models:
        model_name = model_name_raw.split("_")[0].lower()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model: {model_name}")
        model, tokenizer = load_model(model_name)
        print("Model loaded successfully!")

        for dataset_path in datasets:
            dataset_name = os.path.basename(dataset_path).split('.')[0]
            dataset_data = load_dataset(dataset_path)

            for cur_window in window_list:
                for cur_budget in budget_list:
                    for cur_method in methods:
                        cur_idx += 1
                        
                        config = CompressionConfig()
                        config["compression_method"] = cur_method
                        config["observation_window"] = cur_window
                        config["total_budget"] = cur_budget
                        config["a"] = 10
                        config["b"] = cur_window
                        # single fixed WAITS action override (sweep one (a,b))
                        if cur_method == "waits" and args.sigmoid_a is not None:
                            config["a"] = float(args.sigmoid_a)
                            config["b"] = float(args.sigmoid_b)

                        results = evaluate_model(
                            model=model,
                            tokenizer=tokenizer,
                            dataset=dataset_data,
                            device=device,
                            method=cur_method,
                            config=config,
                            model_name=model_name,
                            window=cur_window,
                            budget=cur_budget,
                        )

                        metrics = calculate_metrics(results)

                        # overall accuracy (micro-avg over all samples) + append to sweep CSV
                        tot_c = sum(m["correct_count"] for m in metrics.values())
                        tot_n = sum(m["total_count"] for m in metrics.values())
                        overall_acc = 100.0 * tot_c / tot_n if tot_n else 0.0
                        ab = f"a{config['a']}_b{config['b']}" if cur_method == "waits" else f"w{cur_window}"
                        os.makedirs("result_txt/needle", exist_ok=True)
                        with open("result_txt/needle/sweep_results.csv", "a") as cf:
                            cf.write(f"{model_name},{cur_method},{ab},{cur_budget},{overall_acc:.2f},{tot_c}/{tot_n}\n")
                        print(f"[NIAH] {model_name} {cur_method} {ab} b{cur_budget} -> OVERALL {overall_acc:.2f}% ({tot_c}/{tot_n})")

                        if cur_method == "full":
                            output_file = f"plots/needle/needle_heatmap_{model_name}_full.png"
                        else:
                            output_file = f"plots/needle/needle_heatmap_{model_name}_{cur_method}_window{cur_window}_budget{cur_budget}.png"
                        
                        create_heatmap(metrics, output_file)
                        
                        result_key = f"{model_name}_{dataset_name}_{cur_method}_window{cur_window}_budget{cur_budget}"
                        
                        all_results["results"][result_key] = {
                            "model": model_name,
                            "dataset": dataset_name,
                            "method": cur_method,
                            "window": cur_window,
                            "budget": cur_budget,
                            "metrics": {
                                f"{length}_{pos}": metrics[(length, pos)]
                                for length, pos in metrics.keys()
                            }
                        }
                        
                        print(f"\nConfig {cur_idx}/{max_len} | model={model_name}, dataset={dataset_name}, method={cur_method}, window={cur_window}, budget={cur_budget}")
                        print("Position | Accuracy | Correct/Total")
                        print("-" * 40)
                        for position in sorted(set(k[1] for k in metrics.keys())):
                            pos_samples = [v for k, v in metrics.items() if k[1] == position]
                            total_correct = sum(s["correct_count"] for s in pos_samples)
                            total_samples = sum(s["total_count"] for s in pos_samples)
                            avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
                            print(f"{position:3.2f} | {avg_accuracy:.2%} | {total_correct}/{total_samples}")
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_result_file = f"result_json/needle/needle_results_{timestamp}.json"
    with open(final_result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {final_result_file}")

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Evaluate model predictions on needle-in-haystack task.")
    parser.add_argument("--model", type=str, nargs='+', default=["llama3-8b"], choices=["llama3-1b", "llama3-8b", "qwen2", "mistral-7b"])
    parser.add_argument("--sigmoid_a", type=float, default=None, help="Override sigmoid a (single fixed WAITS action). Use with --method sigmoid.")
    parser.add_argument("--sigmoid_b", type=float, default=None, help="Override sigmoid b (single fixed WAITS action). Use with --method sigmoid.")
    parser.add_argument("--dataset", type=str, nargs='+', default=["datasets/needle_dataset.jsonl"])
    parser.add_argument("--budget", type=int, nargs='+', default=[128], help="Total budget for compression")
    parser.add_argument("--method", type=str, nargs='+', default=["snap"], help="Compression method (full, waits, h2o, snap, sigmoid)")
    parser.add_argument("--window", type=int, nargs='+', default=[16], help="Observation window size")

    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)