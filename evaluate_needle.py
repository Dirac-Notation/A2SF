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
from RL.a2sf_model import ModelConfig
from RL.agent.neural_ucb_agent import NeuralUCBAgent
from RL.env import AttentionEncoder
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

def load_rl_agent(checkpoint_path, device, target_model, target_tokenizer):
    """Load RL agent and frozen metadata encoder for inference."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg_any = checkpoint.get("model_config", None) or checkpoint.get("config", None)
    config = ModelConfig(model="llama3")
    if model_cfg_any is not None:
        if isinstance(model_cfg_any, dict):
            if "a_values" in model_cfg_any:
                config.a_values = model_cfg_any["a_values"]
            if "b_values" in model_cfg_any:
                config.b_values = model_cfg_any["b_values"]
        else:
            if hasattr(model_cfg_any, "a_values"):
                config.a_values = model_cfg_any.a_values
            if hasattr(model_cfg_any, "b_values"):
                config.b_values = model_cfg_any.b_values

    context_encoder = AttentionEncoder(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        device=device,
        output_dim=-1,
        num_query_tokens=16,
    ).to(device)

    state_dim = int(context_encoder.output_dim)
    agent = NeuralUCBAgent(
        state_dim=state_dim,
        a_values=config.a_values,
        b_values=config.b_values,
        metric_heads=METRIC_HEADS,
    ).to(device)

    # Load agent weights (supports legacy policy_state_dict checkpoints)
    if "agent_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["agent_state_dict"])
    elif "policy_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["policy_state_dict"])
    else:
        raise ValueError("Checkpoint missing agent_state_dict (and legacy policy_state_dict).")

    agent.eval()
    context_encoder.eval()
    return agent, context_encoder, config


def get_rl_action(
    agent,
    context_encoder,
    prompt,
    generation_length,
    token_budget,
    device,
    task_type=None,
    dataset=None,
    metric_type="qa_f1_score",
):
    state = context_encoder.encode_context(
        prompt,
        generation_length,
        token_budget,
        task_type=task_type,
        dataset=dataset,
    ).to(device, dtype=torch.float32)
    with torch.no_grad():
        (a_tensor, b_tensor), _ = agent.act(state, metric_type=metric_type)
    a_val = float(a_tensor.view(-1)[0].item()) if isinstance(a_tensor, torch.Tensor) else float(a_tensor)
    b_val = float(b_tensor.view(-1)[0].item()) if isinstance(b_tensor, torch.Tensor) else float(b_tensor)
    return a_val, b_val


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
    rl_agent=None,
    context_encoder=None,
):
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

            run_config = config
            if run_config and method == "sigmoid" and rl_agent is not None and context_encoder is not None:
                token_budget = int(run_config.total_budget) if run_config.total_budget is not None else int(budget)
                a_val, b_val = get_rl_action(
                    rl_agent,
                    context_encoder,
                    prompt,
                    generation_length=64,
                    token_budget=token_budget,
                    device=device,
                    task_type=sample.get("task_type"),
                    dataset=sample.get("dataset"),
                    metric_type=sample.get("metric_type") or "qa_f1_score",
                )
                run_config = CompressionConfig()
                for k, v in config.items():
                    run_config[k] = v
                run_config.a = float(a_val)
                run_config.b = float(b_val)
            
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

        rl_agent = None
        context_encoder = None
        rl_config = None
        if args.rl_checkpoint:
            first_layer_device = next(model.model.layers[0].parameters()).device
            rl_agent, context_encoder, rl_config = load_rl_agent(
                args.rl_checkpoint, first_layer_device, model, tokenizer
            )
            print(f"Loaded RL agent from: {args.rl_checkpoint}")

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
                            rl_agent=rl_agent,
                            context_encoder=context_encoder,
                        )

                        metrics = calculate_metrics(results)
                        
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
    parser.add_argument("--model", type=str, nargs='+', default=["llama2"], choices=["llama", "llama2", "llama3", "opt", "qwen2"])
    parser.add_argument("--dataset", type=str, nargs='+', default=["datasets/needle_dataset.jsonl"])
    parser.add_argument("--budget", type=int, nargs='+', default=[128], help="Total budget for compression")
    parser.add_argument("--method", type=str, nargs='+', default=["snap"], help="Compression method (full, a2sf, h2o, snap, sigmoid)")
    parser.add_argument("--window", type=int, nargs='+', default=[16], help="Observation window size")
    parser.add_argument("--rl_checkpoint", type=str, default=None, help="Optional RL checkpoint path. If set, sigmoid method uses RL-selected a,b per sample.")
    
    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)