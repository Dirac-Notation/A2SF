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

from utils import load_configs, load_model, set_seed
from RL.config import A2SFRLConfig
from RL.policy import A2SFPolicy
from RL.features import ContextEncoder


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

def load_rl_policy(checkpoint_path, device):
    """Load RL policy from checkpoint"""
    print(f"Loading RL policy from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load checkpoint with weights_only=False to allow custom classes
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config from checkpoint
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        # Create default config if not found in checkpoint
        config = A2SFRLConfig()
        print("Warning: Config not found in checkpoint, using default config")
    
    # Initialize context encoder
    context_encoder = ContextEncoder(
        model_name=config.sentence_transformer_model,
        device=device,
        context_window=config.context_window,
        max_context=config.max_context
    )
    
    # Initialize policy
    policy = A2SFPolicy(state_dim=config.max_context).to(device)
    
    # Load policy weights
    if "policy_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"Loaded policy from iteration {checkpoint.get('iteration', 'unknown')}")
    else:
        raise ValueError("Policy state dict not found in checkpoint")
    
    policy.eval()  # Set to evaluation mode
    
    return policy, context_encoder, config

def get_rl_action(policy, context_encoder, prompt, model_name, device):
    """Get RL action (forgetting factor) for given prompt"""
    # Encode context
    context_embedding = context_encoder.encode_context(prompt)
    
    # Build state
    state = context_embedding.to(device, dtype=torch.float32)
    
    # Get action from policy (no exploration during inference)
    with torch.no_grad():
        action, _, _ = policy.act(state)
    
    # action is a tuple of (a, b) tensors
    # For a2sf compression, we use a as forgetting_factor
    a = action[0].item() if isinstance(action[0], torch.Tensor) else action[0]
    return a

def evaluate_model(model, tokenizer, dataset, device, method, config=None, rl_policy=None, context_encoder=None, model_name=None):
    """Evaluate the model on the needle-in-haystack task with budget settings."""
    results = defaultdict(list)
    
    # Create directory for saving results
    os.makedirs("result_txt/needle", exist_ok=True)
    if rl_policy and context_encoder:
        result_file = f"result_txt/needle/{method}_RL.jsonl"
    else:
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
            if rl_policy and context_encoder:
                # Use RL policy to determine forgetting factor
                forgetting_factor = get_rl_action(rl_policy, context_encoder, prompt, model_name, device)
                
                # Create compression config with RL-determined forgetting factor
                from utils import CompressionConfig
                rl_config = CompressionConfig()
                rl_config.compression_method = "a2sf"
                rl_config.total_budget = 128
                rl_config.layerwise_ratios = [1.0 for i in range(32)]
                rl_config.local_ratios = 0.125
                rl_config.forgetting_factors = [forgetting_factor for i in range(32)]
                
                model.init_cache(rl_config)
            elif config:
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

            # Calculate LCS ratio
            lcs_ratio = calculate_lcs_ratio(expected_answer, model_answer)
            
            # Record the result
            result_entry = {
                "expected": expected_answer,
                "model_answer": model_answer,
                "lcs_ratio": lcs_ratio
            }
            
            # Add forgetting factor if using RL
            if rl_policy and context_encoder:
                result_entry["forgetting_factor"] = forgetting_factor
            
            results[(total_tokens, needle_position)].append(result_entry)
            
            # Save to JSONL file
            jsonl_entry = {
                "sentence": model_answer,
                "position": needle_position,
                "length": total_tokens,
                "expected_answer": expected_answer,
                "lcs_ratio": lcs_ratio
            }
            
            # Add forgetting factor if using RL
            if rl_policy and context_encoder:
                jsonl_entry["forgetting_factor"] = forgetting_factor
            
            f.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')
    
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

    # For RL mode, we don't need budget/method combinations
    if args.use_rl:
        budget_list = [0]  # Dummy budget for RL
        methods = ["RL"]   # Dummy method for RL
    
    # Check and extend list lengths
    max_len = len(datasets) * len(budget_list) * len(methods) * len(models)
    
    # Create directory for saving results
    os.makedirs("result_json/needle", exist_ok=True)
    os.makedirs("plots/needle", exist_ok=True)
    
    # Dictionary to store all results
    all_results = {
        "experiment_info": {
            "models": models,
            "datasets": datasets,
            "budgets": budget_list,
            "methods": methods
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
        
        # Load RL policy if requested
        rl_policy = None
        context_encoder = None
        rl_config = None
        if args.use_rl:
            if not args.rl_checkpoint:
                raise ValueError("--rl_checkpoint is required when --use_rl is specified")
            
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                rl_policy, context_encoder, rl_config = load_rl_policy(args.rl_checkpoint, device)
                
                # Override config parameters if provided
                if hasattr(args, 'sentence_transformer_model'):
                    rl_config.sentence_transformer_model = args.sentence_transformer_model
                if hasattr(args, 'context_window'):
                    rl_config.context_window = args.context_window
                
                print(f"RL Policy loaded successfully")
                print(f"Config: {rl_config}")
            except Exception as e:
                print(f"Error loading RL policy: {e}")
                print("Falling back to standard evaluation mode")
                args.use_rl = False

        for dataset in datasets:
            dataset_name = os.path.basename(dataset).split('.')[0]
            
            # Load dataset
            dataset_data = load_dataset(dataset)

            # Nested loops: budget → method
            for cur_budget in budget_list:
                for cur_method in methods:
                    cur_idx += 1
                    
                    # Load compression config (skip if using RL)
                    config = None
                    if not args.use_rl:
                        config = load_configs(args.config_file, cur_method, cur_budget, "Single-doc QA")  # Default task for needle

                    # Evaluate with current configuration
                    results = evaluate_model(
                        model=model,
                        tokenizer=tokenizer,
                        dataset=dataset_data,
                        device=device,
                        method=cur_method,
                        config=config,
                        rl_policy=rl_policy,
                        context_encoder=context_encoder,
                        model_name=model_name
                    )

                    # Calculate metrics
                    metrics = calculate_metrics(results)
                    
                    # Create heatmap
                    if args.use_rl:
                        output_file = f"plots/needle/needle_heatmap_{model_name}_RL.png"
                    elif cur_method == "full":
                        output_file = f"plots/needle/needle_heatmap_{model_name}_full.png"
                    else:
                        output_file = f"plots/needle/needle_heatmap_{model_name}_{cur_method}_budget{cur_budget}.png"
                    
                    create_heatmap(metrics, output_file)
                    
                    # Store results in the dictionary
                    if args.use_rl:
                        result_key = f"{model_name}_{dataset_name}_RL"
                        method_info = "RL"
                        budget_info = "RL"
                    else:
                        result_key = f"{model_name}_{dataset_name}_{cur_method}_{cur_budget}"
                        method_info = cur_method
                        budget_info = cur_budget
                    
                    all_results["results"][result_key] = {
                        "model": model_name,
                        "dataset": dataset_name,
                        "method": method_info,
                        "budget": budget_info,
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
                    if args.use_rl:
                        print(f"\nConfig {cur_idx}/{max_len} | model={model_name}, dataset={dataset_name}, method=RL")
                    else:
                        print(f"\nConfig {cur_idx}/{max_len} | model={model_name}, dataset={dataset_name}, method={cur_method}, budget={cur_budget}")
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
    if args.use_rl:
        result_file = f"result_json/needle/needle_results_RL_{timestamp}.json"
    else:
        result_file = f"result_json/needle/needle_results_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {result_file}")

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Evaluate model predictions with various budgets on needle-in-haystack task.")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument("--model", type=str, nargs='+', default=["llama2"], choices=["llama", "llama2", "llama3", "opt", "qwen2"])
    parser.add_argument("--dataset", type=str, nargs='+', default=["datasets/needle_dataset.jsonl"])
    parser.add_argument("--budget", type=int, nargs='+', default=[100])
    parser.add_argument("--method", type=str, nargs='+', default=["a2sf"])
    parser.add_argument("--config_file", type=str, default="config/compression_configs.json", help="Path to compression config file")
    
    # RL-related arguments
    parser.add_argument("--use_rl", action="store_true", help="Use RL policy for compression")
    parser.add_argument("--rl_checkpoint", type=str, help="Path to RL model checkpoint (.pt file)")
    parser.add_argument("--sentence_transformer_model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model for context encoding")
    parser.add_argument("--context_window", type=int, default=64, help="Context window size for RL state encoding")
    
    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args()
    
    # Example usage:
    # Standard evaluation:
    # python evaluate_needle.py --model llama2 --dataset datasets/needle_dataset.jsonl --method a2sf --budget 100
    # 
    # RL evaluation:
    # python evaluate_needle.py --model llama2 --dataset datasets/needle_dataset.jsonl --use_rl --rl_checkpoint path/to/checkpoint.pt
    
    main(args)
