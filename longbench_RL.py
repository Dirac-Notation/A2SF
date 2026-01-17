import os
import json
from tqdm import tqdm
import argparse
import torch
import sys

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_model, set_seed
from RL.main import A2SFRLConfig
from RL.policy import NeuralUCBPolicy
from RL.env import AttentionEncoder

# Import metrics and evaluation functions from longbench.py
from longbench import (
    dataset2metric,
    data_group,
    scorer,
    calculate_group_averages,
    evaluate_results
)

# ============================================================================
# Prediction Functions (from longbench_pred_RL.py)
# ============================================================================

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="LongBench end-to-end evaluation with RL-trained A2SF model")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument('--model', type=str, required=True, choices=["llama", "llama2", "llama3", "opt"])
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--task', type=int, nargs='*', default=None, help="List of task numbers (0-5). If not specified, all tasks will be executed.")
    parser.add_argument('--rl_checkpoint', type=str, required=True, help="Path to RL model checkpoint (.pt file)")
    parser.add_argument('--skip_eval', action='store_true', help="Skip evaluation after prediction")
    return parser.parse_args(args)

def load_jsonl_file(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_rl_policy(checkpoint_path, device, target_model, target_tokenizer):
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
    
    # Initialize attention encoder using target model's first layer, first head parameters
    # AttentionEncoder is frozen and uses target model's parameters, so no need to load from checkpoint
    context_encoder = AttentionEncoder(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        device=device,
        output_dim=8192,
        num_query_tokens=16
    ).to(device)
    
    # State dimension is fixed to 8192 (output_dim of AttentionEncoder)
    state_dim = 8192
    
    # Initialize policy with config values
    policy = NeuralUCBPolicy(
        state_dim=state_dim,
        a_values=config.a_values,
        b_values=config.b_values
    ).to(device)
    
    # Load policy weights
    if "policy_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"Loaded policy from iteration {checkpoint.get('iteration', 'unknown')}")
    else:
        raise ValueError("Policy state dict not found in checkpoint")
    
    policy.eval()  # Set to evaluation mode
    context_encoder.eval()  # Set to evaluation mode
    
    return policy, context_encoder, config

def get_rl_action(policy, context_encoder, prompt, dataset, model_name, device, ucb_beta=1.0):
    """Get RL action (a, b) for sigmoid cache from given prompt"""
    # Encode context
    context_embedding = context_encoder.encode_context(prompt)
    
    # Build state (ensure it's on the correct device and dtype)
    state = context_embedding.to(device, dtype=torch.float32)
    
    # Get action from policy using UCB
    with torch.no_grad():
        action, ucb_value = policy.act(state, beta=ucb_beta)
    
    # action is a tuple of (a, b) tensors
    # Extract scalar values
    a = action[0].item() if isinstance(action[0], torch.Tensor) else action[0]
    b = action[1].item() if isinstance(action[1], torch.Tensor) else action[1]
    
    return a, b

def get_pred_rl(data, max_length, max_gen, dataset, model, tokenizer, out_path, model_name, 
                rl_policy, context_encoder, rl_config, device, budget):
    """Generate predictions using RL-determined compression parameters"""
    for json_obj in tqdm(data):
        prompt = json_obj["input_prompt"]

        # Get RL action (a, b) for sigmoid cache from this prompt
        a, b = get_rl_action(
            rl_policy, context_encoder, prompt, dataset, model_name, device, 
            ucb_beta=rl_config.ucb_beta
        )

        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                    tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            if "llama" in model_name:
                prompt = f"[INST]{prompt}[/INST]"
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt")
        
        input_ids = input.input_ids.to(model.device)
        attention_mask = input.attention_mask.to(torch.bfloat16).to(model.device)
        
        context_length = input_ids.shape[-1]
        
        # Get number of layers from model config
        num_layers = model.config.num_hidden_layers
        
        # Create compression config with RL-determined sigmoid parameters (a, b)
        from utils import CompressionConfig
        config = CompressionConfig()
        config.compression_method = "sigmoid"
        config.total_budget = budget
        config.layerwise_ratios = [1.0 for i in range(num_layers)]
        config.local_ratios = 0.125
        config.a = a
        config.b = b
        
        model.init_cache(config)
        
        with torch.inference_mode():
            if dataset == "samsum":
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                    pad_token_id=tokenizer.eos_token_id,
                )[0]
            else:
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]
        
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        
        # Save prediction with RL sigmoid parameters
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({
                "pred": pred, 
                "answers": json_obj["answers"], 
                "all_classes": json_obj["all_classes"], 
                "length": json_obj["length"],
                "a": a,  # Add RL sigmoid parameter a to output
                "b": b   # Add RL sigmoid parameter b to output
            }, f, ensure_ascii=False)
            f.write('\n')

if __name__ == '__main__':
    set_seed(42)
    args = parse_args()
    
    gpus = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    
    model_name = args.model
    model_name = model_name.split("_")[0].lower()
    max_length = json.load(open("config/model2maxlen.json", "r"))[model_name]
    
    # Load base model
    model, tokenizer = load_model(model_name, args.gpus)
    
    # Load RL policy (requires target model and tokenizer for AttentionEncoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rl_policy, context_encoder, rl_config = load_rl_policy(args.rl_checkpoint, device, model, tokenizer)
    
    print(f"RL Policy loaded successfully")
    print(f"Config: {rl_config}")

    # Task 이름 리스트
    task_list = [
        "Code Complete",
        "Few Shot",
        "Single-doc QA",
        "Multi-doc QA",
        "Passage Retrieval",
        "Summarization",
    ]
    
    # Task 번호를 task 이름으로 변환
    if args.task is None:
        selected_tasks = task_list
    else:
        selected_tasks = []
        for task_num in args.task:
            if 0 <= task_num < len(task_list):
                selected_tasks.append(task_list[task_num])
            else:
                print(f"Warning: Task number {task_num} is out of range (0-{len(task_list)-1}), skipping")
    
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    if not os.path.exists("result_txt/pred"):
        os.makedirs("result_txt/pred")
    
    output_dir = None
    
    for task in selected_tasks:
        datasets = data_group[task]
        
        for dataset in datasets:
            print(f"\nProcessing dataset: {dataset}")
            
            jsonl_path = f"datasets/longbench/{dataset}.jsonl"
            if not os.path.exists(jsonl_path):
                print(f"Warning: {jsonl_path} not found, skipping {dataset}")
                continue
            data = load_jsonl_file(jsonl_path)
            
            # Create output directory with RL indicator
            output_dir = f"result_txt/pred/{args.model}_sigmoid_{args.budget}_RL"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            out_path = f"{output_dir}/{dataset}.jsonl"
            
            # Clear existing file
            if os.path.exists(out_path):
                os.remove(out_path)
            
            max_gen = dataset2maxlen[dataset]
            
            get_pred_rl(data, max_length, max_gen, dataset, model, tokenizer, out_path, 
                       model_name, rl_policy, context_encoder, rl_config, device, args.budget)
            
            print(f"Completed {dataset} with RL policy")
    
    # Evaluate results if not skipped
    if not args.skip_eval and output_dir and os.path.exists(output_dir):
        evaluate_results(output_dir)
    
    print("\nRL LongBench evaluation completed!")

