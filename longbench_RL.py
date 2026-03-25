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
from longbench_eval import dataset2metric

# Import evaluation functions from longbench_eval.py
from longbench_eval import evaluate_results

# ============================================================================
# Prediction Functions (from longbench_pred_RL.py)
# ============================================================================

TASK2DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "task2dataset.json")
with open(TASK2DATASET_PATH, "r", encoding="utf-8") as f:
    TASK_TO_DATASETS = json.load(f)
TASK_LIST = list(TASK_TO_DATASETS.keys())

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="LongBench end-to-end evaluation with RL-trained A2SF model")
    parser.add_argument('--model', type=str, required=True, choices=["llama", "llama2", "llama3", "opt"])
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--task', type=int, nargs='*', default=None, help="List of task numbers (0-5). If not specified, all tasks will be executed.")
    parser.add_argument('--datasets', type=str, nargs='*', default=None, help="List of specific dataset names to process. If specified, only these datasets will be processed (ignoring --task).")
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


def resolve_selected_datasets(args):
    if args.datasets is not None:
        print(f"Processing specified datasets: {args.datasets}")
        return args.datasets

    if args.task is None:
        selected_tasks = TASK_LIST
    else:
        selected_tasks = []
        for task_num in args.task:
            if 0 <= task_num < len(TASK_LIST):
                selected_tasks.append(TASK_LIST[task_num])
            else:
                print(f"Warning: Task number {task_num} is out of range (0-{len(TASK_LIST)-1}), skipping")

    selected_datasets = []
    for task in selected_tasks:
        selected_datasets.extend(TASK_TO_DATASETS[task])
    return selected_datasets


def load_rl_policy(checkpoint_path, device, target_model, target_tokenizer):
    """학습(A2SFTrainer)과 동일한 구조로 policy·encoder를 만들고 체크포인트를 로드한다."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", A2SFRLConfig())
    if not isinstance(config, A2SFRLConfig):
        config = A2SFRLConfig()

    dev = device if isinstance(device, torch.device) else torch.device(device)

    context_encoder = AttentionEncoder(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        device=str(dev),
        output_dim=-1,
        num_query_tokens=16,
    ).to(dev)

    state_dim = int(context_encoder.output_dim)
    metric_heads = sorted({fn.__name__ for fn in dataset2metric.values()})

    policy = NeuralUCBPolicy(
        state_dim=state_dim,
        a_values=config.a_values,
        b_values=config.b_values,
        metric_heads=metric_heads,
    ).to(dev)

    if "policy_state_dict" not in checkpoint:
        raise ValueError("Policy state dict not found in checkpoint")
    policy.load_state_dict(checkpoint["policy_state_dict"])
    print(f"Loaded RL policy from iteration {checkpoint.get('iteration', 'unknown')}")

    policy.eval()
    context_encoder.eval()
    return policy, context_encoder, config


def get_rl_action(
    policy,
    context_encoder,
    prompt,
    generation_length,
    token_budget,
    device,
    task_type=None,
    dataset=None,
):
    """Get RL action (a, b) for sigmoid cache from given prompt"""
    # Encode context with generation_length and token_budget
    context_embedding = context_encoder.encode_context(
        prompt,
        generation_length,
        token_budget,
        task_type=task_type,
        dataset=dataset,
    )
    
    # Build state (ensure it's on the correct device and dtype)
    state = context_embedding.to(device, dtype=torch.float32)
    
    metric_type = "qa_f1_score"
    metric_fn = dataset2metric.get(str(dataset or "").lower())
    if metric_fn is not None:
        metric_type = metric_fn.__name__

    # Inference-time action selection (pure exploitation).
    with torch.no_grad():
        (a_tensor, b_tensor), _ = policy.act(state, metric_type=metric_type)
    
    # Extract scalar a, b values
    if isinstance(a_tensor, torch.Tensor):
        a_val = float(a_tensor.view(-1)[0].item())
    else:
        a_val = float(a_tensor)
    if isinstance(b_tensor, torch.Tensor):
        b_val = float(b_tensor.view(-1)[0].item())
    else:
        b_val = float(b_tensor)
    
    return a_val, b_val

def get_pred_rl(data, max_length, max_gen, dataset, model, tokenizer, out_path, model_name, 
                rl_policy, context_encoder, rl_config, device, budget):
    """Generate predictions using RL-determined sigmoid cache parameters (a, b)"""
    for json_obj in tqdm(data):
        prompt = json_obj["input_prompt"]

        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                    tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            if "llama" in model_name:
                prompt = f"[INST]{prompt}[/INST]"

        # RL action은 실제 모델 입력 prompt와 동일한 텍스트로 추론
        a_val, b_val = get_rl_action(
            rl_policy,
            context_encoder,
            prompt,
            max_gen,
            budget,
            device,
            task_type=json_obj.get("task_type"),
            dataset=dataset,
        )
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt")
        
        input_ids = input.input_ids.to(model.device)
        attention_mask = input.attention_mask.to(torch.bfloat16).to(model.device)
        
        context_length = input_ids.shape[-1]
        
        # Get number of layers from model config
        num_layers = model.config.num_hidden_layers
        
        # Create compression config with RL-determined sigmoid cache parameters (a, b)
        from utils import CompressionConfig
        config = CompressionConfig()
        config.compression_method = "sigmoid"
        config.total_budget = budget
        config.layerwise_ratios = [1.0 for _ in range(num_layers)]
        config.local_ratios = 0.125
        # Sigmoid cache parameters
        config.a = float(a_val)
        config.b = float(b_val)
        
        model.init_cache(config)
        
        with torch.inference_mode():
            if dataset == "samsum":
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
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
                "a": a_val,  # Sigmoid cache steepness parameter
                "b": b_val,  # Sigmoid cache shift parameter
            }, f, ensure_ascii=False)
            f.write('\n')

if __name__ == '__main__':
    set_seed(42)
    args = parse_args()
    
    model_name = args.model
    model_name = model_name.split("_")[0].lower()
    max_length = json.load(open("config/model2maxlen.json", "r"))[model_name]
    
    model, tokenizer = load_model(model_name)
    
    first_layer_device = next(model.model.layers[0].parameters()).device
    device = first_layer_device
    print(f"Loading RL policy from: {args.rl_checkpoint}")
    rl_policy, context_encoder, rl_config = load_rl_policy(
        args.rl_checkpoint, device, model, tokenizer
    )
    
    print(f"RL Policy loaded successfully")
    print(f"Config: {rl_config}")

    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    if not os.path.exists("result_txt/pred"):
        os.makedirs("result_txt/pred")
    
    output_dir = None
    
    selected_datasets = resolve_selected_datasets(args)
    
    # Process each dataset
    for dataset in selected_datasets:
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

