import os
import json
from tqdm import tqdm
import argparse
import torch
import sys

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_configs, load_model, set_seed
from RL.config import A2SFRLConfig
from RL.policy import A2SFPolicy
from RL.features import ContextEncoder

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="LongBench evaluation with RL-trained A2SF model")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument('--model', type=str, required=True, choices=["llama", "llama2", "llama3", "opt"])
    parser.add_argument('--method', type=str, default="full")
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--task', type=str, nargs='+', required=True, 
                       choices=["Code Complete", "Few Shot", "Single-doc QA", "Multi-doc QA", "Passage Retrieval", "Summarization"])
    parser.add_argument('--rl_checkpoint', type=str, required=True, 
                       help="Path to RL model checkpoint (.pt file)")
    parser.add_argument('--sentence_transformer_model', type=str, default="all-MiniLM-L6-v2",
                       help="Sentence transformer model for context encoding")
    parser.add_argument('--context_window', type=int, default=64,
                       help="Context window size for RL state encoding")
    return parser.parse_args(args)

def load_jsonl_file(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
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
        device=device
    )
    
    # Calculate state dimension
    state_dim = context_encoder.embedding_dim
    
    # Initialize policy
    policy = A2SFPolicy(
        state_dim=state_dim,
        action_min=config.action_min,
        action_max=config.action_max
    ).to(device)
    
    # Load policy weights
    if "policy_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"Loaded policy from iteration {checkpoint.get('iteration', 'unknown')}")
    else:
        raise ValueError("Policy state dict not found in checkpoint")
    
    policy.eval()  # Set to evaluation mode
    
    return policy, context_encoder, config

def get_rl_action(policy, context_encoder, prompt, dataset, model_name, device):
    """Get RL action (forgetting factor) for given prompt"""
    # Encode context
    context_embedding = context_encoder.encode_context(prompt)
    
    # Build state
    state = context_embedding.to(device, dtype=torch.float32)
    
    # Get action from policy (no exploration during inference)
    with torch.no_grad():
        action, _, _ = policy.act(state)
    
    return action.item()

def get_pred_rl(data, max_length, max_gen, dataset, model, tokenizer, out_path, model_name, 
                rl_policy, context_encoder, rl_config, device):
    """Generate predictions using RL-determined compression parameters"""
    for json_obj in tqdm(data):
        prompt = json_obj["input_prompt"]

        # Get RL action (forgetting factor) for this prompt
        forgetting_factor = get_rl_action(
            rl_policy, context_encoder, prompt, dataset, model_name, device
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
        
        # Create compression config with RL-determined forgetting factor
        from utils import CompressionConfig
        config = CompressionConfig()
        config.compression_method = "a2sf"
        config.total_budget = 128
        config.layerwise_ratios = [1.0 for i in range(32)]
        config.local_ratios = 0.125
        config.forgetting_factors = [forgetting_factor for i in range(32)]
        
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
                )[0]
            else:
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
        
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        
        # Save prediction with RL forgetting factor
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({
                "pred": pred, 
                "answers": json_obj["answers"], 
                "all_classes": json_obj["all_classes"], 
                "length": json_obj["length"],
                "forgetting_factor": forgetting_factor  # Add RL forgetting factor to output
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
    
    # Load RL policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rl_policy, context_encoder, rl_config = load_rl_policy(args.rl_checkpoint, device)
    
    # Override config parameters if provided
    if hasattr(args, 'sentence_transformer_model'):
        rl_config.sentence_transformer_model = args.sentence_transformer_model
    if hasattr(args, 'context_window'):
        rl_config.context_window = args.context_window
    
    print(f"RL Policy loaded successfully")
    print(f"Action range: [{rl_config.action_min}, {rl_config.action_max}]")
    print(f"Context window: {rl_config.context_window}")
    print(f"Sentence transformer: {rl_config.sentence_transformer_model}")

    data_group = {
        "Code Complete": ["repobench-p", "lcc"],
        "Few Shot": ["trec", "triviaqa", "samsum"],
        "Single-doc QA": ["narrativeqa", "qasper", "multifieldqa_en"],
        "Multi-doc QA": ["hotpotqa", "2wikimqa", "musique"],
        "Summarization": ["gov_report", "qmsum", "multi_news"],
        "Passage Retrieval": ["passage_retrieval_en", "passage_count"],
    }
    
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    if not os.path.exists("result_txt/pred"):
        os.makedirs("result_txt/pred")
    
    for task in args.task:
        config = load_configs(args.config_file, args.method, args.budget, task)
        
        datasets = data_group[task]
        
        for dataset in datasets:
            print(f"\nProcessing dataset: {dataset}")
            
            jsonl_path = f"datasets/longbench/{dataset}.jsonl"
            if not os.path.exists(jsonl_path):
                print(f"Warning: {jsonl_path} not found, skipping {dataset}")
                continue
            data = load_jsonl_file(jsonl_path)
            
            # Create output directory with RL indicator
            output_dir = f"result_txt/pred/{args.model}_{args.method}_{args.budget}_RL"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            out_path = f"{output_dir}/{dataset}.jsonl"
            
            max_gen = dataset2maxlen[dataset]
            
            get_pred_rl(data, max_length, max_gen, dataset, model, tokenizer, out_path, 
                       model_name, rl_policy, context_encoder, rl_config, device)
            
            print(f"Completed {dataset} with RL policy")
    
    print("\nRL LongBench evaluation completed!")
    print(f"Results saved in: result_txt/pred/{args.model}_{args.method}_{args.budget}_RL/")
