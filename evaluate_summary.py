import os
import torch
import json
import argparse
import sys

from rouge_score import rouge_scorer
from tqdm import tqdm

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_configs, load_model, set_seed, CompressionConfig
from RL.config import A2SFRLConfig
from RL.policy import A2SFPolicy
from RL.features import ContextEncoder

def load_datasets(
    dataset_path: str,
    tokenizer,
    model_name: str
):
    with open(dataset_path, "r") as f:
        datalines = f.readlines()
        articles = []
        
        for dataline in datalines:
            articles.append(json.loads(dataline))

    inputs = []
    answers = []
    output_indices = []

    for data in tqdm(articles, desc="Tokenizing"):
        input = data["article"]
        answer = data["summary_gt"]
        
        if "llama" in model_name.lower():
            input = f"[INST]{input}[/INST]"
        
        input_data = tokenizer(input, return_tensors="pt")
        output_ids = tokenizer(answer, return_tensors="pt").input_ids

        inputs.append(input_data)
        answers.append(answer)
        output_indices.append(output_ids)
    
    num_input_ids = sum([input_data.input_ids.numel() for input_data in inputs])/len(inputs)
    num_output_ids = sum([output_ids.numel() for output_ids in output_indices])/len(output_indices)
    
    print(f"Average input ids length : {num_input_ids:.2f}")
    print(f"Average output ids length : {num_output_ids:.2f}")
    
    return inputs, answers, output_indices

def load_rl_policy(checkpoint_path):
    """Load RL policy from checkpoint"""
    print(f"Loading RL policy from: {checkpoint_path}")
    device = "cuda:0"
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

def get_rl_action(policy, context_encoder, prompt, model_name):
    """Get RL action (forgetting factor) for given prompt"""
    # Encode context
    context_embedding = context_encoder.encode_context(prompt)
    
    # Build state
    state = context_embedding.to(dtype=torch.float32)
    
    # Get action from policy (no exploration during inference)
    with torch.no_grad():
        action, _, _ = policy.act(state)
    
    return action.item()

def evaluate_model(
    model,
    tokenizer,
    inputs,
    answers,
    output_indices,
    device,
    dataset_name,
    model_name,
    budget,
    method,
    desc="Generating",
    init_cache_fn=None,
    compression_config=None,
    rl_policy=None,
    context_encoder=None,
    use_rl=False
):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.eos_token_id
    predictions = []
    throughput_samples = []

    # Create directory for saving results
    os.makedirs("result_txt/summary", exist_ok=True)
    if use_rl:
        result_file = f"result_txt/summary/{dataset_name}_{model_name}_{budget}_{method}_RL.jsonl"
    else:
        result_file = f"result_txt/summary/{dataset_name}_{model_name}_{budget}_{method}.jsonl"

    # Open file in write mode to overwrite any existing content
    with open(result_file, 'w', encoding='utf-8') as f:
        for idx, input_data in enumerate(tqdm(inputs, desc=desc)):
            # Convert input data to proper format
            input_ids = input_data.input_ids.to(device)
            attention_mask = input_data.attention_mask.to(torch.bfloat16).to(device)
            
            # Get the original prompt for RL
            original_prompt = inputs[idx].input_ids
            prompt_text = tokenizer.decode(original_prompt[0], skip_special_tokens=True)

            # GPU timing events
            torch.cuda.synchronize()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()

            # Initialize cache if needed
            if use_rl and rl_policy and context_encoder:
                # Use RL policy to determine forgetting factor
                forgetting_factor = get_rl_action(rl_policy, context_encoder, prompt_text, model_name)
                
                # Create compression config with RL-determined forgetting factor
                rl_config = CompressionConfig()
                rl_config.compression_method = "a2sf"
                rl_config.total_budget = budget
                rl_config.layerwise_ratios = [1.0 for i in range(32)]
                rl_config.local_ratios = 0.125
                rl_config.forgetting_factors = [forgetting_factor for i in range(32)]
                
                init_cache_fn(rl_config)
            elif init_cache_fn and compression_config:
                init_cache_fn(compression_config)

            # Generate with proper input format and explicit pad_token_id
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # max_new_tokens=output_indices[idx].numel(),
                max_new_tokens=1,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=False
            )

            # Record timing
            end_evt.record()
            torch.cuda.synchronize()
            elapsed = start_evt.elapsed_time(end_evt) / 1000.0  # ms → s
            toks = gen_ids.shape[1] - input_ids.shape[1]
            throughput_samples.append(toks / elapsed if elapsed > 0 else 0)

            # Decode
            pred_text = tokenizer.decode(
                gen_ids[0, input_ids.shape[1]:],
                skip_special_tokens=True
            )
            predictions.append(pred_text)

            # Save to JSONL file
            result_entry = {
                "generated_summary": pred_text,
                "reference_summary": answers[idx],
                "input_length": input_ids.shape[1],
                "output_length": output_indices[idx].numel()
            }
            
            # Add RL forgetting factor if using RL
            if use_rl and rl_policy and context_encoder:
                result_entry["forgetting_factor"] = forgetting_factor
            
            f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

    # Calculate ROUGE scores
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for ref, pred in zip(answers, predictions):
        score = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(score[key].fmeasure)

    # Calculate averages
    avg_r1 = sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1'])
    avg_r2 = sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2'])
    avg_rL = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])
    avg_tp = sum(throughput_samples) / len(throughput_samples)

    return {
        'predictions': predictions,
        'rouge1': avg_r1,
        'rouge2': avg_r2,
        'rougeL': avg_rL,
        'throughput': avg_tp
    }

def main(args):
    set_seed(42)
    
    # Initialize budget and hyperparameter lists
    model_name = args.model
    datasets = args.dataset
    budget_list = args.budget
    methods = args.method

    # Check and extend list lengths
    max_len = len(datasets) * len(budget_list) * len(methods)
    
    # Set GPU environment
    gpus = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    
    # Prepare device, model, and tokenizer
    device = f"cuda:0"  # Use first GPU for device reference
    
    # Load model and tokenizer using the utility function
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(args.model, args.gpus)
    print("Model loaded successfully!")
    
    # Load RL policy if requested
    rl_policy = None
    context_encoder = None
    rl_config = None
    if args.use_rl:
        if not args.rl_checkpoint:
            raise ValueError("--rl_checkpoint is required when --use_rl is specified")

        device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rl_policy, context_encoder, rl_config = load_rl_policy(args.rl_checkpoint)
        
        # Override config parameters if provided
        if hasattr(args, 'sentence_transformer_model'):
            rl_config.sentence_transformer_model = args.sentence_transformer_model
        if hasattr(args, 'context_window'):
            rl_config.context_window = args.context_window
        
        print(f"RL Policy loaded successfully")
        print(f"Config: {rl_config}")
        
    cur_idx = 0
    for dataset in datasets:
        dataset_name = os.path.basename(dataset).split('.')[0]
        
        # Load dataset
        inputs, answers, output_indices = load_datasets(
            dataset_path=dataset,
            tokenizer=tokenizer,
            model_name=model_name
        )

        # Nested loops: budget → method
        for cur_budget in budget_list:
            for cur_method in methods:
                cur_idx += 1
                # Load compression config
                config = load_configs(args.config_file, cur_method, cur_budget, "Summarization")

                # Evaluate with current configuration
                results = evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=inputs,
                    answers=answers,
                    output_indices=output_indices,
                    device=device,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    budget=cur_budget,
                    method=cur_method,
                    desc=f"{model_name} - dataset={dataset_name}, method={cur_method}, budget={cur_budget}, cfg={cur_idx}/{max_len}",
                    init_cache_fn=model.init_cache,
                    compression_config=config,
                    rl_policy=rl_policy,
                    context_encoder=context_encoder,
                    use_rl=args.use_rl
                )

                # Print results
                print(f"Config {cur_idx}/{max_len} | dataset={dataset_name}, method={cur_method}, budget={cur_budget}")
                print(
                    f"  ROUGE-1: {results['rouge1']:.4f}, ROUGE-2: {results['rouge2']:.4f}, ROUGE-L: {results['rougeL']:.4f}\n"
                    f"  Throughput: {results['throughput']:.2f} toks/s\n"
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions with various budgets and RL support.")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument("--model", type=str, default="llama2", choices=["llama", "llama2", "llama3", "opt"])
    parser.add_argument("--dataset", type=str, nargs='+', default=["datasets/cnn_dailymail-3shot.jsonl"])
    parser.add_argument("--budget", type=int, nargs='+', default=[100])
    parser.add_argument("--method", type=str, nargs='+', default=["h2o"])
    parser.add_argument("--config_file", type=str, default="config/dataset2prompt.json", help="Path to compression config file")
    parser.add_argument("--use_rl", action="store_true", help="Use RL policy for dynamic compression")
    parser.add_argument("--rl_checkpoint", type=str, help="Path to RL model checkpoint (.pt file)")
    parser.add_argument("--sentence_transformer_model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model for context encoding")
    parser.add_argument("--context_window", type=int, default=64, help="Context window size for RL state encoding")
    args = parser.parse_args()
    main(args)