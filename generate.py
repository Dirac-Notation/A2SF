import os
import json
import argparse
import torch
from tqdm import tqdm

from utils import load_model, set_seed, CompressionConfig

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Generate responses using LLM with configurable settings and compare methods")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument('--model', type=str, required=True, choices=["llama", "llama2", "llama3", "opt"], help="Model to use")
    parser.add_argument('--budget', type=int, default=128, help="Budget for compression")
    parser.add_argument('--a', type=float, default=0.001, help="Parameter a for sigmoid cache")
    parser.add_argument('--b', type=float, default=4096, help="Observation window for snap cache (or parameter b for sigmoid)")
    return parser.parse_args(args)

def get_predefined_prompts():
    """
    Define your prompts here. You can modify these prompts as needed.
    """
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "What are the main differences between supervised and unsupervised learning?",
        "Describe the process of training a neural network.",
        "What is the importance of data preprocessing in machine learning?",
        "Explain the concept of overfitting and how to prevent it.",
        "What are the advantages and disadvantages of deep learning?",
        "Describe the role of activation functions in neural networks.",
        "What is gradient descent and how does it work?",
        "Explain the concept of regularization in machine learning.",
        "What are the key components of a machine learning pipeline?"
    ]
    return prompts

def format_prompt(prompt, model_name):
    """
    Format prompt based on model type (similar to longbench_pred.py)
    """
    if "llama" in model_name:
        return f"[INST]{prompt}[/INST]"
    else:
        return prompt

def generate_response(model, tokenizer, prompt, max_length, max_gen, model_name, config, method_name):
    """
    Generate response for a single prompt with given method
    """
    # Format prompt based on model type
    formatted_prompt = format_prompt(prompt, model_name)
    
    # Tokenize prompt
    tokenized_prompt = tokenizer(formatted_prompt, truncation=False, return_tensors="pt").input_ids[0]
    
    # Truncate if too long
    if len(tokenized_prompt) > max_length:
        half = int(max_length/2)
        formatted_prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                          tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    
    # Prepare input
    input = tokenizer(formatted_prompt, truncation=False, return_tensors="pt")
    input_ids = input.input_ids.to(model.device)
    attention_mask = input.attention_mask.to(torch.bfloat16).to(model.device)
    
    context_length = input_ids.shape[-1]
    
    # Initialize cache if using compression
    if hasattr(model, 'init_cache') and method_name != "full":
        model.init_cache(config)
    
    # Generate response
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )[0]
    
    # Decode response
    response = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    
    return response

def main():
    set_seed(42)
    args = parse_args()
    
    # Set GPU environment
    gpus = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    
    # Load model configuration
    model_name = args.model
    model_name = model_name.split("_")[0].lower()
    
    # Fixed parameters
    max_length = json.load(open("config/model2maxlen.json", "r"))[model_name]
    max_gen = 100
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, args.gpus)
    print("Model loaded successfully!")
    
    # Get number of layers from model config
    num_layers = model.config.num_hidden_layers
    print(f"Number of layers: {num_layers}")
    
    # Methods to compare (excluding a2sf)
    methods = ["full", "sigmoid", "snap"]
    
    # Create compression configs for each method
    configs = {}
    for method in methods:
        config = CompressionConfig()
        config.compression_method = method
        config.total_budget = args.budget
        config.layerwise_ratios = [1.0 for i in range(num_layers)]
        config.local_ratios = 0.125  # Fixed value
        
        if method == "sigmoid":
            config.a = args.a
            config.b = args.b
        elif method == "snap":
            config.observation_window = int(args.b)  # Use --b as observation_window
        
        configs[method] = config
    
    # Get predefined prompts
    prompts = get_predefined_prompts()
    
    print(f"\nGenerating responses for {len(prompts)} prompts...")
    print(f"Comparing methods: {', '.join(methods)}")
    print(f"Budget: {args.budget}")
    print(f"Max input length: {max_length}, Max generation: {max_gen}")
    print(f"Temperature: 0.0, Sampling: False")
    if "sigmoid" in methods:
        print(f"Sigmoid parameters: a={args.a}, b={args.b}")
    if "snap" in methods:
        print(f"Snap observation window: {int(args.b)}")
    print("=" * 80)
    
    # Generate responses for each method and compare
    all_results = {method: [] for method in methods}
    
    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        print(f"\n{'='*80}")
        print(f"Prompt {i+1}: {prompt}")
        print(f"{'='*80}")
        
        prompt_results = {}
        
        for method in methods:
            print(f"\n[{method.upper()}]")
            
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=max_length,
                max_gen=max_gen,
                model_name=model_name,
                config=configs[method],
                method_name=method
            )
            
            print(f"Response: {response}")
            
            result = {
                "prompt_id": i + 1,
                "prompt": prompt,
                "response": response,
                "model": model_name,
                "method": method,
                "budget": args.budget,
                "temperature": 0.0,
                "do_sample": False
            }
            
            if method == "sigmoid":
                result["a"] = args.a
                result["b"] = args.b
            elif method == "snap":
                result["observation_window"] = int(args.b)
            
            all_results[method].append(result)
            prompt_results[method] = response
        
        # Compare responses side by side
        print(f"\n{'='*80}")
        print("COMPARISON:")
        print(f"{'='*80}")
        for method in methods:
            print(f"\n[{method.upper()}]:")
            print(f"  {prompt_results[method]}")
    
    # Summary
    print(f"\n{'='*80}")
    print("GENERATION COMPLETED!")
    print(f"{'='*80}")
    print(f"Total prompts processed: {len(prompts)}")
    print(f"Methods compared: {', '.join(methods)}")
    for method in methods:
        print(f"  {method}: {len(all_results[method])} responses generated")

if __name__ == '__main__':
    main()
