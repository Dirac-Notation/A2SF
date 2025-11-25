import os
import json
import argparse
import torch
from tqdm import tqdm

from utils import load_model, set_seed, CompressionConfig

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Generate responses using LLM with configurable settings")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument('--model', type=str, required=True, choices=["llama", "llama2", "llama3", "opt"], help="Model to use")
    parser.add_argument('--method', type=str, default="full", help="Compression method (full, a2sf, sigmoid, etc.)")
    parser.add_argument('--budget', type=int, default=128, help="Budget for compression")
    parser.add_argument('--a', type=float, default=0.001, help="Parameter a for sigmoid cache")
    parser.add_argument('--b', type=float, default=4096, help="Parameter b for sigmoid cache")
    parser.add_argument('--num_layers', type=int, default=32, help="Number of layers (default: 32)")
    parser.add_argument('--local_ratios', type=float, default=0.125, help="Local ratios for compression (default: 0.125)")
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

def generate_response(model, tokenizer, prompt, max_length, max_gen, model_name, config):
    """
    Generate response for a single prompt
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
    if hasattr(model, 'init_cache'):
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
    
    # Create compression config
    config = CompressionConfig()
    config.compression_method = args.method
    config.total_budget = args.budget
    config.layerwise_ratios = [1.0 for i in range(args.num_layers)]
    config.local_ratios = args.local_ratios
    
    # Method-specific parameters
    if args.method == "sigmoid":
        config.a = args.a
        config.b = args.b
    elif args.method == "a2sf":
        # For a2sf, you might need forgetting_factors
        # Default to 0.5 if not specified
        config.forgetting_factors = [0.5 for i in range(args.num_layers)]
    
    # Get predefined prompts
    prompts = get_predefined_prompts()
    
    print(f"Generating responses for {len(prompts)} prompts...")
    print(f"Using method: {args.method}, budget: {args.budget}")
    print(f"Max input length: {max_length}, Max generation: {max_gen}")
    print(f"Temperature: 0.0, Sampling: False")
    print("-" * 50)
    
    # Generate responses
    results = []
    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        print(f"\nPrompt {i+1}: {prompt}")

        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=max_length,
            max_gen=max_gen,
            model_name=model_name,
            config=config
        )
        
        print(f"Response: {response}")
        
        # Store result (no file output, just in memory)
        result = {
            "prompt_id": i + 1,
            "prompt": prompt,
            "response": response,
            "model": model_name,
            "method": args.method,
            "budget": args.budget,
            "temperature": 0.0,
            "do_sample": False
        }
        results.append(result)
    
    # No file output - results are only stored in memory
    print(f"\nGeneration completed!")
    print(f"Total prompts processed: {len(results)}")

if __name__ == '__main__':
    main()
