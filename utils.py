import torch
import json
import random
import numpy as np

from datasets import load_dataset
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoConfig
from utils_real_drop import KVLlamaForCausalLM, KVOPTForCausalLM, KVQwen2ForCausalLM, Qwen2Tokenizer

@dataclass
class CompressionConfig:
    use_compression: bool = False
    compression_method: str = None
    total_budget: int = None
    streaming_budget: int = None
    layerwise_ratio: list = None
    forgetting_factors: list = None
    punctuation_ids: list = None

def load_configs(model_name, method, total_budget, tokenizer=None):
    """
    Load compression configuration based on model name and method.
    
    Args:
        model_name (str): Name of the model (e.g., 'llama2', 'llama3')
        method (str): Compression method ('a2sf', 'h2o', 'streamingLLM', 'average', 'full')
        total_budget (int): Total budget for compression
        tokenizer: Tokenizer instance (required for a2sf method)
    
    Returns:
        CompressionConfig: Configuration object for the specified method
    """
    import json
    
    # Load compression configurations from JSON file
    try:
        with open("config/compression_configs.json", "r") as f:
            compression_configs = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("compression_configs.json not found in config/ directory")
    
    # Validate method and get configuration
    if method == "a2sf":
        if model_name not in compression_configs:
            raise ValueError(f"Model '{model_name}' not found in compression configurations")
        config_data = compression_configs[model_name]
        
        # Validate a2sf method requires tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer is required for a2sf method")
        
        # Get punctuation IDs for a2sf method
        punctuation_ids = [
            tokenizer.encode(".", add_special_tokens=False)[0],
            tokenizer.encode(" .", add_special_tokens=False)[0],
        ]
        
        return CompressionConfig(
            use_compression=True,
            compression_method=config_data["compression_method"],
            total_budget=total_budget,
            layerwise_ratio=config_data["layerwise_ratio"],
            forgetting_factors=config_data["forgetting_factors"],
            punctuation_ids=punctuation_ids
        )
    
    elif method in ["h2o", "average"]:
        if method not in compression_configs:
            raise ValueError(f"Method '{method}' not found in compression configurations")
        config_data = compression_configs[method]
        
        return CompressionConfig(
            use_compression=True,
            compression_method=config_data["compression_method"],
            total_budget=total_budget,
            layerwise_ratio=config_data["layerwise_ratio"],
            forgetting_factors=config_data["forgetting_factors"]
        )
    
    elif method == "streamingLLM":
        if method not in compression_configs:
            raise ValueError(f"Method '{method}' not found in compression configurations")
        config_data = compression_configs[method]
        
        return CompressionConfig(
            use_compression=True,
            compression_method=config_data["compression_method"],
            total_budget=total_budget,
            layerwise_ratio=config_data["compression_ratio"],
            streaming_budget=config_data["streaming_budget"]
        )
    
    elif method == "full":
        return CompressionConfig(use_compression=False)
    
    else:
        raise ValueError(f"Unsupported method: {method}. Supported methods: a2sf, h2o, streamingLLM, average, full")

def load_model(model_name, gpu_list=None, model_path=None):
    """
    Load model and tokenizer based on model name and GPU configuration.
    
    Args:
        model_name (str): Name of the model (e.g., 'llama2', 'llama3', 'opt', 'qwen2')
        gpu_list (list): List of GPU IDs for multi-GPU setup. If None, uses single GPU.
        model_path (str): Path to the model. If None, loads from config/model2path.json
    
    Returns:
        tuple: (model, tokenizer)
    """
    # Load model path from config if not provided
    if model_path is None:
        with open("config/model2path.json", "r") as f:
            model2path = json.load(f)
        model_path = model2path[model_name]
    
    # Handle single GPU case
    if gpu_list is None:
        device = "cuda:0"
        
        if "llama" in model_name.lower():
            model = KVLlamaForCausalLM.from_pretrained(model_path).to(torch.bfloat16).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif "opt" in model_name.lower():
            model = KVOPTForCausalLM.from_pretrained(model_path).to(torch.bfloat16).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif "qwen" in model_name.lower():
            model = KVQwen2ForCausalLM.from_pretrained(model_path).to(torch.bfloat16).to(device)
            tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported model: {model_name}. Only Llama, OPT, and Qwen2 models are supported.")
    
    # Handle multi-GPU case
    else:
        num_gpus = len(gpu_list)
        
        # Load tokenizer first
        if "qwen" in model_name.lower():
            tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model config to get actual number of layers
        config = AutoConfig.from_pretrained(model_path)
        total_layers = config.num_hidden_layers
        
        # Create device map based on GPU count
        device_map = {}
        
        # Multiple GPUs: distribute layers across GPUs
        # Reserve last GPU for lm_head, distribute other layers across remaining GPUs
        layers_per_gpu = total_layers // (num_gpus - 1) if num_gpus > 1 else total_layers
        
        # Distribute layers across GPUs (excluding the last GPU)
        for i in range(total_layers):
            if num_gpus == 1:
                gpu_idx = 0
            else:
                gpu_idx = min(i // layers_per_gpu, num_gpus - 2)  # Use up to second-to-last GPU
            device_map[f"model.layers.{i}"] = f"cuda:{gpu_list[gpu_idx]}"
        
        # Place embedding and norm layers on first GPU
        device_map["model.embed_tokens"] = f"cuda:{gpu_list[0]}"
        device_map["model.norm"] = f"cuda:{gpu_list[-1]}"
        device_map["lm_head"] = f"cuda:{gpu_list[-1]}"

        print(f"Device map: {device_map}")

        # Load appropriate model based on model name
        if "llama" in model_name.lower():
            model = KVLlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device_map)
        elif "opt" in model_name.lower():
            model = KVOPTForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device_map)
        elif "qwen" in model_name.lower():
            model = KVQwen2ForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device_map)
        else:
            raise ValueError(f"Unsupported model: {model_name}. Only Llama, OPT, and Qwen2 models are supported.")
    
    model = model.eval()
    return model, tokenizer

def get_prompt(index: int = 0):
    with open("datasets/cnn_dailymail-2shot.jsonl", "r") as f:
        articles = [json.loads(line)["article"] for line in f]
    return articles[index]

def make_optimal_mask(attention_maps, prompt_length, total_budget):
    optimal_maps = attention_maps.clone()
    
    for i in range(prompt_length, attention_maps.size(2)):
        # Select top-k tokens
        selected_scores = optimal_maps[:,:,[i],:].topk(k=total_budget, dim=3).indices
        
        # Create and apply mask
        mask = torch.zeros_like(optimal_maps[:,:,[i],:], device=attention_maps.device)
        mask[:,:,:,i:] = 1
        mask.scatter_(3, selected_scores, 1)
        
        # Apply mask and normalize
        optimal_maps[:,:,[i],:] = optimal_maps[:,:,[i],:] * mask
        divider = optimal_maps[:,:,i,:].sum(dim=2, keepdim=True)
        optimal_maps[:,:,i,:] = optimal_maps[:,:,i,:] / divider
    
    return optimal_maps

def make_a2sf_mask(attention_maps, prompt_length, total_budget, compression_ratio, forgetting_factors, method="a2sf", input_ids=None, punctuation_ids=None):
    a2sf_maps = attention_maps.clone()
    
    if isinstance(forgetting_factors, list):
        forgetting_factors = torch.tensor(forgetting_factors, device=attention_maps.device)
    
    num_layers = len(compression_ratio)
    recent_budget = [round(total_budget*compression_ratio[i]) for i in range(num_layers)]
    select_budget = [round(total_budget*(1-compression_ratio[i])) for i in range(num_layers)]
    
    # Initialize scores for each layer
    scores = []
    for layer_idx in range(num_layers):            
        if method == "a2sf":
            if input_ids is None:
                raise ValueError("input_ids must be provided for a2sf method")
            if punctuation_ids is None:
                punctuation_ids = [13, 27]  # default: period and comma token IDs
            
            # Get punctuation positions
            prompt_ids = input_ids[:, :prompt_length]
            flattened_input_ids = prompt_ids.view(-1)
            num_all = flattened_input_ids.size(0)
            
            pos = torch.isin(flattened_input_ids, torch.tensor(punctuation_ids, device=input_ids.device)).nonzero(as_tuple=True)[0].tolist()
            num_t = len(pos)
            
            starts = [0] + [p + 1 for p in pos]
            ends = pos + [num_all - 1]
            
            # Initialize scores with sentence-based decay
            exponents = torch.empty_like(flattened_input_ids)
            for i, (s, e) in enumerate(zip(starts, ends)):
                exponents[s : e + 1] = num_t - i
            exponents = exponents.view(1, prompt_length, 1)
            
            forgetting = (forgetting_factors[layer_idx]**exponents).view(1, prompt_length, 1)
            layer_scores = (a2sf_maps[layer_idx,:,:prompt_length,:] * forgetting).sum(dim=1, keepdim=True)
        elif method == "h2o":
            layer_scores = a2sf_maps[layer_idx,:,:prompt_length,:].sum(dim=1, keepdim=True)
            
        scores.append(layer_scores)
    
    # Process each layer
    for layer_idx in range(num_layers):
        for i in range(prompt_length, attention_maps.size(2)):
            window_start = i - recent_budget[layer_idx]
            
            # Select top-k tokens within the window
            selected_scores = scores[layer_idx][:,:window_start].topk(k=select_budget[layer_idx], dim=2).indices
            
            # Create and apply mask
            mask = torch.zeros_like(scores[layer_idx], device=attention_maps.device)
            mask[:,:,window_start:] = 1
            mask.scatter_(2, selected_scores, 1)
            
            # Apply mask and normalize
            a2sf_maps[layer_idx,:,i:,:] = a2sf_maps[layer_idx,:,i:,:] * mask
            divider = a2sf_maps[layer_idx,:,i,:].sum(dim=1, keepdim=True)
            a2sf_maps[layer_idx,:,i,:] = a2sf_maps[layer_idx,:,i,:] / divider
            
            # Update scores
            scores[layer_idx] = scores[layer_idx] * mask
            scores[layer_idx] = scores[layer_idx] + a2sf_maps[layer_idx,:,i,:].unsqueeze(1)
            
            # Apply forgetting based on method
            if method == "a2sf":
                current_token = input_ids[0, i]
                if current_token in punctuation_ids:
                    scores[layer_idx] *= forgetting_factors[layer_idx]
    
    return a2sf_maps

def generate(model, input_ids, generation_length):
    with torch.inference_mode():        
        past_key_values = None
        
        outputs = model(input_ids, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        
        for _ in range(generation_length):
            next_token_scores = next_token_logits
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=1)
            
            outputs = model(next_tokens.unsqueeze(-1), past_key_values=past_key_values, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        outputs = model(input_ids, output_attentions=True)
        
        attention_maps = torch.cat(outputs.attentions, dim=0)
        values = torch.cat([past_key_values[i][1] for i in range(32)], dim=0)
    
    return outputs, attention_maps, values