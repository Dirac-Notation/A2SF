import torch
import json
import random
import numpy as np

from datasets import load_dataset
from dataclasses import dataclass

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
    data = {
        "streamingLLM": {
            "compression_method": "streamingLLM", "compression_ratio": [1.0 for _ in range(32)], "streaming_budget": 10
        },
        "h2o": {
            "compression_method": "h2o",
            "layerwise_ratio" : [1.0 for _ in range(32)],
            "forgetting_factors": [1.0 for _ in range(32)]
        },
        "llama2": {
            "compression_method": "a2sf",
            "layerwise_ratio": [1.67, 1.53, 1.0, 1.07, 0.43, 0.55, 1.59, 1.41, 2.07, 1.0, 1.0, 2.39, 1.0, 1.0, 1.27, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.52, 0.14, 0.3, 0.79, 0.27, 0.5],
            "forgetting_factors": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        },
        "llama2_factor_fixed": {
            "compression_method": "a2sf",
            "layerwise_ratio": [1.91, 1.51, 1.0, 1.07, 0.43, 0.55, 1.51, 1.41, 2.07, 1.0, 1.0, 2.25, 1.0, 1.0, 1.27, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.52, 0.14, 0.3, 0.79, 0.27, 0.5],
            "forgetting_factors": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        },
        "llama2_ratio_fixed": {
            "compression_method": "a2sf",
            "layerwise_ratio": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "forgetting_factors": [0.0, 0.01, 0.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        },
        "llama3": {
            "compression_method": "a2sf",
            "layerwise_ratio": [0.59, 0.23, 3.84, 0.26, 1.19, 1.14, 0.79, 1.0, 1.0, 1.0, 1.0, 1.0, 0.42, 1.0, 1.0, 1.0, 1.0, 1.0, 1.02, 1.0, 1.82, 0.38, 1.0, 1.0, 1.4, 1.0, 1.0, 1.59, 0.65, 0.66, 1.0, 0.02],
            "forgetting_factors": [0.0, 0.0, 0.88, 0.88, 0.88, 0.88, 0.88, 0.9, 0.9, 0.88, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93]
        }
    }

    punctuation_ids = None
    if method == "a2sf" and tokenizer is None:
        raise ValueError("Tokenizer is required for a2sf method")
    elif method == "a2sf" and tokenizer is not None:
        punctuation_ids = [
            tokenizer.encode(".", add_special_tokens=False)[0],
            tokenizer.encode(" .", add_special_tokens=False)[0],
        ]
    
    if method == "a2sf":
        data_point = data[model_name]
    elif "h2o" in method or method == "streamingLLM":
        data_point = data[method]
    elif method == "full":
        data_point = None
    
    if method == "a2sf":
        config = CompressionConfig(
            use_compression=True,
            compression_method=data_point["compression_method"],
            total_budget=total_budget,
            layerwise_ratio=data_point["layerwise_ratio"],
            forgetting_factors=data_point["forgetting_factors"],
            punctuation_ids=punctuation_ids
        )
    elif "h2o" in method:
        config = CompressionConfig(
            use_compression=True,
            compression_method=data_point["compression_method"],
            total_budget=total_budget,
            layerwise_ratio=data_point["layerwise_ratio"],
            forgetting_factors=data_point["forgetting_factors"]
        )
    elif "streamingLLM" in method:
        config = CompressionConfig(
            use_compression=True,
            compression_method=data_point["compression_method"],
            total_budget=total_budget,
            layerwise_ratio=data_point["compression_ratio"],
            streaming_budget=data_point["streaming_budget"]
        )
    elif method == "full":
        config = CompressionConfig(
            use_compression=False,
        )
    
    return config

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