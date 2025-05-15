import torch
import json

from datasets import load_dataset
from dataclasses import dataclass

@dataclass
class CompressionConfig:
    use_compression: bool = False
    compression_method: str = None
    total_budget: int = None
    streaming_budget: int = None
    compression_ratio: list = None
    forgetting_factors: list = None
    punctuation_ids: list = None

def load_configs(model_name, method, total_budget, tokenizer=None):
    data = {
        "streamingLLM": {
            "compression_method": "h2o", "compression_ratio": [1.0 for _ in range(32)], "streaming_ratio": 0.05
        },
        "h2o": {
            "compression_method": "h2o", "compression_ratio": [0.5 for _ in range(32)]
        },
        "h2o_1": {
            "compression_method": "h2o", "compression_ratio": [0.1 for _ in range(32)]
        },
        "h2o_2": {
            "compression_method": "h2o", "compression_ratio": [0.2 for _ in range(32)]
        },
        "h2o_3": {
            "compression_method": "h2o", "compression_ratio": [0.3 for _ in range(32)]
        },
        "h2o_4": {
            "compression_method": "h2o", "compression_ratio": [0.4 for _ in range(32)]
        },
        "h2o_5": {
            "compression_method": "h2o", "compression_ratio": [0.5 for _ in range(32)]
        },
        "h2o_6": {
            "compression_method": "h2o", "compression_ratio": [0.6 for _ in range(32)]
        },
        "h2o_7": {
            "compression_method": "h2o", "compression_ratio": [0.7 for _ in range(32)]
        },
        "h2o_8": {
            "compression_method": "h2o", "compression_ratio": [0.8 for _ in range(32)]
        },
        "h2o_9": {
            "compression_method": "h2o", "compression_ratio": [0.9 for _ in range(32)]
        },
        "llama2": {
            "compression_method": "a2sf",
            "compression_ratio" : [0.95, 0.85, 0.05, 0.05, 0.1, 0.25, 0.2, 0.15, 0.2, 0.25, 0.2, 0.2, 0.2, 0.2, 0.15, 0.2, 0.15, 0.2, 0.2, 0.15, 0.15, 0.15, 0.2, 0.15, 0.2, 0.2, 0.2, 0.1, 0.25, 0.2, 0.25, 0.3],
            "forgetting_factors": [0.0, 0.01, 0.01, 0.01, 0.01, 0.92, 0.96, 0.97, 0.95, 0.97, 0.97, 1.0, 0.97, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 1.0, 0.93, 0.99, 0.01, 0.94, 1.0, 0.94, 0.9]
        },
        "llama3": {
            "compression_method": "a2sf",
            "compression_ratio" : [0.95, 0.55, 0.2, 0.3, 0.15, 0.1, 0.25, 0.15, 0.25, 0.2, 0.25, 0.25, 0.3, 0.1, 0.1, 0.15, 0.1, 0.25, 0.1, 0.1, 0.15, 0.25, 0.15, 0.1, 0.1, 0.05, 0.1, 0.15, 0.25, 0.25, 0.15, 0.25],
            "forgetting_factors": [0.92, 0.83, 0.82, 0.9, 0.88, 0.78, 0.87, 0.89, 0.86, 0.86, 0.88, 0.87, 0.9, 0.86, 0.86, 0.89, 0.86, 0.9, 0.84, 0.81, 0.87, 0.84, 0.82, 0.83, 0.84, 0.79, 0.88, 0.79, 0.88, 0.91, 0.9, 0.91]
        },
        "opt": {
            "compression_method": "a2sf",
            "compression_ratio" : [0.6, 0.95, 0.6, 0.75, 0.55, 0.15, 0.1, 0.1, 0.1, 0.1, 0.15, 0.1, 0.15, 0.2, 0.15, 0.15, 0.1, 0.15, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.1, 0.15, 0.05, 0.05],
            "forgetting_factors": [0.95, 0.99, 0.0, 0.0, 0.0, 0.11, 0.08, 0.15, 0.22, 0.19, 0.33, 0.28, 0.74, 0.77, 0.79, 0.78, 0.79, 0.81, 0.09, 0.83, 0.86, 0.85, 0.86, 0.86, 0.86, 0.85, 0.15, 0.78, 0.83, 0.81, 0.22, 0.14]
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
            compression_ratio=data_point["compression_ratio"] if "compression_ratio" in data_point else None,
            forgetting_factors=data_point["forgetting_factors"] if "forgetting_factors" in data_point else None,
            punctuation_ids=punctuation_ids
        )
    elif "h2o" in method:
        config = CompressionConfig(
            use_compression=True,
            compression_method=data_point["compression_method"],
            total_budget=total_budget,
            compression_ratio=data_point["compression_ratio"] if "compression_ratio" in data_point else None,
        )
    elif "streamingLLM" in method:
        config = CompressionConfig(
            use_compression=True,
            compression_method=data_point["compression_method"],
            total_budget=total_budget,
            streaming_ratio=data_point["streaming_ratio"] if "streaming_ratio" in data_point else None,
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
        num_round = 0  # For a2sf_round method
        
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