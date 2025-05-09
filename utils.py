import torch

from datasets import load_dataset
from dataclasses import dataclass

@dataclass
class CompressionConfig:
    use_compression: bool = False
    total_budget: int = None
    compression_ratio: list = None
    forgetting_factors: list = None

def load_configs(model_name, method, total_budget):
    data = {
        "Llama-2-7b-chat-hf": {
            "h2o": {
                "compression_ratio": [(0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5)],
                "forgetting_factors": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            },
            "a2sf": {
                # "compression_ratio": [(0.95, 0.05), (0.8, 0.2), (0.3, 0.7), (0.5, 0.5), (0.3, 0.7), (0.35, 0.65), (0.45, 0.55), (0.45, 0.55), (0.45, 0.55), (0.45, 0.55), (0.4, 0.6), (0.4, 0.6), (0.45, 0.55), (0.4, 0.6), (0.45, 0.55), (0.4, 0.6), (0.45, 0.55), (0.4, 0.6), (0.15, 0.85), (0.15, 0.85), (0.15, 0.85), (0.15, 0.85), (0.15, 0.85), (0.25, 0.75), (0.15, 0.85), (0.2, 0.8), (0.15, 0.85), (0.25, 0.75), (0.25, 0.75), (0.15, 0.85), (0.2, 0.8), (0.25, 0.75)],
                "compression_ratio": [(0.95, 0.05), (0.95, 0.05), (0.85, 0.15), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.35, 0.65), (0.25, 0.75), (0.3, 0.7), (0.3, 0.7), (0.3, 0.7), (0.3, 0.7), (0.3, 0.7), (0.3, 0.7), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.35, 0.65), (0.4, 0.6), (0.45, 0.55), (0.35, 0.65), (0.45, 0.55), (0.45, 0.55), (0.35, 0.65), (0.45, 0.55), (0.45, 0.55), (0.45, 0.55), (0.45, 0.55), (0.45, 0.55), (0.45, 0.55), (0.6, 0.4)],
                "forgetting_factors": [1.0, 0.608, 0.722, 0.956, 0.914, 0.85, 0.936, 0.988, 0.988, 0.992, 0.988, 0.994, 0.992, 0.992, 0.998, 0.994, 0.998, 0.992, 0.642, 0.636, 0.608, 0.638, 0.878, 0.682, 0.652, 0.868, 0.608, 0.762, 0.67, 0.876, 0.846, 0.766]
            },
            "h2o_fixed": {
                "compression_ratio": [(0.95, 0.05), (0.95, 0.05), (0.9, 0.1), (0.9, 0.1), (0.70, 0.30), (0.55, 0.45), (0.60, 0.4), (0.60, 0.4), (0.55, 0.45), (0.65, 0.35), (0.60, 0.4), (0.55, 0.45), (0.60, 0.4), (0.55, 0.45), (0.5, 0.5), (0.60, 0.4), (0.5, 0.5), (0.5, 0.5), (0.55, 0.45), (0.55, 0.45), (0.5, 0.5), (0.4, 0.60), (0.45, 0.55), (0.5, 0.5), (0.45, 0.55), (0.45, 0.55), (0.5, 0.5), (0.60, 0.4), (0.60, 0.4), (0.5, 0.5), (0.60, 0.4), (0.70, 0.30)],
                "forgetting_factors": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            },
            "h2o_small": {
                "compression_ratio": [(0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75)],
                "forgetting_factors": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            },
            "h2o_large": {
                "compression_ratio": [(0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25)],
                "forgetting_factors": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            },
            "a2sf_fixed": {
                "compression_ratio" : [(0.95, 0.05), (0.8, 0.2), (0.5, 0.5), (0.65, 0.35), (0.4, 0.6), (0.4, 0.6), (0.65, 0.35), (0.6, 0.4), (0.4, 0.6), (0.4, 0.6), (0.35, 0.65), (0.35, 0.65), (0.35, 0.65), (0.35, 0.65), (0.3, 0.7), (0.4, 0.6), (0.35, 0.65), (0.35, 0.65), (0.35, 0.65), (0.35, 0.65), (0.35, 0.65), (0.25, 0.75), (0.3, 0.7), (0.35, 0.65), (0.35, 0.65), (0.35, 0.65), (0.35, 0.65), (0.4, 0.6), (0.35, 0.65), (0.35, 0.65), (0.35, 0.65), (0.45, 0.55)],
                "forgetting_factors": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
            }
        }
    }
    
    if method == "full":
        config = CompressionConfig(
            use_compression=False,
        )
    else:
        config = CompressionConfig(
        use_compression=True,
        total_budget=total_budget,
        compression_ratio=data[model_name][method]["compression_ratio"],
        forgetting_factors=data[model_name][method]["forgetting_factors"]
    )
    
    return config

datasets = load_dataset("abisee/cnn_dailymail", "3.0.0")
prompt_list = [datasets["test"][i]["article"] for i in range(len(datasets["test"]))]
prompt_list.sort(key=len)
def get_prompt(index: int = 0):
    return prompt_list[7000+index]

def make_optimal_mask(attention_maps, prompt_length, total_budget):
    optimal_maps = attention_maps.clone()
    
    for i in range(prompt_length, attention_maps.size(2)):
        # Select top-k tokens
        selected_scores = optimal_maps[:,:,[i],:].topk(k=2*total_budget, dim=3).indices
        
        # Create and apply mask
        mask = torch.zeros_like(optimal_maps[:,:,[i],:], device=attention_maps.device)
        mask[:,:,:,i:] = 1
        mask.scatter_(3, selected_scores, 1)
        
        # Apply mask and normalize
        optimal_maps[:,:,[i],:] = optimal_maps[:,:,[i],:] * mask
        divider = optimal_maps[:,:,i,:].sum(dim=2, keepdim=True)
        optimal_maps[:,:,i,:] = optimal_maps[:,:,i,:] / divider
    
    return optimal_maps

def make_a2sf_mask(attention_maps, prompt_length, total_budget, compression_ratio, forgetting_factors):
    a2sf_maps = attention_maps.clone()
    
    if isinstance(forgetting_factors, list):
        forgetting_factors = torch.tensor(forgetting_factors, device=attention_maps.device)
    
    exponent = torch.pow(forgetting_factors.unsqueeze(1), torch.arange(prompt_length-1,-1,-1, device=attention_maps.device).unsqueeze(0)).view(forgetting_factors.size(0),1,prompt_length,1)
    scores = (a2sf_maps[:,:,:prompt_length,:] * exponent).sum(dim=2, keepdim=True)
    
    num_layers = len(compression_ratio)
    recent_budget = [round(total_budget*compression_ratio[i][0]) for i in range(num_layers)]
    select_budget = [round(total_budget*compression_ratio[i][1]) for i in range(num_layers)]
    
    for layer_idx in range(num_layers):
        for i in range(prompt_length, attention_maps.size(2)):
            window_start = i - recent_budget[layer_idx]
            
            # Select top-k tokens within the window
            selected_scores = scores[layer_idx,:,:window_start].topk(k=select_budget[layer_idx], dim=2).indices
            
            # Create and apply mask
            mask = torch.zeros_like(scores[layer_idx,:,:,:], device=attention_maps.device)
            mask[:,:,window_start:] = 1
            mask.scatter_(2, selected_scores, 1)
            
            # Apply mask and normalize
            a2sf_maps[layer_idx,:,i:,:] = a2sf_maps[layer_idx,:,i:,:] * mask
            divider = a2sf_maps[layer_idx,:,i,:].sum(dim=1, keepdim=True)
            a2sf_maps[layer_idx,:,i,:] = a2sf_maps[layer_idx,:,i,:] / divider
            
            scores[layer_idx,:,:] = scores[layer_idx,:,:] * mask
            scores[layer_idx,:,:] = scores[layer_idx,:,:] + a2sf_maps[layer_idx,:,i,:].unsqueeze(1)
    
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