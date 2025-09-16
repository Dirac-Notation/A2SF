import os
import torch
import argparse
import torch.nn.functional as F
import json
import random
import itertools
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

seed=42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Constants (replacing argparse arguments)
PROMPT_LENGTH = 4095
GENERATION_LENGTH = 1
TOTAL_BUDGET = 64

def get_prompt(task):
    with open("datasets/calibration_dataset.jsonl", "r") as f:
        articles = []
        for line in f:
            line_data = json.loads(line)
            if line_data["group"] != task:
                continue
            article = line_data["article"]
            if len(article) > PROMPT_LENGTH:
                articles.append(article)
    return articles

def make_layerwise_a2sf_mask(
    attention_maps,
    budget_ratio,
    a2sf_factor,
    local_ratio,
    ):
    a2sf_maps = attention_maps.clone()
    
    layer_cache_budget = int(TOTAL_BUDGET * budget_ratio)
    layer_recent_budget = round(layer_cache_budget * local_ratio)
    layer_select_budget = round(layer_cache_budget * (1-local_ratio))

    forgetting = torch.arange(PROMPT_LENGTH-1,-1,-1).view(1,1,-1,1)
    forgetting = a2sf_factor**forgetting
    layer_scores = (a2sf_maps[:,:,:PROMPT_LENGTH,:] * forgetting.to(a2sf_maps.device)).sum(dim=2, keepdim=True)
    
    for i in range(attention_maps.size(2)-PROMPT_LENGTH):
        current_pos = PROMPT_LENGTH + i
        window_start = PROMPT_LENGTH - layer_recent_budget
        
        selected_scores = layer_scores[:,:,:window_start].topk(k=layer_select_budget, dim=3).indices
        
        mask = torch.zeros_like(layer_scores)
        mask[:,:,:,window_start:] = 1
        mask.scatter_(3, selected_scores, 1)
        
        a2sf_maps[:,:,current_pos:,:] = a2sf_maps[:,:,current_pos:,:] * mask
        divider = a2sf_maps[:,:,current_pos,:].sum(dim=2, keepdim=True)
        a2sf_maps[:,:,current_pos,:] = a2sf_maps[:,:,current_pos,:] / divider
        
        layer_scores *= mask
        layer_scores += a2sf_maps[:,:,current_pos,:].unsqueeze(2)

        layer_scores *= a2sf_factor
        
    return a2sf_maps

def mul_att_value(attention_maps, values, num_attention_heads, num_key_value_heads):
    orig_shape = values.shape
    n_repeat = num_attention_heads // num_key_value_heads
    expanded_values = values.view(*orig_shape[:-2], 1, *orig_shape[-2:]).expand(*orig_shape[:-2], n_repeat, *orig_shape[-2:]).reshape(*orig_shape[:-3], num_attention_heads, *orig_shape[-2:])
    output = torch.matmul(attention_maps, expanded_values)
    output = output.transpose(-2, -3).contiguous()
    output = output.reshape(*output.shape[:-2], -1)
    return output

def mul_out_residual_mlp(hidden_states, residual, model, layer_idx):
    hidden_states = model.model.layers[layer_idx].self_attn.o_proj(hidden_states)
    hidden_states += residual
    hidden_states = model.model.layers[layer_idx].post_attention_layernorm(hidden_states)
    hidden_states = model.model.layers[layer_idx].mlp(hidden_states)
    return hidden_states

def load_model_and_tokenizer(model_name):
    model2path = json.load(open("config/model2path.json", "r"))
    model_path = model2path[model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def process_batch_prompts(model, tokenizer, prompts):
    with torch.inference_mode():
        batch_input_ids = []
        for prompt in prompts:
            if "llama" in model.config.model_type.lower():
                prompt = f"[INST]{prompt}[/INST]"
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
            input_ids = torch.cat([input_ids[:, :PROMPT_LENGTH//2], input_ids[:, -PROMPT_LENGTH//2:]], dim=1)
            batch_input_ids.append(input_ids)
        input_ids = torch.cat(batch_input_ids, dim=0)
        
        with torch.no_grad():
            total_ids = model.generate(
                input_ids.to(model.device),
                max_new_tokens=GENERATION_LENGTH,
                min_new_tokens=GENERATION_LENGTH,
                do_sample=False,
                temperature=0.0
            )
            outputs = model(total_ids, output_attentions=True, output_hidden_states=True)
        
        attention_maps = torch.stack([attention.cpu() for attention in outputs.attentions], dim=0)
        values = torch.stack([outputs.past_key_values[i][1].cpu() for i in range(len(outputs.past_key_values))], dim=0)
        hidden_states = torch.stack([outputs.hidden_states[i].cpu() for i in range(len(outputs.hidden_states))], dim=0)

        return attention_maps, values, hidden_states

def process_model(model, tokenizer, prompts, task):    
    attention_map_buffer = []
    values_buffer = []
    hidden_states_buffer = []
    
    for batch_prompts in tqdm(prompts, desc="Processing prompts"):
        batch_attention_maps, batch_values, batch_hidden_states = process_batch_prompts(
            model, tokenizer, batch_prompts
        )

        attention_map_buffer.append(batch_attention_maps)
        values_buffer.append(batch_values)
        hidden_states_buffer.append(batch_hidden_states)
        
        torch.cuda.empty_cache()
    
    num_layers = model.config.num_hidden_layers
    
    # Search space
    # local_ratio_step = 0.1
    
    # local_ratios = [local_ratio_step*i for i in range(int(1/local_ratio_step)+1)]
    local_ratios = [0.125]
    a2sf_factors = [i/10 for i in range(8)] + [0.80+i/20 for i in range(2)] + [0.90+i/100 for i in range(8)] + [0.9800+i/10000 for i in range(201)]
    
    all_grid = list(itertools.product(local_ratios, a2sf_factors))
    
    layerwise_budget_ratio = [1.0 for i in range(num_layers)]
    layerwise_local_ratio = [0.5 for i in range(num_layers)]
    layerwise_a2sf_factors = [1.0 for i in range(num_layers)]

    # Model configuration
    num_attention_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads if hasattr(model.config, 'num_key_value_heads') else num_attention_heads

    with torch.no_grad():
        for _ in range(1):
            grid_score = [[0.0 for _ in range(len(all_grid))] for _ in range(num_layers)]
            for prompt_idx in tqdm(range(len(prompts))):
                attention_maps = attention_map_buffer[prompt_idx].to("cuda")
                values = values_buffer[prompt_idx].to("cuda")
                hidden_states = hidden_states_buffer[prompt_idx].to("cuda")
                
                original_output = mul_att_value(attention_maps[:,:,:,PROMPT_LENGTH:,:], values, num_attention_heads, num_key_value_heads)
                for layer_idx in range(attention_maps.size(0)):
                    original_output[layer_idx] = mul_out_residual_mlp(original_output[layer_idx], hidden_states[layer_idx][:,PROMPT_LENGTH:,:], model, layer_idx)
                
                for layer_idx in tqdm(range(num_layers)):
                    layer_ratio = layerwise_budget_ratio[layer_idx]
                    for grid_idx, (local_ratio, a2sf_factor) in enumerate(all_grid):
                        condition_maps = make_layerwise_a2sf_mask(attention_maps[layer_idx], layer_ratio, a2sf_factor, local_ratio)
                        condition_output = mul_att_value(condition_maps[:,:,PROMPT_LENGTH:,:], values[layer_idx], num_attention_heads, num_key_value_heads)
                        condition_output = mul_out_residual_mlp(condition_output, hidden_states[layer_idx][:,PROMPT_LENGTH:,:], model, layer_idx)
                        grid_score[layer_idx][grid_idx] += torch.norm(original_output[layer_idx] - condition_output.to("cuda"), dim=2).mean().item()
                
                del attention_maps, values, hidden_states, original_output, condition_maps, condition_output
                torch.cuda.empty_cache()
            
            for layer_idx in range(num_layers):
                min_idx = grid_score[layer_idx].index(min(grid_score[layer_idx]))
                layerwise_local_ratio[layer_idx] = all_grid[min_idx][0]
                layerwise_a2sf_factors[layer_idx] = all_grid[min_idx][1]
                
            # for prompt_idx in tqdm(range(len(prompts))):
            #     attention_maps = attention_map_buffer[prompt_idx].to("cuda")
            #     values = values_buffer[prompt_idx].to("cuda")
            #     hidden_states = hidden_states_buffer[prompt_idx].to("cuda")

            #     original_output = mul_att_value(attention_maps[:,:,:,PROMPT_LENGTH:,:], values, num_attention_heads, num_key_value_heads)
            #     for layer_idx in range(attention_maps.size(0)):
            #         original_output[layer_idx] = mul_out_residual_mlp(original_output[layer_idx], hidden_states[layer_idx][:,PROMPT_LENGTH:,:], model, layer_idx)

            #     condition_maps = []
            #     for layer_idx in range(num_layers):
            #         layer_a2sf_factor = layerwise_a2sf_factors[layer_idx]
            #         layer_ratio = layerwise_budget_ratio[layer_idx]
            #         condition_maps.append(make_layerwise_a2sf_mask(attention_maps[layer_idx], layer_ratio, layer_a2sf_factor, local_ratio))

            #     condition_maps = torch.stack(condition_maps, dim=0)
            #     condition_output = mul_att_value(condition_maps[:,:,:,PROMPT_LENGTH:,:], values, num_attention_heads, num_key_value_heads)
            #     for layer_idx in range(num_layers):
            #         condition_output[layer_idx] = mul_out_residual_mlp(condition_output[layer_idx], hidden_states[layer_idx][:,PROMPT_LENGTH:,:], model, layer_idx)
            #     sim_score = torch.norm(original_output - condition_output.to("cuda"), dim=3).mean(dim=(1,2))
                
            #     for _ in tqdm(range(100)):
            #         min_idx = sim_score.argmin()
            #         max_idx = sim_score.argmax()
                    
            #         layerwise_budget_ratio[min_idx] -= 0.01
            #         layerwise_budget_ratio[max_idx] += 0.01
                    
            #         condition_maps[min_idx] = make_layerwise_a2sf_mask(attention_maps[min_idx], layerwise_budget_ratio[min_idx], layerwise_a2sf_factors[min_idx], layerwise_local_ratio[min_idx])
            #         condition_maps[max_idx] = make_layerwise_a2sf_mask(attention_maps[max_idx], layerwise_budget_ratio[max_idx], layerwise_a2sf_factors[max_idx], layerwise_local_ratio[max_idx])
                    
            #         condition_output = mul_att_value(condition_maps[:,:,:,PROMPT_LENGTH:,:], values, num_attention_heads, num_key_value_heads)
            #         for layer_idx in range(num_layers):
            #             condition_output[layer_idx] = mul_out_residual_mlp(condition_output[layer_idx], hidden_states[layer_idx][:,PROMPT_LENGTH:,:], model, layer_idx)
            #         sim_score = torch.norm(original_output - condition_output.to("cuda"), dim=3).mean(dim=(1,2))
                
            #     del attention_maps, values, hidden_states, original_output, condition_maps, condition_output, sim_score
            #     torch.cuda.empty_cache()

    layerwise_budget_ratio = [round(ratio, 4) for ratio in layerwise_budget_ratio]
    layerwise_a2sf_factors = [round(factor, 4) for factor in layerwise_a2sf_factors]
    layerwise_local_ratio = [round(ratio, 4) for ratio in layerwise_local_ratio]

    return {
        "layerwise_ratios": layerwise_budget_ratio,
        "forgetting_factors": layerwise_a2sf_factors,
        "local_ratios": layerwise_local_ratio
    }

def main(args):
    tasks = ["Code Complete", "Few Shot", "Single-doc QA", "Multi-doc QA", "Summarization", "Passage Retrieval"]
  
    model, tokenizer = load_model_and_tokenizer(args.model)

    results = {}
    for task in tasks:
        prompts = get_prompt(task)
        batch_size = 1
        prompts = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
        results[task] = process_model(
            model = model,
            tokenizer = tokenizer,
            prompts = prompts,
            task = task
        )

    print("\nSearch Results")
    print(f"{{")
    print(f"\"a2sf\": {{")
    for task in tasks:
        print(f"\"{task}\" : {{")
        print(f"\"compression_method\": \"a2sf\",")
        print(f"\"layerwise_ratios\": {results[task]['layerwise_ratios']},")
        print(f"\"forgetting_factors\": {results[task]['forgetting_factors']},")
        print(f"\"local_ratios\": {results[task]['local_ratios']}")
        print(f"}}{"," if task != tasks[-1] else ""}")
    print(f"}}")
    print(f"}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument("--model", type=str, default="llama2", choices=["llama", "llama2", "llama3", "opt", "qwen2"])
    args = parser.parse_args()
    
    gpu_list = ",".join(str(gpu) for gpu in args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    
    main(args)