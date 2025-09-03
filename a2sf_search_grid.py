import os
import torch
import argparse
import torch.nn.functional as F
import json
import random
import itertools

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils_real_drop import Qwen2Tokenizer, Qwen2ForCausalLM

seed=42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Constants (replacing argparse arguments)
PROMPT_LENGTH = 900
GENERATION_LENGTH = 1
TOTAL_BUDGET = 128
BATCH_SIZE = 4
FULL_SEARCH = True

def get_prompt():
    with open("datasets/calibration_dataset.jsonl", "r") as f:
        articles = [json.loads(line)["article"] for line in f]
    return articles

def make_sentence_exp(input_ids, puntuation_ids):
    pos = torch.isin(input_ids, torch.tensor(puntuation_ids)).nonzero().tolist()

    exponents = torch.zeros_like(input_ids)
    for n_batch, pos_punc in pos:
        exponents[n_batch, :pos_punc] += 1
    
    return exponents

def make_layerwise_a2sf_mask(
    attention_maps,
    total_ids,
    budget_ratio,
    a2sf_factor,
    local_ratio,
    sentence_exp,
    puntuation_ids
    ):
    a2sf_maps = attention_maps.clone()
    
    layer_cache_budget = int(TOTAL_BUDGET * budget_ratio)
    layer_recent_budget = round(layer_cache_budget * local_ratio)
    layer_select_budget = round(layer_cache_budget * (1-local_ratio))
    
    # forgetting = (a2sf_factor**sentence_exp).view(BATCH_SIZE,1,-1,1)
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
        # for batch_idx in range(BATCH_SIZE):
            # if total_ids[batch_idx,current_pos].item() in puntuation_ids:
            #     layer_scores[batch_idx] *= a2sf_factor
        
    return a2sf_maps

def mul_att_value(attention_maps, values, num_attention_heads, num_key_value_heads):
    orig_shape = values.shape
    n_repeat = num_attention_heads // num_key_value_heads
    expanded_values = values.view(*orig_shape[:-2], 1, *orig_shape[-2:]).expand(*orig_shape[:-2], n_repeat, *orig_shape[-2:]).reshape(*orig_shape[:-3], num_attention_heads, *orig_shape[-2:])
    output = torch.matmul(attention_maps, expanded_values)
    output = output.transpose(-2, -3).contiguous()
    output = output.reshape(*output.shape[:-2], -1)
    return output

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer based on model name."""
    
    model2path = json.load(open("config/model2path.json", "r"))
    model_path = model2path[model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def get_punctuation_ids(tokenizer):
    """Get punctuation token IDs from tokenizer."""
    return [
        tokenizer.encode(".", add_special_tokens=False)[0],
        tokenizer.encode(" .", add_special_tokens=False)[0],
    ]

def process_batch_prompts(model, tokenizer, prompts, model_name, puntuation_ids):
    """Process a batch of prompts and return attention maps and values."""
    with torch.inference_mode():

        # Prepare batch input
        batch_input_ids = []
        for prompt in prompts[:BATCH_SIZE]:
            if "llama" in model_name.lower():
                prompt = f"[INST]{prompt}[/INST]"
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
            input_ids = torch.cat([input_ids[:, :PROMPT_LENGTH//2], input_ids[:, -PROMPT_LENGTH//2:]], dim=1)
            batch_input_ids.append(input_ids)
        input_ids = torch.cat(batch_input_ids, dim=0)
        
        total_ids = model.generate(
            input_ids.to(model.device),
            max_new_tokens=GENERATION_LENGTH,
            min_new_tokens=GENERATION_LENGTH,
            do_sample=False,
            temperature=0.0
        )

        outputs = model(total_ids, output_attentions=True, output_hidden_states=True)
        
        attn_shape = outputs.attentions[0].shape

        # Sentence Exponents
        sentence_exp = make_sentence_exp(input_ids, puntuation_ids)

        # Attention Maps
        attention_maps = torch.stack([attention.cpu() for attention in outputs.attentions], dim=0)
        
        # Values
        values = torch.stack([outputs.past_key_values[i][1].cpu() for i in range(len(outputs.past_key_values))], dim=0)
        
        # Hidden States
        hidden_states = torch.stack([outputs.hidden_states[i].cpu() for i in range(len(outputs.hidden_states))], dim=0)

        return total_ids, sentence_exp, attention_maps, values, hidden_states

def process_model(model_name, args):
    print(f"\nProcessing model: {model_name}")
    
    model, tokenizer = load_model_and_tokenizer(model_name)
    puntuation_ids = get_punctuation_ids(tokenizer)
    
    # Get model configuration
    num_attention_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads if hasattr(model.config, 'num_key_value_heads') else num_attention_heads
    
    print(f"Model config: num_attention_heads={num_attention_heads}, num_key_value_heads={num_key_value_heads}")
    
    total_ids_buffer = []
    sentence_exp_buffer = []
    attention_map_buffer = []
    values_buffer = []
    hidden_states_buffer = []
    
    prompts = get_prompt()
    prompts = [prompts[batch_start:batch_start+BATCH_SIZE] for batch_start in range(0, len(prompts), BATCH_SIZE)]
    
    for batch_prompts in tqdm(prompts, desc="Processing prompts"):
        batch_total_ids, batch_sentence_exp, batch_attention_maps, batch_values, batch_hidden_states = process_batch_prompts(
            model, tokenizer, batch_prompts, model_name, puntuation_ids
        )
        
        total_ids_buffer.append(batch_total_ids)
        sentence_exp_buffer.append(batch_sentence_exp)
        attention_map_buffer.append(batch_attention_maps)
        values_buffer.append(batch_values)
        hidden_states_buffer.append(batch_hidden_states)
        
        torch.cuda.empty_cache()
    
    num_layers = len(model.model.layers)
    
    # Search space
    factor_step = 0.05
    # local_ratio_step = 0.1
    
    # local_ratios = [local_ratio_step*i for i in range(int(1/local_ratio_step)+1)]
    local_ratios = [0.5]
    a2sf_factors = [factor_step*i for i in range(int(1/factor_step)+1)]
    
    all_grid = list(itertools.product(local_ratios, a2sf_factors))
    
    layerwise_budget_ratio = [1.0 for i in range(num_layers)]
    layerwise_local_ratio = [0.5 for i in range(num_layers)]
    layerwise_a2sf_factors = [1.0 for i in range(num_layers)]
    
    for _ in range(3):
        grid_score = [[0.0 for _ in range(len(all_grid))] for _ in range(num_layers)]
        for prompt_idx in tqdm(range(len(prompts))):
            attention_maps = attention_map_buffer[prompt_idx].to("cuda")
            values = values_buffer[prompt_idx].to("cuda")
            total_ids = total_ids_buffer[prompt_idx].to("cuda")
            sentence_exp = sentence_exp_buffer[prompt_idx].to("cuda")
            hidden_states = hidden_states_buffer[prompt_idx].to("cuda")
            
            original_output = mul_att_value(attention_maps[:,:,:,PROMPT_LENGTH:,:], values, num_attention_heads, num_key_value_heads)
            if FULL_SEARCH:
                for layer_idx in range(attention_maps.size(0)):
                    original_output[layer_idx] = model.model.layers[layer_idx].self_attn.o_proj(original_output[layer_idx])
                    original_output[layer_idx] += hidden_states[layer_idx][:,PROMPT_LENGTH:,:]
                    original_output[layer_idx] = model.model.layers[layer_idx].post_attention_layernorm(original_output[layer_idx])
                    original_output[layer_idx] = model.model.layers[layer_idx].mlp(original_output[layer_idx])

            for layer_idx in tqdm(range(num_layers)):
                layer_ratio = layerwise_budget_ratio[layer_idx]
                for grid_idx, (local_ratio, a2sf_factor) in enumerate(all_grid):
                    condition_maps = make_layerwise_a2sf_mask(attention_maps[layer_idx], total_ids, layer_ratio, a2sf_factor, local_ratio, sentence_exp, puntuation_ids)
                    condition_output = mul_att_value(condition_maps[:,:,PROMPT_LENGTH:,:], values[layer_idx], num_attention_heads, num_key_value_heads)
                    if FULL_SEARCH:
                        condition_output = model.model.layers[layer_idx].self_attn.o_proj(condition_output)
                        condition_output += hidden_states[layer_idx][:,PROMPT_LENGTH:,:].to(condition_output.device)
                        condition_output = model.model.layers[layer_idx].post_attention_layernorm(condition_output)
                        condition_output = model.model.layers[layer_idx].mlp(condition_output)
                    grid_score[layer_idx][grid_idx] += F.cosine_similarity(original_output[layer_idx], condition_output.to("cuda"), dim=2).mean().item()
            
            for layer_idx in range(num_layers):
                max_idx = grid_score[layer_idx].index(max(grid_score[layer_idx]))
                layerwise_local_ratio[layer_idx] = all_grid[max_idx][0]
                layerwise_a2sf_factors[layer_idx] = all_grid[max_idx][1]
            
            del attention_maps, values, total_ids, sentence_exp, hidden_states, original_output, condition_maps, condition_output
            torch.cuda.empty_cache()

        for prompt_idx in tqdm(range(len(prompts))):
            attention_maps = attention_map_buffer[prompt_idx].to("cuda")
            values = values_buffer[prompt_idx].to("cuda")
            total_ids = total_ids_buffer[prompt_idx].to("cuda")
            sentence_exp = sentence_exp_buffer[prompt_idx].to("cuda")
            hidden_states = hidden_states_buffer[prompt_idx].to("cuda")

            original_output = mul_att_value(attention_maps[:,:,:,PROMPT_LENGTH:,:], values, num_attention_heads, num_key_value_heads)
            if FULL_SEARCH:
                for layer_idx in range(attention_maps.size(0)):
                    original_output[layer_idx] = model.model.layers[layer_idx].self_attn.o_proj(original_output[layer_idx])
                    original_output[layer_idx] += hidden_states[layer_idx][:,PROMPT_LENGTH:,:]
                    original_output[layer_idx] = model.model.layers[layer_idx].post_attention_layernorm(original_output[layer_idx])
                    original_output[layer_idx] = model.model.layers[layer_idx].mlp(original_output[layer_idx])

            condition_maps = []
            for layer_idx in range(num_layers):
                layer_a2sf_factor = layerwise_a2sf_factors[layer_idx]
                layer_ratio = layerwise_budget_ratio[layer_idx]
                
                condition_maps.append(
                    make_layerwise_a2sf_mask(attention_maps[layer_idx], total_ids, layer_ratio, layer_a2sf_factor, local_ratio, sentence_exp, puntuation_ids)
                )

            condition_maps = torch.stack(condition_maps, dim=0)
            condition_output = mul_att_value(condition_maps[:,:,:,PROMPT_LENGTH:,:], values, num_attention_heads, num_key_value_heads)
            if FULL_SEARCH:
                for layer_idx in range(num_layers):
                    condition_output[layer_idx] = model.model.layers[layer_idx].self_attn.o_proj(condition_output[layer_idx])
                    condition_output[layer_idx] += hidden_states[layer_idx][:,PROMPT_LENGTH:,:].to(condition_output[layer_idx].device)
                    condition_output[layer_idx] = model.model.layers[layer_idx].post_attention_layernorm(condition_output[layer_idx])
                    condition_output[layer_idx] = model.model.layers[layer_idx].mlp(condition_output[layer_idx])
            sim_score = F.cosine_similarity(original_output, condition_output.to("cuda"), dim=3).mean(dim=(1,2))
            
            for _ in tqdm(range(100)):
                min_idx = sim_score.argmin()
                max_idx = sim_score.argmax()
                
                layerwise_budget_ratio[min_idx] += 0.01
                layerwise_budget_ratio[max_idx] -= 0.01
                
                condition_maps[min_idx] = make_layerwise_a2sf_mask(attention_maps[min_idx], total_ids, layerwise_budget_ratio[min_idx], layerwise_a2sf_factors[min_idx], layerwise_local_ratio[min_idx], sentence_exp, puntuation_ids)
                condition_maps[max_idx] = make_layerwise_a2sf_mask(attention_maps[max_idx], total_ids, layerwise_budget_ratio[max_idx], layerwise_a2sf_factors[max_idx], layerwise_local_ratio[max_idx], sentence_exp, puntuation_ids)
                
                condition_output = mul_att_value(condition_maps[:,:,:,PROMPT_LENGTH:,:], values, num_attention_heads, num_key_value_heads)
                if FULL_SEARCH:
                    for layer_idx in range(num_layers):
                        condition_output[layer_idx] = model.model.layers[layer_idx].self_attn.o_proj(condition_output[layer_idx])
                        condition_output[layer_idx] += hidden_states[layer_idx][:,PROMPT_LENGTH:,:].to(condition_output[layer_idx].device)
                        condition_output[layer_idx] = model.model.layers[layer_idx].post_attention_layernorm(condition_output[layer_idx])
                        condition_output[layer_idx] = model.model.layers[layer_idx].mlp(condition_output[layer_idx])
                sim_score = F.cosine_similarity(original_output, condition_output.to("cuda"), dim=3).mean(dim=(1,2))
            
            del attention_maps, values, total_ids, sentence_exp, hidden_states, original_output, condition_maps, condition_output, sim_score
            torch.cuda.empty_cache()

    layerwise_budget_ratio = [round(ratio, 2) for ratio in layerwise_budget_ratio]
    layerwise_a2sf_factors = [round(factor, 2) for factor in layerwise_a2sf_factors]
    layerwise_local_ratio = [round(ratio, 2) for ratio in layerwise_local_ratio]

    return {
        "model": model_name,
        "layerwise_ratios": layerwise_budget_ratio,
        "forgetting_factors": layerwise_a2sf_factors,
        "local_ratios": layerwise_local_ratio
    }

def main(args):    
    # Process each model
    results = []
    for model_name in args.models:
        result = process_model(
            model_name=model_name,
            args=args,
        )
        results.append(result)
        
        # Print results for this model
        print("\nSearch Results Summary:")
        print("=" * 50)
        print(f"Model: {model_name}")
        print(f'''
\"layerwise_ratios\": {result['layerwise_ratios']},
\"forgetting_factors\": {result['forgetting_factors']},
\"local_ratios\": {result['local_ratios']}
''')
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument("--models", type=str, nargs='+', default=["llama2"], choices=["llama", "llama2", "llama3", "opt", "qwen2"])
    args = parser.parse_args()
    
    gpu_list = ",".join(str(gpu) for gpu in args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    
    main(args)