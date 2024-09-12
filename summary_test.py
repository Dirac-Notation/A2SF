import argparse
import logging

import numpy as np
import torch
import json
from tqdm import tqdm
import random
import copy

from rouge import Rouge

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils_real_drop.modify_llama import H2OLlamaAttention

def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is not None:
        input_ids = input_ids[:, -1:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

def evaluate_model(model, prompts, input_idses, device):
    results = []
    with tqdm(list(zip(prompts, input_idses))) as pbar:
        for prompt, input_ids in pbar:
            pbar.set_description(f"{input_ids.numel()}")
            generate_ids = model.generate(input_ids.to(device), max_new_tokens=args.length, do_sample=False, temperature=1.0, top_p=1.0)
            result = tokenizer.batch_decode(generate_ids)[0]
            result = result.replace(prompt, "")
            result = result[:result.find("###")].strip()
            results.append(result)
            
            for layer_idx in range(len(model.model.layers)):
                if isinstance(model.model.layers[layer_idx].self_attn, H2OLlamaAttention):
                    model.model.layers[layer_idx].self_attn._clean_cache()

            torch.cuda.empty_cache()
            
    return results

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default='huggyllama/llama-7b')
    parser.add_argument("--cache_budget", type=int, default=20)
    parser.add_argument("--forgetting_factor", type=float, default=0.1)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--data_path", type=str, default="data/xsum-3shot.jsonl")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.warning(f"device: {args.device}")

    rouge = Rouge()

    # Model Load
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).half().eval() 
    check_point = copy.deepcopy(model.state_dict())
    model.to(args.device)

    # Data Load
    with open(args.data_path, "r") as f:
        file = f.readlines()
        prompts, input_idses, answers, input_lengths = list(), list(), list(), list()
        for line in file:
            data = json.loads(line)
            # data = json.loads(random.choice(file))
            prompt = data["article"]
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
            answer = data["summary_gt"]
            seq_len = input_ids.numel()
            
            if seq_len > 4096:
                continue
            
            prompts.append(prompt)
            input_idses.append(input_ids)
            answers.append(answer)
            input_lengths.append(seq_len)
        print(f"Average Prompt Length: {sum(input_lengths)/len(input_lengths)}")
    
    ######## Full Cache
    full_cache_results = evaluate_model(model, prompts, input_idses, args.device)
    with open("results/full_model.jsonl", "w") as f:
        for item in full_cache_results:
            json_line = json.dumps({"response": item})
            f.write(f"{json_line}\n")

    model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model, type(model))
    
    ######### H2O 
    config.scoring_policy = "h2o"
    config.hh_size = int(args.cache_budget/2)
    config.recent_size = int(args.cache_budget/2)
    config.forgetting_factor = 1.0
    for layer_idx in range(config.num_hidden_layers):
        model.model.layers[layer_idx].self_attn = H2OLlamaAttention(config)
    model.load_state_dict(check_point)
    model.half().eval().to(args.device)

    h2o_results = evaluate_model(model, prompts, input_idses, args.device)
    with open("results/h2o_model.jsonl", "w") as f:
        for item in h2o_results:
            json_line = json.dumps({"response": item})
            f.write(f"{json_line}\n")

    ######### A2SF
    for layer_idx in range(config.num_hidden_layers):
        model.model.layers[layer_idx].self_attn.kv_cache.scoring_policy = "a2sf"
        
    for factor in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for mode in ["Local", "No Local"]:
            print(f"Factor: {factor} / Mode: {mode}")
            for layer_idx in range(config.num_hidden_layers):
                if mode == "Local":
                    model.model.layers[layer_idx].self_attn.kv_cache.hh_size = int(args.cache_budget/2)
                    model.model.layers[layer_idx].self_attn.kv_cache.recent_size = int(args.cache_budget/2)
                elif mode == "No Local":
                    model.model.layers[layer_idx].self_attn.kv_cache.hh_size = int(args.cache_budget)
                    model.model.layers[layer_idx].self_attn.kv_cache.recent_size = 0
                model.model.layers[layer_idx].self_attn.kv_cache.forgetting_factor = factor

            a2sf_results = evaluate_model(model, prompts, input_idses, args.device)

            with open(f"results/a2sf_{factor}_{mode}_model.jsonl", "w") as f:
                for item in a2sf_results:
                    json_line = json.dumps({"response": item})
                    f.write(f"{json_line}\n")
            
            full_cache_score, h2o_score, a2sf_score = list(), list(), list()
            
            for rouge_type in ["1", "2", "l"]:
                for answer, full_cache, h2o, a2sf in zip(answers, full_cache_results, h2o_results, a2sf_results):
                    try:
                        full_cache_score.append(rouge.get_scores(answer, full_cache)[0][f"rouge-{rouge_type}"]["f"])
                        h2o_score.append(rouge.get_scores(answer, h2o)[0][f"rouge-{rouge_type}"]["f"])
                        a2sf_score.append(rouge.get_scores(answer, a2sf)[0][f"rouge-{rouge_type}"]["f"])
                    except:
                        print(f"answer : {answer}")
                        print(f"full_cache : {full_cache}")
                        print(f"h2o : {h2o}")
                        print(f"a2sf : {a2sf}")
                        continue
                
                print(rouge_type)
                print(f"Full Cache Rouge Score: {sum(full_cache_score)/len(full_cache_score)}")
                print(f"H2O Rouge Score: {sum(h2o_score)/len(h2o_score)}")
                print(f"A2SF Rouge Score: {sum(a2sf_score)/len(a2sf_score)}")
            print()