import copy
import torch
import json
import os
import numpy as np
import random
import matplotlib.pyplot as plt

from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def get_prompt(json_line):
    data = json.loads(json_line)
    return data["prompt"]

def make_mask(input_tensor, heavy_ratio, penalty):
    tensor = torch.clone(input_tensor)
    cache_budget = int(tensor.shape[-2]*heavy_ratio)
    
    a2sf = torch.zeros_like(tensor[:,:,0,:])
    tmp_mask = torch.ones_like(tensor[:,:,0,:])
    
    for i in range(cache_budget):
        a2sf = penalty*a2sf + tensor[:,:,i,:]
    
    for i in range(cache_budget, tensor.shape[-2]):
        current_score = tensor[:,:,i,:]
        
        current_score *= tmp_mask
        current_score /= (torch.sum(current_score, dim=-1).unsqueeze(dim=-1) + 1e-10)
        
        if i != tensor.shape[-2]-1:
            if penalty != 0.0:
                a2sf = penalty*a2sf + current_score
            else:
                a2sf[a2sf!=torch.inf] = 0
                a2sf += current_score
        
            min_index = torch.argmin(a2sf[:,:,:i+1], axis=-1).unsqueeze(dim=-1)
            tmp_mask.scatter_(-1, min_index, 0)
            a2sf.scatter_(-1, min_index, np.inf)

    return tensor

def similarity(tensor_a, tensor_b):
    return torch.sum(torch.multiply(tensor_a, tensor_b))/(torch.norm(tensor_a)*torch.norm(tensor_b) + 1e-10)

model_name = "meta-llama/Llama-2-7b-hf"

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name).half().eval().cuda()

plt.figure(figsize=(9,6))
for idx, dataset in enumerate(["mathqa", "piqa", "arc_challenge", "arc_easy", "openbookqa"]):
    file_path = f"/home/smp9898/A2SF/lm/{dataset}-5.jsonl"

    with open(file_path, "r") as file:
        lines = file.readlines()

    ratios = 0.2
    penalties = torch.arange(0.0, 1.0, 0.05)
    similarities = torch.zeros_like(penalties)
    data_size = 200

    for _ in tqdm(range(data_size)):
        prompt = random.choice(lines)
        
        input_ids = tokenizer(get_prompt(prompt), add_special_tokens=True, return_tensors='pt').input_ids.cuda()

        with torch.no_grad():
            result = model(input_ids, output_attentions=True)

        tensors = torch.stack(result.attentions).squeeze(1)

        tmp = []
        for penalty in penalties:
            masked_tensors = make_mask(tensors, ratios, penalty)
            tmp.append(torch.mean(torch.tensor([similarity(tensors[i], masked_tensors[i]) for i in range(tensors.shape[0])])))
        tmp = torch.tensor(tmp)

        if torch.any(torch.isnan(tmp)):
            continue

        similarities += tmp

    similarities /= data_size

    for i in range(similarities.shape[0]): print(f"{penalties[i]:.2f} {similarities[i]:.4f}")
    print(f"best[{dataset}] : {penalties[torch.argmax(similarities)]:.2f}")

    plt.subplot(2, 3, idx+1)
    plt.title(f"{dataset} : {penalties[torch.argmax(similarities)]:.2f}")
    plt.plot(penalties, similarities)

plt.tight_layout()
plt.savefig(f"forgetting_factor_fixed_ratio_020.png")