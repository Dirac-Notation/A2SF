import copy
import torch
import os
import random
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import math

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

sys.path.append("/home/smp9898/A2SF")

from utils_lm_eval.modify_llama import convert_kvcache_llama_heavy_recent
from utils_lm_eval.ideal_llama import convert_kvcache_llama_heavy_recent_ideal

def get_prompt(json_line):
    data = json.loads(json_line)
    return data["prompt"]

model_name = "meta-llama/Llama-2-7b-hf"

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name).half().eval()
check_point = copy.deepcopy(model.state_dict())
model.cuda()

dir_path = os.path.dirname(__file__)

num_layers = 32
ratio = 0.2

datasets = ["piqa"]#, "openbookqa", "arc_easy", "arc_challenge", "mathqa"]
prompts = []
for dataset in datasets:
    file_path = f"/home/smp9898/A2SF/data/{dataset}-1shot.jsonl"
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    prompt = get_prompt(random.choice(lines))
    prompts.append(tokenizer(prompt, add_special_tokens=True, return_tensors='pt').input_ids.cuda())

methods = {
    "FULL": (0.0, 0.0, 1.0, 1.0, False, None),
    # "STREAMING_LLM": (ratio/2, 0.0, ratio/2, 1.0, False),
    "LOCAL": (0.0, 0.0, ratio, 1.0, False, None),
    "H2O": (0.0, ratio/2, ratio/2, 1.0, False, None),
    "A2SF_010": (0.0, ratio, 0.0, 0.1, False, None),
    "A2SF_050": (0.0, ratio, 0.0, 0.5, False, None),
    # "STREAMING A2SF": (ratio/3, ratio/3, ratio/3, 0.2, False),
    # "A2SF_1": (0.0, ratio/2, ratio/2, 1.0, False, 1),
    # "A2SF_3": (0.0, ratio/2, ratio/2, 1.0, False, 3)
}

column = 3
row = math.ceil(len(methods)/3)
result_dict = {}

for method, (i, j, k, h, ideal, tmp) in methods.items():
    config.streaming_ratio = i
    config.selecting_ratio = j
    config.recent_ratio = k
    config.forgetting_factor = h
    config.tmp = tmp
    
    if ideal:
        convert_kvcache_llama_heavy_recent_ideal(model, config)
    else:
        convert_kvcache_llama_heavy_recent(model, config)
    
    model.load_state_dict(check_point)
    torch.cuda.empty_cache()
    model.half().eval().cuda()

    for prompt in prompts:
        with torch.no_grad():
            result = model(prompt, output_attentions=True)
        
        if method not in result_dict.keys():
            result_dict[method] = []
        result_dict[method].append(result.attentions)

for index in range(len(prompts)):
    for layer in tqdm(range(num_layers)):
        data_dict = {}
        result_path = os.path.join(dir_path, "mask", str(index), str(layer))
        
        for method in methods:
            data_dict[method] = result_dict[method][index][layer].cpu().detach().squeeze(0)

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        for ln in range(32):
            plt.figure(figsize=(column*7, row*7))
            
            for idx, (method, data) in enumerate(data_dict.items()):
                tmp = torch.pow(data[ln], 1/3).numpy()

                plt.subplot(row, column, idx+1)
                plt.title(method, fontsize=20)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(tmp, cmap="Blues")
            plt.tight_layout()
            plt.savefig(os.path.join(result_path, f"test_{ln}.png"))
            plt.close()