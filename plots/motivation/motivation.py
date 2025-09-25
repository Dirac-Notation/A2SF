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
PROMPT_LENGTH = 4096
TOTAL_BUDGET = 128
LOCAL_WINDOW = int(TOTAL_BUDGET * 0.125)

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

def load_model_and_tokenizer(model_name):
    model2path = json.load(open("config/model2path.json", "r"))
    model_path = model2path[model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def process_prompt(model, tokenizer, prompt):
    with torch.inference_mode():
        if "llama" in model.config.model_type.lower():
            prompt = f"[INST]{prompt}[/INST]"
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        input_ids = torch.cat([input_ids[:, :PROMPT_LENGTH//2], input_ids[:, -PROMPT_LENGTH//2:]], dim=1)
        
        with torch.no_grad():
            outputs = model(input_ids.to(model.device), output_attentions=True)
        
        attention_maps = torch.stack([attention.cpu() for attention in outputs.attentions], dim=0)

        return attention_maps

def process_model(model, tokenizer, prompts, task):
    snap_windows = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
    bounds = [512*(i+1) for i in range(8)]
    window_results = torch.zeros(len(snap_windows), len(bounds))
    
    for prompt in tqdm(prompts, desc="Processing prompts"):
        attention_maps = process_prompt(model, tokenizer, prompt)
        for snap_idx, snap_window in enumerate(snap_windows):
            score = attention_maps[:,:,:,-snap_window:,:].sum(dim=3)
            score[:,:,:,-LOCAL_WINDOW:] = score.max()
            
            selected_indices = score.topk(k=TOTAL_BUDGET, dim=3).indices
            
            for b_idx, b in enumerate(bounds):
                window_results[snap_idx, b_idx] += ((selected_indices < b).sum(dim=3) / TOTAL_BUDGET).mean()
    
    window_results /= len(prompts)

    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 15,
        'figure.titlesize': 20
    })
    
    plt.figure(figsize=(12, 7))
    for i in range(len(bounds)):
        plt.plot(range(len(snap_windows)), window_results[:,i])
        if i == 0:
            plt.fill_between(range(len(snap_windows)), window_results[:,i]*0, window_results[:,i], alpha=0.5, label=f"0 ~ {bounds[i]}")
        else:
            plt.fill_between(range(len(snap_windows)), window_results[:,i-1], window_results[:,i], alpha=0.5, label=f"{bounds[i-1]} ~ {bounds[i]}")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xticks(range(len(snap_windows)), snap_windows)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    plt.xlabel("Observation Window Size")
    plt.ylabel("Ratio of Selected Tokens")
    # plt.title(f"{task}")
    plt.savefig(f"plots/motivation/{task}.png")
    plt.close()

def main(args):
    tasks = ["Code Complete", "Few Shot", "Single-doc QA", "Multi-doc QA", "Summarization", "Passage Retrieval"]
  
    model, tokenizer = load_model_and_tokenizer(args.model)

    for task in tasks:
        prompts = get_prompt(task)
        process_model(
            model = model,
            tokenizer = tokenizer,
            prompts = prompts,
            task = task
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument("--model", type=str, default="llama2", choices=["llama", "llama2", "llama3", "opt", "qwen2"])
    args = parser.parse_args()
    
    gpu_list = ",".join(str(gpu) for gpu in args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    
    main(args)