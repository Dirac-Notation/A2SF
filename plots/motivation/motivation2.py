from curses import window
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
PROMPT_LENGTH = 7500
TOTAL_BUDGET = 128
GENERATION_LENGTH = int(TOTAL_BUDGET * 0.125)
SELECT_BUDGET = TOTAL_BUDGET - GENERATION_LENGTH
NUM_BOUNDS = 10

def get_prompt(task, data_group):
    articles = []
    datasets = data_group.get(task, [])
    
    for dataset in datasets:
        file_path = f"datasets/longbench/{dataset}.jsonl"
        try:
            with open(file_path, "r") as f:
                for line in f:
                    line_data = json.loads(line)
                    article = line_data["input_prompt"]
                    articles.append(article)
        except FileNotFoundError:
            print(f"Warning: Dataset file {file_path} not found")
            continue
    
    return random.sample(articles, 100)

def load_model_and_tokenizer(model_name):
    model2path = json.load(open("config/model2path.json", "r"))
    model_path = model2path[model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def logits_to_tokens(logits):
    return torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)

def process_prompt(model, tokenizer, prompt):
    with torch.inference_mode():
        if "llama" in model.config.model_type.lower():
            prompt = f"[INST]{prompt}[/INST]"
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        if input_ids.numel() > PROMPT_LENGTH:
            input_ids = torch.cat([input_ids[:, :PROMPT_LENGTH//2], input_ids[:, -PROMPT_LENGTH//2:]], dim=1)
        prompt_length = input_ids.numel()
        attention_maps = []
        with torch.no_grad():
            input_ids = input_ids.to(model.device)
            outputs = model(input_ids)
            for _ in range(GENERATION_LENGTH):
                input_ids = logits_to_tokens(outputs.logits)
                outputs = model(input_ids, past_key_values=outputs.past_key_values, output_attentions=True)
                attention_maps.append(torch.stack([attention.cpu().squeeze(0) for attention in outputs.attentions], dim=0))
        num_layers, num_heads, _, seq_len = attention_maps[-1].shape
        
        out = torch.zeros(num_layers, num_heads, GENERATION_LENGTH, seq_len)
        for idx, att_map in enumerate(attention_maps):
            cur_len = att_map.size(-1)
            out[:,:,idx,:cur_len] = att_map.squeeze(2)
        
        return out, prompt_length

def process_model(model, tokenizer, prompts, task):
    window_results = torch.zeros(NUM_BOUNDS)
    
    for prompt in tqdm(prompts, desc=f"Processing prompts for {task}"):
        attention_maps, prompt_length = process_prompt(model, tokenizer, prompt)
        
        score = attention_maps[:,:,:,:-GENERATION_LENGTH].sum(dim=2)
        
        selected_indices = score.topk(k=SELECT_BUDGET, dim=2).indices
        
        bounds = [int(prompt_length/NUM_BOUNDS)*(i+1) for i in range(NUM_BOUNDS)]
        
        prev_ratio = 0
        for b_idx, b in enumerate(bounds):
            # cur_ratio = ((selected_indices < b).sum(dim=2) / SELECT_BUDGET).mean() - prev_ratio
            # prev_ratio += cur_ratio
            cur_ratio = ((selected_indices < b).sum(dim=2) / SELECT_BUDGET).mean()
            window_results[b_idx] += cur_ratio
    
    window_results /= len(prompts)
    return window_results

def create_combined_plot(all_results):
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 15,
        'figure.titlesize': 20
    })
    
    x = range(NUM_BOUNDS)
    plt.figure(figsize=(15, 8))
    
    # 각 task별로 다른 색상 사용
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 바 그래프를 위한 데이터 준비
    task_names = list(all_results.keys())
    num_tasks = len(task_names)
    bar_width = 0.8 / num_tasks  # 바 너비 조정
    
    for i, (task, results) in enumerate(all_results.items()):
        # x_pos = [pos + (i - num_tasks/2 + 0.5) * bar_width for pos in x]
        # plt.bar(x_pos, results, bar_width, label=task, color=colors[i], alpha=0.8)
        plt.plot(x, results, label=task, color=colors[i], linewidth=2.5)
    
    plt.xticks(x, [f"~{10*b+10}%" for b in range(NUM_BOUNDS)])
    plt.ylim(0, 1)
    plt.xlabel("Observation Window Size")
    plt.ylabel("Ratio of Selected Tokens")
    plt.title("Attention Pattern Analysis Across Different Tasks")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("plots/motivation/Combined_Tasks.png", dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    data_group = {
        "Code Complete": ["repobench-p", "lcc"],
        "Few Shot": ["trec", "triviaqa", "samsum"],
        "Single-doc QA": ["narrativeqa", "qasper", "multifieldqa_en"],
        "Multi-doc QA": ["hotpotqa", "2wikimqa", "musique"],
        "Summarization": ["gov_report", "qmsum", "multi_news"],
        "Passage Retrieval": ["passage_retrieval_en", "passage_count"],
    }
  
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # 모든 task의 결과를 저장할 딕셔너리
    all_results = {}

    for task in data_group:
        prompts = get_prompt(task, data_group)
        window_results = process_model(
            model = model,
            tokenizer = tokenizer,
            prompts = prompts,
            task = task
        )
        all_results[task] = window_results.numpy()  # torch tensor를 numpy로 변환
    
    # 모든 task의 결과를 하나의 그래프에 표시
    create_combined_plot(all_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument("--model", type=str, default="llama2", choices=["llama", "llama2", "llama3", "opt", "qwen2"])
    args = parser.parse_args()
    
    gpu_list = ",".join(str(gpu) for gpu in args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    
    main(args)