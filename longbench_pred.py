import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse

from utils import load_configs, load_model

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument('--model', type=str, default=None, choices=["llama", "llama2", "llama3", "opt", "qwen2"])
    parser.add_argument('--method', type=str, default="a2sf")
    parser.add_argument('--budget', type=int, default=100)
    return parser.parse_args(args)

def build_chat(tokenizer, prompt, model_name):
    if "llama" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    return prompt

def get_pred(data, max_length, max_gen, prompt_format, dataset, model, tokenizer, out_path, args):
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        half = int(max_length/2)
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        prompt = build_chat(tokenizer, prompt, args.model)
        input = tokenizer(prompt, truncation=False, return_tensors="pt")
        
        input_ids = input.input_ids.to(model.device)
        attention_mask = input.attention_mask.to(torch.bfloat16).to(model.device)
        
        context_length = input_ids.shape[-1]
        model.init_cache(load_configs(args.model, args.method, args.budget, tokenizer))
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

from utils import load_model

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    
    # GPU list is already a list of integers
    gpus = args.gpus
    
    # Load configurations
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model_name = args.model
    max_length = model2maxlen[model_name]
    
    # Load model and tokenizer once
    print(f"Loading model and tokenizer for {model_name} on GPUs: {gpus}...")
    model, tokenizer = load_model(model_name, args.gpus)
    print("Model and tokenizer loaded successfully!")
    
    # Define datasets
    datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    
    # Load prompt and max length configurations
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    # Create output directory
    if not os.path.exists("result_txt/pred"):
        os.makedirs("result_txt/pred")
    
    # Process each dataset
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
        output_dir = f"result_txt/pred/{model_name}_{args.method}_{args.budget}"
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_path = f"{output_dir}/{dataset}.jsonl"
        
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        
        # Process data using the pre-loaded model
        get_pred(data, max_length, max_gen, prompt_format, dataset, model, tokenizer, out_path, args)
