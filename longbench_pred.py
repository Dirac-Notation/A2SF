import os
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse

from utils import load_configs, load_model
from models_skew.skew_llama import skew_model

def load_jsonl_file(file_path):
    """Load data from a JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument('--model', type=str, default="llama3")
    parser.add_argument('--method', type=str, default="full")
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--task', type=str, choices=["Code Complete", "Few Shot", "Single-doc QA", "Multi-doc QA", "Summarization", "Passage Retrieval"])
    return parser.parse_args(args)

def build_chat(prompt, model_name):
    if "llama" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    return prompt

def get_pred(data, max_length, max_gen, dataset, model, tokenizer, out_path, args):
    for json_obj in tqdm(data):
        # Use the already formatted prompt from the jsonl file
        prompt = json_obj["input_prompt"]
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(prompt, args.model)
        input = tokenizer(prompt, truncation=False, return_tensors="pt")
        
        input_ids = input.input_ids.to(model.device)
        attention_mask = input.attention_mask.to(torch.bfloat16).to(model.device)
        
        context_length = input_ids.shape[-1]
        if hasattr(model, "init_cache"):
            model.init_cache(load_configs(args.model, args.method, args.task, args.budget))
        with torch.inference_mode():
            if dataset == "samsum":
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=0.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=0.0,
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
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    
    # Load configurations
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model_name = args.model
    max_length = model2maxlen[model_name]
    
    # Load model and tokenizer once
    print(f"Loading model and tokenizer for {model_name} on GPUs: {gpus}...")
    model, tokenizer = load_model(model_name, args.gpus)
    print("Model and tokenizer loaded successfully!")

    data_group = {
        "Code Complete": ["repobench-p", "lcc"],
        "Few Shot": ["trec", "triviaqa", "samsum", "lsht"],
        "Single-doc QA": ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
        "Multi-doc QA": ["hotpotqa", "2wikimqa", "musique", "dureader"],
        "Summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
        "Passage Retrieval": ["passage_retrieval_en", "passage_retrieval_zh", "passage_count"],
    }
    
    datasets = data_group[args.task]

    # # Define datasets
    # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
    #             "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
    #             "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    
    # Load prompt and max length configurations
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    # Create output directory
    if not os.path.exists("result_txt/pred"):
        os.makedirs("result_txt/pred")
    
    # Process each dataset
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        # Load data from local jsonl file
        jsonl_path = f"datasets/longbench/{dataset}.jsonl"
        if not os.path.exists(jsonl_path):
            print(f"Warning: {jsonl_path} not found, skipping {dataset}")
            continue
        data = load_jsonl_file(jsonl_path)
        output_dir = f"result_txt/pred/{model_name}_{args.method}_{args.budget}"
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_path = f"{output_dir}/{dataset}.jsonl"
        
        max_gen = dataset2maxlen[dataset]
        
        # Process data using the pre-loaded model
        get_pred(data, max_length, max_gen, dataset, model, tokenizer, out_path, args)
