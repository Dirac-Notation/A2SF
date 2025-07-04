import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import random
import argparse

from utils_real_drop import KVLlamaForCausalLM
from utils import load_configs

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model', type=str, default=None, choices=["llama", "llama2", "llama3", "opt", "qwen2"])
    parser.add_argument('--method', type=str, default="a2sf")
    parser.add_argument('--total_budget', type=int, default=100)
    return parser.parse_args(args)

def build_chat(tokenizer, prompt, model_name):
    if "llama" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    return prompt

def get_pred(data, max_length, max_gen, prompt_format, dataset, model, tokenizer, out_path, args):
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, args.model)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
        context_length = input.input_ids.shape[-1]
        model.init_cache(load_configs(args.model, args.method, args.total_budget, tokenizer))
        with torch.inference_mode():
            if dataset == "samsum":
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model.generate(
                    **input,
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

def load_model_and_tokenizer(path, model_name, device):
    device_1 = 4
    device_2 = 5
    device_map = {f"model.layers.{i}": device_1 for i in range(0,16)}
    device_map.update({f"model.layers.{i}": device_2 for i in range(16,32)})
    device_map["model.embed_tokens"] = device_1
    device_map["model.norm"] = device_2
    device_map["lm_head"] = device_2

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = KVLlamaForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Load configurations
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model_name = args.model
    max_length = model2maxlen[model_name]
    
    # Load model and tokenizer once
    print(f"Loading model and tokenizer for {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    print("Model and tokenizer loaded successfully!")
    
    # Define datasets
    datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    
    # Load prompt and max length configurations
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    # Create output directory
    if not os.path.exists("result_txt/pred"):
        os.makedirs("result_txt/pred")
    
    # Process each dataset
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        output_dir = f"result_txt/pred/{model_name}_{args.method}_{args.total_budget}"
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_path = f"{output_dir}/{dataset}.jsonl"
        
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        
        # Process data using the pre-loaded model
        get_pred(data, max_length, max_gen, prompt_format, dataset, model, tokenizer, out_path, args)
