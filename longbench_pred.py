import os
import json
from tqdm import tqdm
import argparse
import torch

from utils import load_configs, load_model, set_seed

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument('--model', type=str, required=True, choices=["llama", "llama2", "llama3", "opt"])
    parser.add_argument('--method', type=str, default="full")
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--task', type=str, nargs='+', required=True, choices=["Code Complete", "Few Shot", "Single-doc QA", "Multi-doc QA", "Passage Retrieval", "Summarization"])
    return parser.parse_args(args)

def load_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_pred(data, max_length, max_gen, dataset, model, tokenizer, out_path, model_name, config):
    for json_obj in tqdm(data):
        prompt = json_obj["input_prompt"]
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            if "llama" in model_name:
                prompt = f"[INST]{prompt}[/INST]"
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt")
        
        input_ids = input.input_ids.to(model.device)
        attention_mask = input.attention_mask.to(torch.bfloat16).to(model.device)
        
        context_length = input_ids.shape[-1]
        
        model.init_cache(config)
        
        with torch.inference_mode():
            if dataset == "samsum":
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
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
                    temperature=1.0,
                )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

if __name__ == '__main__':
    set_seed(42)
    args = parse_args()
    
    gpus = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    
    model_name = args.model
    model_name = model_name.split("_")[0].lower()
    max_length = json.load(open("config/model2maxlen.json", "r"))[model_name]
    
    model, tokenizer = load_model(model_name, args.gpus)

    data_group = {
        "Code Complete": ["repobench-p", "lcc"],
        "Few Shot": ["trec", "triviaqa", "samsum"],
        "Single-doc QA": ["narrativeqa", "qasper", "multifieldqa_en"],
        "Multi-doc QA": ["hotpotqa", "2wikimqa", "musique"],
        "Summarization": ["gov_report", "qmsum", "multi_news"],
        "Passage Retrieval": ["passage_retrieval_en", "passage_count"],
    }
    
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    if not os.path.exists("result_txt/pred"):
        os.makedirs("result_txt/pred")
    
    for task in args.task:
        config = load_configs(args.config_file, args.method, args.budget, task)
        
        datasets = data_group[task]
        
        for dataset in datasets:
            print(f"\nProcessing dataset: {dataset}")
            
            jsonl_path = f"datasets/longbench/{dataset}.jsonl"
            if not os.path.exists(jsonl_path):
                print(f"Warning: {jsonl_path} not found, skipping {dataset}")
                continue
            data = load_jsonl_file(jsonl_path)
            output_dir = f"result_txt/pred/{args.model}_{args.method}_{args.budget}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            out_path = f"{output_dir}/{dataset}.jsonl"
            
            max_gen = dataset2maxlen[dataset]
            
            get_pred(data, max_length, max_gen, dataset, model, tokenizer, out_path, model_name, config)
