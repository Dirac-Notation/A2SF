#!/usr/bin/env python3
import json
import os
import random
from typing import Dict, List, Any
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

PROMPT_LENGTH = 7500
TOTAL_BUDGET = 128
GENERATION_LENGTHS = [128]
MAX_GENERATION_LENGTH = max(GENERATION_LENGTHS)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_for_generation(model_name, gpu_list=None):
    with open('./config/model2path.json', 'r') as f:
        model2path = json.load(f)
    
    model_path = model2path[model_name]
    
    if gpu_list is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
    
    print(f"Loading model: {model_name} from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model = model.eval()
    return model, tokenizer

def logits_to_tokens(logits):
    return torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)

def generate_answer(model, tokenizer, prompt: str, dataset: str, model_name: str) -> tuple:
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    
    if len(tokenized_prompt) > PROMPT_LENGTH:
        half = int(PROMPT_LENGTH/2)
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    
    if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        if "llama" in model_name:
            prompt = f"[INST]{prompt}[/INST]"
    
    input = tokenizer(prompt, truncation=False, return_tensors="pt")
    input_ids = input.input_ids.to(model.device)
    
    attention_maps = []
    with torch.no_grad():
        input_ids = input_ids.to(model.device)
        outputs = model(input_ids)
        input_ids = logits_to_tokens(outputs.logits)
        for _ in range(MAX_GENERATION_LENGTH):
            outputs = model(input_ids, past_key_values=outputs.past_key_values, output_attentions=True)
            attention_maps.append(torch.stack([attention.cpu().squeeze(0) for attention in outputs.attentions], dim=0))
    
    results = {}
    for gen_len in GENERATION_LENGTHS:
        out = attention_maps[gen_len - 1].clone()
        for idx in range(gen_len - 1):
            out[:,:,:,:-(gen_len-idx-1)] += attention_maps[idx]
        
        selected_indices = out[:,:,:,:-gen_len].topk(k=TOTAL_BUDGET, dim=3).indices.squeeze(2)
        results[gen_len] = selected_indices
    
    return prompt, results

def load_longbench_dataset(dataset_name: str, num_samples: int = 10) -> List[Dict[str, Any]]:
    dataset_path = f"./datasets/longbench/{dataset_name}.jsonl"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        return []
    
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line.strip())
                samples.append(sample)
    
    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)
    
    return samples

def process_dataset(
    dataset_name: str, 
    samples: List[Dict[str, Any]], 
    model, 
    tokenizer, 
    model_name: str,
    output_file_handle
) -> int:
    count = 0
    print(f"Processing {dataset_name}")
    
    for i, sample in enumerate(tqdm(samples, desc=f"Processing {dataset_name}")):
        prompt = sample["input_prompt"]
        prompt, selected_indices_dict = generate_answer(model, tokenizer, prompt, dataset_name, model_name)
        
        for gen_len in GENERATION_LENGTHS:
            selected_indices = selected_indices_dict[gen_len]
            
            training_sample = {
                "dataset": dataset_name,
                "input_prompt": prompt,
                "selected_indices": selected_indices.cpu().numpy().tolist(),
                "generation_length": gen_len,
            }
            
            # separators 옵션을 추가하여 공백 없이 한 줄로 저장
            output_file_handle.write(json.dumps(training_sample, ensure_ascii=False, separators=(',', ':')) + "\n")
            output_file_handle.flush()
            count += 1
    
    return count

def main():
    parser = argparse.ArgumentParser(description="Generate training data from LongBench datasets")
    parser.add_argument("--output_file", type=str, default="./datasets/training_data.jsonl", help="Output file for training data")
    parser.add_argument("--num_samples_per_dataset", type=int, default=10,help="Number of samples per dataset")
    parser.add_argument("--seed", type=int, default=42,help="Random seed")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of GPU IDs")
    parser.add_argument("--model", type=str, default="llama3", choices=["llama", "llama2", "llama3", "opt"], help="Model to use")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    model_name = args.model
    model, tokenizer = load_model_for_generation(model_name, args.gpus)
    
    dataset_names = os.listdir("./datasets/longbench")
    dataset_names = [dataset_name.replace(".jsonl", "") for dataset_name in dataset_names]
    print(f"Processing {len(dataset_names)} datasets: {dataset_names}")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    total_samples = 0
    dataset_counts = {}
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for dataset_name in dataset_names:
            print(f"\n{'='*50}")
            print(f"Processing dataset: {dataset_name}")
            print(f"{'='*50}")
            
            samples = load_longbench_dataset(dataset_name, args.num_samples_per_dataset)
            
            if not samples:
                print(f"No samples loaded for {dataset_name}, skipping...")
                continue
            
            count = process_dataset(dataset_name, samples, model, tokenizer, model_name, f)
            
            dataset_counts[dataset_name] = count
            total_samples += count
            print(f"Generated {count} training samples for {dataset_name}")
    
    print(f"\n{'='*50}")
    print(f"Training data generation complete!")
    print(f"Total samples: {total_samples}")
    print(f"Saved to: {args.output_file}")
    print(f"{'='*50}")
    
    print(f"\nSamples per dataset:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"  {dataset}: {count}")

if __name__ == "__main__":
    main()