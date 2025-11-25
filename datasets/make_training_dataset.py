#!/usr/bin/env python3
"""
Generate training data using models on LongBench datasets following longbench_pred.py approach.
Creates samples per dataset with answers generated using dataset2maxlen and model2maxlen configs.
Supports multiple models (llama, llama2, llama3, opt) with proper prompt formatting and generation parameters.
"""

import json
import os
import random
from typing import Dict, List, Any
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

from tqdm import tqdm

PROMPT_LENGTH = 7500
TOTAL_BUDGET = 128
GENERATION_LENGTH = int(TOTAL_BUDGET * 0.125)

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_for_generation(model_name, gpu_list=None):
    """Load model and tokenizer directly"""
    # Load model2path config
    with open('/root/A2SF/config/model2path.json', 'r') as f:
        model2path = json.load(f)
    
    model_path = model2path[model_name]
    
    if gpu_list is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
    
    print(f"Loading model: {model_name} from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model = model.eval()
    
    return model, tokenizer

def logits_to_tokens(logits):
    return torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)

def generate_answer(model, tokenizer, prompt: str, dataset: str, model_name: str) -> str:
    """Generate answer using model following longbench_pred.py approach"""
    # Tokenize prompt without truncation first
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    
    # Handle truncation like longbench_pred.py
    if len(tokenized_prompt) > PROMPT_LENGTH:
        half = int(PROMPT_LENGTH/2)
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    
    # Add [INST] tags for llama models (except specific datasets)
    if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        if "llama" in model_name:
            prompt = f"[INST]{prompt}[/INST]"
    
    # Tokenize final prompt
    input = tokenizer(prompt, truncation=False, return_tensors="pt")
    input_ids = input.input_ids.to(model.device)
    
    attention_maps = []
    # Generate following longbench_pred.py approach (deterministic)
    with torch.no_grad():
        input_ids = input_ids.to(model.device)
        outputs = model(input_ids)
        input_ids = logits_to_tokens(outputs.logits)
        for _ in range(GENERATION_LENGTH):
            outputs = model(input_ids, past_key_values=outputs.past_key_values, output_attentions=True)
            attention_maps.append(torch.stack([attention.cpu().squeeze(0) for attention in outputs.attentions], dim=0))
    
    out = attention_maps[-1]
    for idx in range(GENERATION_LENGTH-1):
        out[:,:,:,:-(GENERATION_LENGTH-idx-1)] += attention_maps[idx]
    
    selected_indices = out[:,:,:,:-GENERATION_LENGTH].topk(k=TOTAL_BUDGET, dim=3).indices.squeeze(2)
    
    return prompt, selected_indices

def load_longbench_dataset(dataset_name: str, num_samples: int = 10) -> List[Dict[str, Any]]:
    """Load samples from existing LongBench dataset files"""
    # Load from existing jsonl file
    dataset_path = f"/root/A2SF/datasets/longbench/{dataset_name}.jsonl"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        return []
    
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line.strip())
                samples.append(sample)
    
    # Sample random examples
    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)
    
    return samples

def process_dataset(
    dataset_name: str, 
    samples: List[Dict[str, Any]], 
    model, 
    tokenizer, 
    model_name: str
) -> List[Dict[str, Any]]:
    """Process samples from a dataset to generate training data"""
    training_data = []
    
    print(f"Processing {dataset_name}")
    
    for i, sample in enumerate(tqdm(samples, desc=f"Processing {dataset_name}")):
        prompt = sample["input_prompt"]
        
        # Generate answer using model
        prompt, selected_indices = generate_answer(model, tokenizer, prompt, dataset_name, model_name)
        
        # Create training sample following longbench_pred.py output format
        training_sample = {
            "dataset": dataset_name,
            "input_prompt": prompt,
            "selected_indices": selected_indices.cpu().numpy().tolist(),  # Generated prediction
        }
        
        training_data.append(training_sample)
    
    return training_data

def main():
    parser = argparse.ArgumentParser(description="Generate training data from LongBench datasets")
    parser.add_argument("--output_file", type=str, default="/root/A2SF/datasets/training_data.json", help="Output file for training data")
    parser.add_argument("--num_samples_per_dataset", type=int, default=10,help="Number of samples per dataset")
    parser.add_argument("--seed", type=int, default=42,help="Random seed")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument("--model", type=str, default="llama3", choices=["llama", "llama2", "llama3", "opt"], help="Model to use for generation")
    
    args = parser.parse_args()
    
    # Set random seed using utils.py
    set_seed(args.seed)
    
    # Load model
    model_name = args.model
    model, tokenizer = load_model_for_generation(model_name, args.gpus)
    
    # Get all dataset names
    dataset_names = os.listdir("/root/A2SF/datasets/longbench")
    dataset_names = [dataset_name.replace(".jsonl", "") for dataset_name in dataset_names]
    print(f"Processing {len(dataset_names)} datasets: {dataset_names}")
    
    all_training_data = []
    
    for dataset_name in dataset_names:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        # Load dataset samples
        samples = load_longbench_dataset(dataset_name, args.num_samples_per_dataset)
        
        if not samples:
            print(f"No samples loaded for {dataset_name}, skipping...")
            continue
        
        # Process samples
        training_data = process_dataset(dataset_name, samples, model, tokenizer, model_name)
        
        all_training_data.extend(training_data)
        print(f"Generated {len(training_data)} training samples for {dataset_name}")
    
    # Save training data
    print(f"\n{'='*50}")
    print(f"Saving {len(all_training_data)} total training samples to {args.output_file}")
    print(f"{'='*50}")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_training_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\nTraining data generation complete!")
    print(f"Total samples: {len(all_training_data)}")
    
    # Count by dataset
    dataset_counts = {}
    for sample in all_training_data:
        dataset = sample["dataset"]
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    
    print(f"\nSamples per dataset:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"  {dataset}: {count}")

if __name__ == "__main__":
    main()
