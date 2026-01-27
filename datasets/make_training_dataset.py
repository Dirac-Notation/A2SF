#!/usr/bin/env python3
import json
import os
import sys
import random
from typing import Dict, List, Any, Optional
from collections import defaultdict
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils_real_drop import KVLlamaForCausalLM
from utils import CompressionConfig
import numpy as np

PROMPT_LENGTH = 7500
MIN_TOKENS = 1024  # 필터링을 위한 최소 토큰 수
MAX_TOKENS = 7500  # 필터링을 위한 최대 토큰 수

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
    
    print(f"Loading KVLlamaForCausalLM: {model_name} from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = KVLlamaForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model = model.eval()
    
    return model, tokenizer

def count_tokens(tokenizer, text: str) -> int:
    """Count tokens in text"""
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))

def format_prompt_for_task(task_type: str, context: str = "", question: str = "") -> str:
    """Format prompt based on task type, similar to LongBench prompt formats"""
    
    if task_type == "summarization":
        return f"You are given a document and a query. Write a summary based on the query.\n\nDocument:\n{context}\n\nQuery: {question}"
    elif task_type == "qa":
        return f"Answer the question based on the given context. Only give me the answer and do not output any other words.\n\nContext:\n{context}\n\nAnswer the question based on the given context. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:"
    elif task_type == "retrieval":
        return f"Here are some paragraphs, along with a query. Please determine which paragraph the query is from.\n\nParagraphs:\n{context}\n\nThe following is a query.\n\n{question}\n\nPlease enter the number of the paragraph that the query is from.\n\nThe answer is: "
    else:
        return f"{question}"

def load_zeroscrolls_datasets(tokenizer) -> List[Dict[str, Any]]:
    """Load all ZeroSCROLLS datasets from HuggingFace with 8K token filtering"""
    all_samples = []
    
    # All ZeroSCROLLS subsets
    subsets = [
        "gov_report", "summ_screen_fd", "qmsum", "quality", "qasper",
        "narrative_qa", "musique", "hotpot_qa", "space_digest", "book_sum_sort"
    ]
    
    task_type_mapping = {
        "gov_report": "summarization",
        "summ_screen_fd": "summarization",
        "qmsum": "summarization",
        "quality": "qa",
        "qasper": "qa",
        "narrative_qa": "qa",
        "musique": "qa",
        "space_digest": "summarization",
        "book_sum_sort": "retrieval"
    }
    
    for subset in subsets:
        print(f"Loading ZeroSCROLLS subset: {subset}")
        try:
            dataset = load_dataset("tau/zero_scrolls", subset, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  Error loading {subset}: {e}")
            continue
        
        task_type = task_type_mapping.get(subset, "qa")
        
        for example in tqdm(dataset, desc=f"Processing {subset}"):
            # Extract fields from ZeroSCROLLS
            all_input = example.get("input", "")
            # Format prompt
            prompt = format_prompt_for_task(task_type, context="", question=all_input)
            
            # Filter by token length (8K 근처)
            token_count = count_tokens(tokenizer, prompt)
            if token_count < MIN_TOKENS or token_count > MAX_TOKENS:
                continue
            
            sample = {
                "input_prompt": prompt,
                "dataset": f"zeroscrolls_{subset}",
                "source": "ZeroSCROLLS",
                "length": token_count,
                "task_type": task_type
            }
            all_samples.append(sample)
        
        print(f"  Loaded {len([s for s in all_samples if s['dataset'] == f'zeroscrolls_{subset}'])} samples from {subset} (after filtering)")
    
    return all_samples

def load_leval_datasets(tokenizer) -> List[Dict[str, Any]]:
    """Load all L-Eval datasets from HuggingFace with 8K token filtering"""
    all_samples = []
    
    # All available L-Eval subsets (from error message)
    subsets = [
        "codeU", "coursera", "financial_qa", "gov_report_summ", "gsm100",
        "legal_contract_qa", "meeting_summ", "multidoc_qa", "narrative_qa",
        "natural_question", "news_summ", "paper_assistant", "patent_summ",
        "quality", "review_summ", "sci_fi", "scientific_qa", "topic_retrieval_longchat",
        "tpo", "tv_show_summ"
    ]
    
    task_type_mapping = {
        "codeU": "qa",
        "coursera": "qa",
        "financial_qa": "qa",
        "gov_report_summ": "summarization",
        "gsm100": "qa",
        "legal_contract_qa": "qa",
        "meeting_summ": "summarization",
        "multidoc_qa": "qa",
        "narrative_qa": "qa",
        "natural_question": "qa",
        "news_summ": "summarization",
        "paper_assistant": "qa",
        "patent_summ": "summarization",
        "quality": "qa",
        "review_summ": "summarization",
        "sci_fi": "qa",
        "scientific_qa": "qa",
        "topic_retrieval_longchat": "retrieval",
        "tpo": "qa",
        "tv_show_summ": "summarization"
    }
    
    for subset in subsets:
        print(f"Loading L-Eval subset: {subset}")
        try:
            dataset = load_dataset("L4NLP/LEval", subset, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  Error loading {subset}: {e}")
            continue
        
        task_type = task_type_mapping.get(subset, "qa")
        
        for example in tqdm(dataset, desc=f"Processing {subset}"):
            # Extract fields from L-Eval
            context = example.get("input", "")
            question = example.get("instructions", "")
            
            # Format prompt
            prompt = format_prompt_for_task(task_type, context=context, question=question)
            
            # Filter by token length (8K 근처)
            token_count = count_tokens(tokenizer, prompt)
            
            if token_count < MIN_TOKENS or token_count > MAX_TOKENS:
                continue
            
            sample = {
                "input_prompt": prompt,
                "dataset": f"leval_{subset}",
                "source": "L-Eval",
                "length": token_count,
                "task_type": task_type
            }
            all_samples.append(sample)
                
        print(f"  Loaded {len([s for s in all_samples if s['dataset'] == f'leval_{subset}'])} samples from {subset} (after filtering)")
    
    return all_samples

def generate_answer(model, tokenizer, prompt: str, dataset: str, model_name: str, generation_length: int) -> tuple:
    # Initialize full cache (no compression) before each generation
    compression_config = CompressionConfig()
    compression_config.compression_method = "full"
    compression_config.total_budget = None
    compression_config.layerwise_ratios = None
    compression_config.local_ratios = None
    model.init_cache(compression_config)
    
    # Add system prompt for llama models
    if "llama" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    
    input = tokenizer(prompt, truncation=False, return_tensors="pt")
    input_ids = input.input_ids.to(model.device)
    attention_mask = input.attention_mask.to(model.device)
    
    # Generate text using full cache (no compression) with model.generate()
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=generation_length,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            stop_strings="[/INST]",
            tokenizer=tokenizer
        )
    
    # Decode only the generated part (excluding the input prompt)
    context_length = input_ids.size(1)
    
    # model.generate() returns tensor of shape [batch_size, seq_len]
    # Get the first sequence (batch index 0) and only the generated part
    if isinstance(output, torch.Tensor):
        # output shape: [1, total_length] or [total_length]
        if output.dim() == 2:
            # Batch dimension exists
            generated_ids = output[0, context_length:]
        else:
            # No batch dimension
            generated_ids = output[context_length:]
    else:
        # Handle dict output if return_dict_in_generate was used
        generated_ids = output.sequences[0, context_length:] if hasattr(output, 'sequences') else output[0, context_length:]
    
    generated_length = generated_ids.size(0)
    # Decode the generated token IDs
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return prompt, generated_text, generated_length

def main(args):
    set_seed(42)
    
    model_name = args.model
    model, tokenizer = load_model_for_generation(model_name, args.gpus)
    
    print(f"\n{'='*50}")
    print(f"Loading datasets with 8K token filtering ({MIN_TOKENS}-{MAX_TOKENS} tokens)")
    print(f"{'='*50}")
    
    # Load all datasets
    all_samples = []
    
    # Load ZeroSCROLLS
    print(f"\nLoading ZeroSCROLLS datasets...")
    zeroscrolls_samples = load_zeroscrolls_datasets(tokenizer)
    all_samples.extend(zeroscrolls_samples)
    print(f"Total ZeroSCROLLS samples: {len(zeroscrolls_samples)}")
    
    # Load L-Eval
    print(f"\nLoading L-Eval datasets...")
    leval_samples = load_leval_datasets(tokenizer)
    all_samples.extend(leval_samples)
    print(f"Total L-Eval samples: {len(leval_samples)}")
    
    print(f"\n{'='*50}")
    print(f"Total samples collected: {len(all_samples)}")
    print(f"{'='*50}")
    
    # Process all samples and generate training data
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    total_samples = 0
    processed_counts = {}
    
    task_length_map = {
        "summarization": 512,
        "retrieval": 32,
        "qa": 128,
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(all_samples, desc="Generating training data"):
            dataset_name = sample.get("dataset", "unknown")
            prompt = sample["input_prompt"]
            generation_length = task_length_map[sample["task_type"]]
            
            try:
                prompt, generated_text, generated_length = generate_answer(model, tokenizer, prompt, dataset_name, model_name, generation_length)
                
                training_sample = {
                    "dataset": dataset_name,
                    "input_prompt": prompt,
                    "generated_text": generated_text,
                    "generation_length": generated_length
                }
                
                f.write(json.dumps(training_sample, ensure_ascii=False, separators=(',', ':')) + "\n")
                f.flush()
                
                total_samples += 1
                if dataset_name not in processed_counts:
                    processed_counts[dataset_name] = 0
                processed_counts[dataset_name] += 1
                
            except Exception as e:
                print(f"\nError processing sample: {e}")
                continue
    
    print(f"\n{'='*50}")
    print(f"Training data generation complete!")
    print(f"Total samples generated: {total_samples}")
    print(f"Saved to: {args.output_file}")
    print(f"{'='*50}")
    
    print(f"\nSamples per dataset:")
    for dataset, count in sorted(processed_counts.items()):
        print(f"  {dataset}: {count}")
    
    # Print detailed statistics after generation
    print(f"\n{'='*50}")
    print("Generating detailed statistics...")
    print(f"{'='*50}")
    check_training_data_stats(args.output_file)

def check_training_data_stats(file_path):
    """training_data.jsonl 파일에서 dataset별 데이터 개수를 확인"""
    dataset_counts = defaultdict(int)
    total_lines = 0
    
    print(f"Reading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                # JSON 파싱 (실제 데이터 내용은 읽지 않고 구조만 확인)
                data = json.loads(line)
                dataset = data.get("dataset", "unknown")
                dataset_counts[dataset] += 1
                total_lines += 1
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON: {e}")
                continue
    
    print(f"\n{'='*60}")
    print(f"Total samples: {total_lines}")
    print(f"{'='*60}\n")
    
    print("Dataset counts:")
    print("-" * 60)
    for dataset in sorted(dataset_counts.keys()):
        count = dataset_counts[dataset]
        percentage = (count / total_lines * 100) if total_lines > 0 else 0
        print(f"  {dataset:40s}: {count:5d} ({percentage:5.2f}%)")
    
    print("\n" + "="*60)
    
    # Task type별로도 집계 (dataset 이름에서 추론)
    task_type_mapping = {
        "gov_report": "summarization",
        "summ_screen_fd": "summarization",
        "qmsum": "summarization",
        "space_digest": "summarization",
        "book_sum_sort": "retrieval",
        "quality": "qa",
        "qasper": "qa",
        "narrative_qa": "qa",
        "musique": "qa",
        "hotpot_qa": "qa",
        "codeU": "qa",
        "coursera": "qa",
        "financial_qa": "qa",
        "gov_report_summ": "summarization",
        "gsm100": "qa",
        "legal_contract_qa": "qa",
        "meeting_summ": "summarization",
        "multidoc_qa": "qa",
        "natural_question": "qa",
        "news_summ": "summarization",
        "paper_assistant": "qa",
        "patent_summ": "summarization",
        "review_summ": "summarization",
        "sci_fi": "qa",
        "scientific_qa": "qa",
        "topic_retrieval_longchat": "retrieval",
        "tpo": "qa",
        "tv_show_summ": "summarization"
    }
    
    task_counts = defaultdict(int)
    for dataset, count in dataset_counts.items():
        # dataset 이름에서 subset 추출 (zeroscrolls_xxx 또는 leval_xxx)
        if dataset.startswith("zeroscrolls_"):
            subset = dataset.replace("zeroscrolls_", "")
        elif dataset.startswith("leval_"):
            subset = dataset.replace("leval_", "")
        else:
            subset = dataset
        
        task_type = task_type_mapping.get(subset, "unknown")
        task_counts[task_type] += count
    
    print("\nTask type counts:")
    print("-" * 60)
    for task_type in sorted(task_counts.keys()):
        count = task_counts[task_type]
        percentage = (count / total_lines * 100) if total_lines > 0 else 0
        print(f"  {task_type:20s}: {count:5d} ({percentage:5.2f}%)")
    
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data from ZeroSCROLLS and L-Eval datasets, or check statistics")
    parser.add_argument("--output_file", type=str, default="./datasets/training_data.jsonl", help="Output file for training data")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of GPU IDs")
    parser.add_argument("--model", type=str, default="llama3", choices=["llama", "llama2", "llama3", "opt"], help="Model to use")
    parser.add_argument("--stats-only", action="store_true", help="Only check statistics of existing file without generating new data")
    
    args = parser.parse_args()
    
    if args.stats_only:
        # Only check statistics
        check_training_data_stats(args.output_file)
    else:
        # Generate training data (original main function)
        main(args)
