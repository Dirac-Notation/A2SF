#!/usr/bin/env python3
import json
import os
import sys
import random
from typing import Dict, List, Any, Optional
from collections import defaultdict
import torch
import argparse
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

PROMPT_LENGTH = 7500
MIN_TOKENS = 4000   # 필터링을 위한 최소 토큰 수
MAX_TOKENS = 7500  # 필터링을 위한 최대 토큰 수

MAX_LEN = {
    "summarization": 512,
    "qa": 64,
    "retrieval": 32
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_for_generation(model_name):
    with open('./config/model2path.json', 'r') as f:
        model2path = json.load(f)
    
    model_path = model2path[model_name]
    
    print(f"Loading KVLlamaForCausalLM: {model_name} from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
    )
    model = model.eval()

    # Ensure pad_token is set for batch padding (use EOS as PAD if not set)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    
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
        "narrative_qa", "musique", "space_digest", "book_sum_sort"
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
            
            # Filter by token length
            token_count = count_tokens(tokenizer, prompt)
            if token_count < MIN_TOKENS or token_count > MAX_TOKENS:
                continue
            
            output_count = MAX_LEN[task_type]

            sample = {
                "input_prompt": prompt,
                "generation_length": output_count,
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
            
            output_count = MAX_LEN[task_type]

            sample = {
                "input_prompt": prompt,
                "generation_length": output_count,
                "dataset": f"leval_{subset}",
                "source": "L-Eval",
                "length": token_count,
                "task_type": task_type
            }
            all_samples.append(sample)
                
        print(f"  Loaded {len([s for s in all_samples if s['dataset'] == f'leval_{subset}'])} samples from {subset} (after filtering)")
    
    return all_samples

def generate_answer(model, tokenizer, sample: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Generate answer for a single sample using full cache (no compression).
    sample should contain: "input_prompt", "generation_length", "dataset", "task_type"
    """
    prompt = sample["input_prompt"]
    generation_length = sample["generation_length"]
    dataset_name = sample.get("dataset", "unknown")
    task_type = sample.get("task_type", "unknown")

    # Add system prompt for llama models
    if "llama" in model_name:
        prompt_for_gen = f"[INST]{prompt}[/INST]"
    else:
        prompt_for_gen = prompt

    encoded = tokenizer(prompt_for_gen, truncation=False, return_tensors="pt")
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)

    # Prefill 단계에서는 attention을 저장하지 않고 cache만 쌓고,
    # 이후 디코딩 단계에서만 attention을 모으기 위한 수동 decoding 루프
    with torch.no_grad():
        # 1) Prefill: 전체 프롬프트에 대해 한 번만 forward (attention X, cache O)
        prefill_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=False,
        )
        past_key_values = prefill_outputs.past_key_values

        # 첫 번째 생성 토큰은 prefill의 마지막 토큰 logits에서 greedy로 선택
        logits = prefill_outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated_tokens = []
        all_decode_attentions = []

        # attention mask는 생성될 때마다 길이를 1씩 늘려서 유지
        attention_mask_full = attention_mask

        # 디코딩 루프: 디코딩 단계에서만 output_attentions=True
        for step in range(generation_length):
            if step == 0:
                cur_input_ids = next_token
            else:
                # 직전 step의 logits에서 greedy decoding
                logits = outputs.logits[:, -1, :]
                cur_input_ids = torch.argmax(logits, dim=-1, keepdim=True)

            generated_tokens.append(cur_input_ids)

            # 전체 sequence 길이에 맞게 attention_mask 확장
            attention_mask_full = torch.cat(
                [
                    attention_mask_full,
                    attention_mask_full.new_ones((attention_mask_full.size(0), 1)),
                ],
                dim=-1,
            )

            outputs = model(
                input_ids=cur_input_ids,
                attention_mask=attention_mask_full,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )
            past_key_values = outputs.past_key_values

            # 디코딩 단계의 attention만 저장
            all_decode_attentions.append(outputs.attentions)

            # 종료 조건 1: EOS 토큰
            if tokenizer.eos_token_id is not None and (cur_input_ids == tokenizer.eos_token_id).all():
                break

            # 종료 조건 2: 생성된 텍스트 안에 '[/INST]'가 나타나는 경우
            generated_ids_tensor = torch.cat(generated_tokens, dim=-1)
            decoded_text_step = tokenizer.decode(
                generated_ids_tensor[0],
                skip_special_tokens=False
            )
            if "[/INST]" in decoded_text_step:
                break

        if generated_tokens:
            generated_ids = torch.cat(generated_tokens, dim=-1)
        else:
            generated_ids = input_ids.new_zeros((1, 0))

        # 전체 시퀀스 (프롬프트 + 생성 토큰)
        output_ids = torch.cat([input_ids, generated_ids], dim=-1)

        # 디코딩에서만 모은 attention을 확인할 수 있도록 변수에 남겨둠
        output = {
            "sequences": output_ids,
            "decode_attentions": [torch.cat(all_decode_attention, dim=0) for all_decode_attention in all_decode_attentions],
        }

    first_decode_attention = output["decode_attentions"][0]
    scores = torch.zeros(*first_decode_attention.shape[:3], input_ids.size(1), device=first_decode_attention.device, dtype=first_decode_attention.dtype)
    for decode_attention in output["decode_attentions"]:
        scores += decode_attention[:,:,:,:input_ids.size(1)]

    answer_indices = scores.topk(128, dim=3).indices.squeeze(2).tolist()
    
    return {
        "dataset": dataset_name,
        "input_prompt": prompt,
        "generation_length": len(output["decode_attentions"]),
        "answer_indices": answer_indices,
        "task_type": task_type,
    }

def main(args):
    set_seed(42)
    
    model_name = args.model
    model, tokenizer = load_model_for_generation(model_name)
    
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
    
    # ------------------------------------------------------------------
    # 1) Generate model outputs ONCE for all original samples (no duplication)
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print("Generating model outputs for all samples (no batch)...")
    print(f"{'='*50}")
    
    generated_all_samples: List[Dict[str, Any]] = []
    for sample in tqdm(all_samples, desc="Generating model outputs"):
        result = generate_answer(model, tokenizer, sample, model_name)
        generated_all_samples.append(result)
    
    print(f"\nTotal generated samples: {len(generated_all_samples)}")
    
    # ------------------------------------------------------------------
    # 2) Save balanced, generated training data (no additional generation)
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    total_samples = 0
    processed_counts = {}
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(generated_all_samples, desc="Saving training data"):
            dataset_name = sample.get("dataset", "unknown")
            prompt = sample["input_prompt"]
            generation_length = sample["generation_length"]
            answer_indices = sample["answer_indices"]
    
            try:
                training_sample = {
                    "dataset": dataset_name,
                    "input_prompt": prompt,
                    "generation_length": generation_length,
                    "answer_indices": answer_indices
                }
    
                f.write(json.dumps(training_sample, ensure_ascii=False, separators=(",", ":")) + "\n")
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
    parser.add_argument("--model", type=str, default="llama3", choices=["llama", "llama2", "llama3", "opt"], help="Model to use")
    parser.add_argument("--stats-only", action="store_true", help="Only check statistics of existing file without generating new data")
    
    args = parser.parse_args()
    
    if args.stats_only:
        # Only check statistics
        check_training_data_stats(args.output_file)
    else:
        # Generate training data (original main function)
        main(args)