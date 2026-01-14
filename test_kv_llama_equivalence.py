"""
Test script to verify that KVLlamaForCausalLM with full_cache produces identical results
to standard AutoModelForCausalLM using LongBench test datasets.
"""
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils_real_drop import KVLlamaForCausalLM
from utils import CompressionConfig, set_seed

def load_jsonl_file(file_path):
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_models(model_name, gpu_list=None):
    """Load both KVLlamaForCausalLM and AutoModelForCausalLM models."""
    model2path = json.load(open("config/model2path.json", "r"))
    model_path = model2path[model_name]
    
    if gpu_list is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load KVLlamaForCausalLM with full_cache
    print(f"Loading KVLlamaForCausalLM from {model_path}...")
    kv_model = KVLlamaForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    kv_model = kv_model.eval()
    
    # Initialize full cache
    compression_config = CompressionConfig()
    compression_config.compression_method = "full"
    compression_config.total_budget = 10000  # Large budget for full cache
    compression_config.layerwise_ratios = [1.0] * kv_model.config.num_hidden_layers
    compression_config.local_ratios = 0.125
    kv_model.init_cache(compression_config)
    print("KVLlamaForCausalLM with full_cache initialized.")
    
    # Load standard AutoModelForCausalLM
    print(f"Loading AutoModelForCausalLM from {model_path}...")
    standard_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    standard_model = standard_model.eval()
    print("AutoModelForCausalLM loaded.")
    
    return kv_model, standard_model, tokenizer

def generate_text(model, tokenizer, prompt, max_length, max_new_tokens, dataset, model_name):
    """Generate text using the model with LongBench formatting."""
    # Truncate if too long
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized_prompt) > max_length:
        half = int(max_length/2)
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    
    # Format prompt if needed
    if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        if "llama" in model_name:
            prompt = f"[INST]{prompt}[/INST]"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    context_length = input_ids.shape[-1]
    
    # Generate
    with torch.inference_mode():
        if dataset == "samsum":
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                pad_token_id=tokenizer.eos_token_id,
            )[0]
        else:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
    
    # Decode only the generated part
    generated_text = tokenizer.decode(outputs[context_length:], skip_special_tokens=True)
    
    return generated_text

def test_equivalence(model_name="llama2", gpu_list=[0], dataset=None, num_samples=None):
    """
    Test that KVLlamaForCausalLM with full_cache produces identical results to AutoModelForCausalLM
    using LongBench test datasets.
    
    Args:
        model_name: Name of the model to test
        gpu_list: List of GPU IDs
        dataset: Specific dataset to test. If None, tests all datasets.
        num_samples: Number of samples to test per dataset. If None, tests all samples.
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Load models
    kv_model, standard_model, tokenizer = load_models(model_name, gpu_list)
    
    # Load configurations
    model_name_clean = model_name.split("_")[0].lower()
    max_length = json.load(open("config/model2maxlen.json", "r"))[model_name_clean]
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    # Available datasets
    datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
                "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
                "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    
    if dataset is not None:
        datasets = [dataset] if dataset in datasets else []
    
    print("\n" + "="*80)
    print("Testing equivalence between KVLlamaForCausalLM (full_cache) and AutoModelForCausalLM")
    print(f"Using LongBench datasets: {', '.join(datasets)}")
    print("="*80 + "\n")
    
    total_tests = 0
    total_matches = 0
    all_match = True
    
    for dataset_name in datasets:
        jsonl_path = f"datasets/longbench/{dataset_name}.jsonl"
        if not os.path.exists(jsonl_path):
            print(f"⚠️  Warning: {jsonl_path} not found, skipping {dataset_name}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'='*80}\n")
        
        # Load dataset
        data = load_jsonl_file(jsonl_path)
        if num_samples is not None and len(data) > num_samples:
            import random
            data = random.sample(data, num_samples)
        
        # max_gen = dataset2maxlen[dataset_name]
        max_gen = 1
        
        dataset_matches = 0
        dataset_tests = 0
        
        for i, json_obj in enumerate(tqdm(data, desc=f"Testing {dataset_name}")):
            prompt = json_obj["input_prompt"]
            
            # Generate with KVLlamaForCausalLM
            kv_text = generate_text(kv_model, tokenizer, prompt, max_length, max_gen, dataset_name, model_name_clean)
            
            # Generate with AutoModelForCausalLM
            standard_text = generate_text(standard_model, tokenizer, prompt, max_length, max_gen, dataset_name, model_name_clean)
            
            # Compare
            match = (kv_text == standard_text)
            dataset_matches += match
            dataset_tests += 1
            total_matches += match
            total_tests += 1
            
            if not match:
                all_match = False
                print(f"\n❌ Mismatch in sample {i+1}/{len(data)}")
                print(f"Prompt: {prompt[:100]}...")
                print(f"\nKVLlamaForCausalLM output:\n{kv_text[:200]}...")
                print(f"\nAutoModelForCausalLM output:\n{standard_text[:200]}...")
                print(f"\nFirst difference:")
                min_len = min(len(kv_text), len(standard_text))
                for j in range(min_len):
                    if kv_text[j] != standard_text[j]:
                        print(f"  Position {j}: '{kv_text[j]}' vs '{standard_text[j]}'")
                        print(f"  KV context: ...{kv_text[max(0,j-20):j+20]}...")
                        print(f"  Standard context: ...{standard_text[max(0,j-20):j+20]}...")
                        break
                if len(kv_text) != len(standard_text):
                    print(f"  Length difference: {len(kv_text)} vs {len(standard_text)}")
                print()
        
        print(f"\n{dataset_name}: {dataset_matches}/{dataset_tests} matches ({100*dataset_matches/dataset_tests:.1f}%)")
    
    # Final result
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Total: {total_matches}/{total_tests} matches ({100*total_matches/total_tests:.1f}%)")
    if all_match:
        print("✅ ALL TESTS PASSED: KVLlamaForCausalLM (full_cache) produces identical results!")
    else:
        print("❌ TESTS FAILED: Some results differ between models.")
    print("="*80)
    
    return all_match

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test equivalence between KVLlamaForCausalLM and AutoModelForCausalLM using LongBench")
    parser.add_argument("--model", type=str, default="llama3", choices=["llama", "llama2", "llama3"], 
                       help="Model name to test")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0], help="GPU IDs to use")
    parser.add_argument("--dataset", type=str, default=None, 
                       help="Specific dataset to test (optional, tests all if not specified)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to test per dataset (optional, tests all if not specified)")
    
    args = parser.parse_args()
    
    test_equivalence(
        model_name=args.model,
        gpu_list=args.gpus,
        dataset=args.dataset,
        num_samples=args.num_samples
    )

