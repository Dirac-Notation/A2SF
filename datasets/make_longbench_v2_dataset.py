import os
from datasets import load_dataset
import json
from tqdm import tqdm
import random
import numpy as np
import shutil

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

def extract_data(data, dataset_name, out_path):
    """Extract input prompts from LongBench v2 dataset"""
    for json_obj in tqdm(data, desc=f"Processing {dataset_name}"):
        # LongBench v2 structure may differ from v1
        # Extract relevant fields
        result = {
            "input_prompt": json_obj.get("context", "") + "\n\n" + json_obj.get("question", ""),
            "question": json_obj.get("question", ""),
            "context": json_obj.get("context", ""),
            "answer": json_obj.get("answer", ""),
            "choices": json_obj.get("choices", {}),
            "domain": json_obj.get("domain", ""),
            "sub_domain": json_obj.get("sub_domain", ""),
            "difficulty": json_obj.get("difficulty", ""),
            "length": json_obj.get("length", 0),
            "dataset": dataset_name
        }
        
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
        
    print(f"Results saved to: {out_path}")

def main():
    # Set random seed for reproducibility
    seed_everything(42)
    
    # LongBench v2 is a single unified dataset, not split by sub-datasets
    # Load the entire dataset
    print("Loading LongBench v2 dataset...")
    data = load_dataset('THUDM/LongBench-v2', split='train', trust_remote_code=True)
    
    # Create output directory
    output_dir = "datasets/longbench_v2"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    
    print(f"Dataset size: {len(data)}")
    
    # Group by domain if needed
    print("\nGrouping by domain...")
    domain_data = {}
    for item in data:
        domain = item.get("domain", "unknown")
        if domain not in domain_data:
            domain_data[domain] = []
        domain_data[domain].append(item)
    
    print(f"Found {len(domain_data)} domains: {list(domain_data.keys())}")
    
    # Save separate files for each domain
    for domain, items in domain_data.items():
        domain_out_path = f"{output_dir}/longbench_v2_{domain}.jsonl"
        extract_data(items, f"longbench_v2_{domain}", domain_out_path)
        print(f"Saved {len(items)} samples for domain: {domain}")
    
    print(f"\nAll results saved to: {output_dir}")

if __name__ == '__main__':
    main()


