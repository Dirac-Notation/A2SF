import os
from datasets import load_dataset
import json
from tqdm import tqdm
import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

def extract_data(data, prompt_format, dataset_name, out_path):
    """Extract input prompts and generate dummy outputs"""
    for json_obj in tqdm(data, desc=f"Processing {dataset_name}"):
        # Format the prompt using the dataset's prompt format
        prompt = prompt_format.format(**json_obj)
        
        # Save to file
        result = {
            "input_prompt": prompt,
            "answers": json_obj.get("answers", []),
            "all_classes": json_obj.get("all_classes", []),
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
    
    # Define datasets (same as in original code)
    datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    
    # Load configurations
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    
    # Create output directory
    output_dir = "result_txt/longbench"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each dataset
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")

        # Load dataset
        data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
        
        # Set output path
        out_path = f"{output_dir}/{dataset}.jsonl"
        
        # Get prompt format and max length for this dataset
        prompt_format = dataset2prompt[dataset]
        
        print(f"Dataset size: {len(data)}")
        
        # Extract data
        extract_data(data, prompt_format, dataset, out_path)
        
if __name__ == '__main__':
    main()