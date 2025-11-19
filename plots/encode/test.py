import json
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from collections import defaultdict
import os
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append("/home/smp9898/A2SF")
from RL.features import ContextEncoder

class LongBenchAnalyzer:
    """Analyze LongBench datasets using upgraded ContextEncoder"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        self.encoder = ContextEncoder(model_name=model_name, device=device, context_window=64, max_context=128)
        self.dataset_path = Path("datasets/longbench")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.max_length = 7500
        
    def load_dataset(self, dataset_name: str) -> List[Dict]:
        """Load a specific dataset from longbench"""
        file_path = self.dataset_path / f"{dataset_name}.jsonl"
        data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tmp = json.loads(line.strip())
                
                prompt = tmp["input_prompt"]
                tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
                
                if len(tokenized_prompt) > self.max_length:
                    half = int(self.max_length / 2)
                    prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                            self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                
                if tmp["dataset"] not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                    prompt = f"[INST]{prompt}[/INST]"
                
                tmp["input_prompt"] = prompt
                        
                data.append(tmp)
        
        return data
    
    def encode_dataset(self, dataset_name: str) -> Dict:
        """Encode a dataset and return similarity statistics"""
        print(f"Processing dataset: {dataset_name}")
        
        data = self.load_dataset(dataset_name)
        
        similarities = []
        
        for i, item in enumerate(tqdm(data)):
            # Encode the input prompt using ContextEncoder
            similarity_vector = self.encoder.encode_context(item["input_prompt"])
            
            # Convert to numpy for analysis
            similarity_np = similarity_vector.cpu()
            similarities.append(similarity_np)
        
        return torch.stack(similarities)
    
    def process_all_datasets(self) -> Dict:
        """Process all datasets in the longbench directory"""
        dataset_files = list(self.dataset_path.glob("*.jsonl"))
        dataset_names = [f.stem for f in dataset_files]
        
        print(f"Found {len(dataset_names)} datasets: {dataset_names}")
        
        all_results = {}
        
        for dataset_name in dataset_names:
            result = self.encode_dataset(dataset_name)
            all_results[dataset_name] = result.mean(dim=0)
            print(f"Completed {dataset_name}")
            
        return all_results
    
    def create_analysis_plots(self, save_path: str = "longbench_analysis", all_results: Dict = None):
        data_group = {
            "Code Complete": ["repobench-p", "lcc"],
            "Few Shot": ["trec", "triviaqa", "samsum"],
            "Single-doc QA": ["narrativeqa", "qasper", "multifieldqa_en"],
            "Multi-doc QA": ["hotpotqa", "2wikimqa", "musique"],
            "Summarization": ["gov_report", "qmsum", "multi_news"],
            "Passage Retrieval": ["passage_retrieval_en", "passage_count"],
        }
        
        os.makedirs(save_path, exist_ok=True)

        # Set larger font sizes to match generate_accuracy_plots.py style
        plt.rcParams.update({
            'font.size': 20,
            'axes.titlesize': 20,
            'axes.labelsize': 20,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 15,
            'figure.titlesize': 20
        })

        for task, datasets in data_group.items():
            # Create figure with proper sizing
            plt.figure(figsize=(12, 6))
            
            # Get colors for each dataset
            colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
            
            for i, dataset in enumerate(datasets):
                if dataset in all_results:
                    result = all_results[dataset]
                    # Convert tensor to numpy if needed
                    if hasattr(result, 'cpu'):
                        result = result.cpu().numpy()
                    elif hasattr(result, 'numpy'):
                        result = result.numpy()
                    
                    # Plot with markers and lines
                    plt.plot(result, 'o-', label=dataset, color=colors[i], 
                            linewidth=2, markersize=6)
                    
                    # Find and highlight the maximum value
                    max_idx = np.argmax(result)
                    max_y = result[max_idx]
                    
                    # Highlight the maximum point with a larger, different marker
                    plt.scatter(max_idx, max_y, s=150, color=colors[i], 
                               marker='*', edgecolors='black', linewidth=1, zorder=5)
            
            # Add labels and formatting
            plt.xlabel('Context Position')
            plt.ylabel('Similarity Score')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            
            # Adjust layout
            plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
            
            # Save with high DPI
            plt.savefig(f"{save_path}/{task}.png", dpi=300, bbox_inches='tight')
            print(f"Saved plot: {save_path}/{task}.png")
            plt.close()
        
def main():
    """Main function to run the analysis"""
    print("Starting LongBench Analysis with Upgraded ContextEncoder")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = LongBenchAnalyzer()  # Use "cuda" if GPU available
    
    # Process all datasets (limit samples for faster processing)
    print("Processing datasets...")
    results = analyzer.process_all_datasets()

    analyzer.create_analysis_plots("plots/encode", results)
    
    print("\nAnalysis complete! Check the 'longbench_analysis_results' directory for results.")

if __name__ == "__main__":
    main()
