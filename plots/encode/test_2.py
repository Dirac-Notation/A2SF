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

class DatasetExampleAnalyzer:
    """Analyze individual examples within each dataset using ContextEncoder"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        self.encoder = ContextEncoder(model_name=model_name, device=device, context_window=64, max_context=128)
        self.dataset_path = Path("datasets/longbench")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.max_length = 7500
        self.num_examples = 4  # Number of examples to plot per dataset (2x2 subplot)
        
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
    
    def encode_examples(self, dataset_name: str) -> List[torch.Tensor]:
        """Encode examples from a dataset and return similarity vectors"""
        print(f"Processing dataset: {dataset_name}")
        
        data = self.load_dataset(dataset_name)
        
        # Select up to num_examples examples
        num_samples = min(self.num_examples, len(data))
        selected_data = data[:num_samples]
        
        similarities = []
        
        for i, item in enumerate(tqdm(selected_data, desc=f"Encoding {dataset_name}")):
            # Encode the input prompt using ContextEncoder
            similarity_vector = self.encoder.encode_context(item["input_prompt"])
            
            # Convert to numpy for analysis
            similarity_np = similarity_vector.cpu()
            similarities.append(similarity_np)
        
        return similarities
    
    def create_dataset_plot(self, dataset_name: str, similarities: List[torch.Tensor], save_path: str):
        """Create a 2x2 subplot for a single dataset showing 4 examples"""
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
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 9))
        axes = axes.flatten()  # Flatten to 1D array for easier indexing
        
        # Get colors for each example
        num_examples = len(similarities)
        colors = plt.cm.tab10(np.linspace(0, 1, num_examples))
        
        for i, similarity in enumerate(similarities):
            # Convert tensor to numpy if needed
            if hasattr(similarity, 'cpu'):
                result = similarity.cpu().numpy()
            elif hasattr(similarity, 'numpy'):
                result = similarity.numpy()
            else:
                result = np.array(similarity)
            
            # Plot on the i-th subplot
            ax = axes[i]
            ax.plot(result, 'o-', color=colors[i], linewidth=2, markersize=6)
            ax.set_title(f'Prompt_{i+1}')
            ax.set_xlabel('Context Position')
            ax.set_ylabel('Similarity Score')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots if we have fewer than 4 examples
        for i in range(num_examples, 4):
            axes[i].axis('off')
        
        # Adjust layout
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        
        # Save with high DPI
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
        plt.close()
    
    def process_all_datasets(self):
        """Process all datasets and create individual plots"""
        dataset_files = list(self.dataset_path.glob("*.jsonl"))
        dataset_names = [f.stem for f in dataset_files]
        
        print(f"Found {len(dataset_names)} datasets: {dataset_names}")
        
        # Create output directory
        output_dir = Path("plots/encode/datasets")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset_name in dataset_names:
            try:
                # Encode examples
                similarities = self.encode_examples(dataset_name)
                
                # Create plot
                save_path = output_dir / f"{dataset_name}.png"
                self.create_dataset_plot(dataset_name, similarities, str(save_path))
                
                print(f"Completed {dataset_name}")
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                continue

def main():
    """Main function to run the analysis"""
    print("Starting Dataset Example Analysis with ContextEncoder")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = DatasetExampleAnalyzer()  # Use "cuda" if GPU available
    
    # Process all datasets
    print("Processing datasets...")
    analyzer.process_all_datasets()
    
    print("\nAnalysis complete! Check the 'plots/encode/datasets' directory for results.")

if __name__ == "__main__":
    main()

