import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm

from utils import CompressionConfig
from utils_real_drop import KVLlamaForCausalLM, KVOPTForCausalLM, OptimalLlamaForCausalLM

def load_datasets(
    dataset_path: str,
    tokenizer
):
    with open(dataset_path, "r") as f:
        datalines = f.readlines()
        articles = []
        
        for dataline in datalines:
            articles.append(json.loads(dataline))

    prompts = []
    answers = []
    output_indices = []

    for data in tqdm(articles, desc="Tokenizing"):
        input = data["article"]
        answer = data["summary_gt"]
        
        input = f"[INST]{input}[/INST]"
        
        input_ids = tokenizer(input, return_tensors="pt").input_ids
        output_ids = tokenizer(answer, return_tensors="pt").input_ids

        prompts.append(input_ids)
        answers.append(answer)
        output_indices.append(output_ids)
    
    return prompts, answers, output_indices

def main(args):
    # Initialize budget and hyperparameter lists
    model_name_hf = "meta-llama/Llama-2-7b-chat-hf"
    dataset_name = "datasets/cnn_dailymail-0shot_Llama-2-7b-chat-hf.jsonl"
    
    # Prepare device, model, and tokenizer
    device = f"cuda:{args.gpu}"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
    model = (OptimalLlamaForCausalLM.from_pretrained(model_name_hf).to(torch.bfloat16).to(device))
    
    # Load dataset
    prompts, answers, output_indices = load_datasets(
        dataset_path=dataset_name,
        tokenizer=tokenizer
    )
    # Create indices to track original order
    indices = list(range(len(prompts)))
    
    # Sort indices based on prompt lengths
    indices.sort(key=lambda i: prompts[i].numel())
    
    # Reorder all lists according to sorted indices
    prompts = [prompts[i] for i in indices]
    answers = [answers[i] for i in indices] 
    output_indices = [output_indices[i] for i in indices]
    prompts.sort(key=lambda x: x.numel())
    
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    eos_token_id = tokenizer.eos_token_id

    rouge_scores = []
    sim_scores = []
    budgets = []

    num_prompts = 5

    print("Budget\tAverage Similarity\tROUGE-1 Score")
    print("-" * 50)

    for i in range(25,525,5):
        rouge_score = 0
        sim_score = 0
        for idx in range(1, num_prompts+1):
            input_ids = prompts[-idx].to(device)

            # Initialize cache if needed
            model.init_cache(CompressionConfig(use_compression=True, total_budget=i))

            # Generate
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=output_indices[-idx].numel(),
                eos_token_id=eos_token_id,
                do_sample=False
            )

            # Decode
            pred_text = tokenizer.decode(gen_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
            
            rouge_score += scorer.score(answers[-idx], pred_text)['rouge1'].fmeasure
            sim_score += torch.tensor([model.model.layers[i].self_attn.past_key_value.sim_scores for i in range(len(model.model.layers))])
        
        avg_rouge = rouge_score/num_prompts
        avg_sim = torch.mean(sim_score/num_prompts).item()
        
        budgets.append(i)
        rouge_scores.append(avg_rouge)
        sim_scores.append(avg_sim)
        
        print(f"{i}\t{avg_sim:.4f}\t\t{avg_rouge:.4f}")

    # Create the plot
    plt.figure(figsize=(9, 6))
    
    # Plot the data points
    plt.scatter(sim_scores, rouge_scores, color='blue', alpha=0.5, label='ROUGE-1 Scores')
    
    # Calculate and plot trend line
    z = np.polyfit(sim_scores, rouge_scores, 3)
    p = np.poly1d(z)
    plt.plot(sim_scores, p(sim_scores), color='red', linewidth=2, label='Trend Line')
    
    plt.title('ROUGE-1 Scores vs Similarity', fontsize=24)
    plt.xlabel('Similarity Score', fontsize=22)
    plt.ylabel('ROUGE-1 Score', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=20, loc='upper left')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    # Create directory if it doesn't exist
    os.makedirs('plots/sim-acc', exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('plots/sim-acc/rouge_vs_sim.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Llama predictions with various budgets.")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)