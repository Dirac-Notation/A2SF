import os
import torch
import json
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm

from utils import load_configs, CompressionConfig
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
        
        input_ids = tokenizer(input, return_tensors="pt").input_ids
        output_ids = tokenizer(answer, return_tensors="pt").input_ids

        prompts.append(input_ids)
        answers.append(answer)
        output_indices.append(output_ids)
    
    num_input_ids = sum([prompt.numel() for prompt in prompts])/len(prompts)
    num_output_ids = sum([output_ids.numel() for output_ids in output_indices])/len(output_indices)
    
    print(f"Average input ids length : {num_input_ids:.2f}")
    print(f"Average output ids length : {num_output_ids:.2f}")
    
    return prompts, answers, output_indices

def main(args):
    # Initialize budget and hyperparameter lists
    model_name_hf = "meta-llama/Llama-2-7b-chat-hf"
    dataset_name = "datasets/cnn_dailymail-3shot.jsonl"
    
    # Prepare device, model, and tokenizer
    device = f"cuda:{args.gpu}"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
    model = (OptimalLlamaForCausalLM.from_pretrained(model_name_hf).to(torch.bfloat16).to(device))
    
    # Load dataset
    prompts, answers, output_indices = load_datasets(
        dataset_path=dataset_name,
        tokenizer=tokenizer
    )

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    eos_token_id = tokenizer.eos_token_id

    rouge_scores = []
    sim_scores = []

    for _ in tqdm(range(100), desc="Predicting"):
        total_budget = random.randint(10, 100)
        
        rouge_score = 0
        sim_score = 0
        
        for idx, input_ids in enumerate(prompts[:10]):
            input_ids = input_ids.to(device)

            # Initialize cache if needed
            model.init_cache(CompressionConfig(use_compression=True, total_budget=total_budget))

            # Generate
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=output_indices[idx].numel(),
                eos_token_id=eos_token_id,
                do_sample=False
            )

            # Decode
            pred_text = tokenizer.decode(gen_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
            
            rouge_score += scorer.score(answers[idx], pred_text)['rouge1'].fmeasure
            sim_score += torch.tensor([model.model.layers[i].self_attn.past_key_value.sim_scores for i in range(len(model.model.layers))]).mean()
        
        rouge_scores.append(rouge_score / 10)
        sim_scores.append(sim_score / 10)

    # Create scatter plot with trend line
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(sim_scores, rouge_scores, color='blue', alpha=0.6, label='Data points')
    
    # Calculate trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(sim_scores, rouge_scores)
    line = slope * np.array(sim_scores) + intercept
    
    # Plot trend line
    plt.plot(sim_scores, line, color='red', label=f'Trend line (r={r_value:.2f})')
    
    # Customize plot
    plt.xlabel('Similarity of Intermediate', fontsize=12)
    plt.ylabel('Rouge-1 Score', fontsize=12)
    plt.title('Relationship between Similarity and Rouge-1 Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot
    plt.savefig('plots/similarity_rouge_relationship.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Llama predictions with various budgets.")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)