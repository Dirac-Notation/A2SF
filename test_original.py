import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from tqdm import tqdm

from utils import load_datasets

def main(args):
    device = f"cuda:{args.gpu}"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(torch.bfloat16).to(device)
    prompts, answers, output_indices = load_datasets(dataset_path=args.datasets, tokenizer=tokenizer)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    eos_token_id = tokenizer.eos_token_id
    predictions = []

    for idx, prompt in enumerate(tqdm(prompts)):
        input_ids = prompt.to(device)
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=output_indices[idx].numel() + 10,
            eos_token_id=eos_token_id,
            do_sample=False
        )
        predictions.append(tokenizer.decode(gen_ids[0, input_ids.numel():], skip_special_tokens=True))
    
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for ref, pred in zip(answers, predictions):
        score = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(score[key].fmeasure)

    avg_rouge1 = sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1'])
    avg_rouge2 = sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2'])
    avg_rougeL = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])
    
    print(f"Average ROUGE-1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-2: {avg_rouge2:.4f}")
    print(f"Average ROUGE-L: {avg_rougeL:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Llama model predictions using ROUGE scores.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--datasets", type=str, default="fewshot_data/cnn_dailymail-3shot.jsonl")
    args = parser.parse_args()
    main(args)
