import os
import torch
import json
import argparse

from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm

from utils import load_configs
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

def evaluate_model(
    model,
    tokenizer,
    prompts,
    answers,
    output_indices,
    device,
    desc="Generating",
    init_cache_fn=None,
    compression_config=None
):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    eos_token_id = tokenizer.eos_token_id

    predictions = []
    throughput_samples = []

    for idx, prompt in enumerate(tqdm(prompts, desc=desc)):
        input_ids = prompt.to(device)

        # Initialize cache if needed
        if init_cache_fn and compression_config:
            init_cache_fn(compression_config)

        # GPU timing events
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

        # Generate
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=output_indices[idx].numel(),
            eos_token_id=eos_token_id,
            do_sample=False
        )

        # Record timing
        end_evt.record()
        torch.cuda.synchronize()
        elapsed = start_evt.elapsed_time(end_evt) / 1000.0  # ms → s
        toks = gen_ids.shape[1] - input_ids.shape[1]
        throughput_samples.append(toks / elapsed if elapsed > 0 else 0)

        # Decode
        pred_text = tokenizer.decode(
            gen_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True
        )
        predictions.append(pred_text)

    # Calculate ROUGE scores
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for ref, pred in zip(answers, predictions):
        score = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(score[key].fmeasure)

    # Calculate averages
    avg_r1 = sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1'])
    avg_r2 = sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2'])
    avg_rL = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])
    avg_tp = sum(throughput_samples) / len(throughput_samples)

    return {
        'predictions': predictions,
        'rouge1': avg_r1,
        'rouge2': avg_r2,
        'rougeL': avg_rL,
        'throughput': avg_tp
    }

def main(args):
    # Initialize budget and hyperparameter lists
    model_name = args.model
    datasets = args.dataset
    budget_list = args.budget
    methods = args.method

    # Check and extend list lengths
    max_len = len(datasets) * len(budget_list) * len(methods)
    
    # Prepare device, model, and tokenizer
    device = f"cuda:{args.gpu}"
    
    # Load appropriate model based on model name
    if "llama" == model_name:
        model_name_hf = "meta-llama/Llama-2-7b-chat-hf"
        model = (KVLlamaForCausalLM.from_pretrained(model_name_hf).to(torch.bfloat16).to(device))
    elif "opt" == model_name:
        model_name_hf = "facebook/opt-6.7b"
        model = (KVOPTForCausalLM.from_pretrained(model_name_hf).to(torch.bfloat16).to(device))
    elif "llama_optimal" == model_name:
        model_name_hf = "meta-llama/Llama-2-7b-chat-hf"
        model = (OptimalLlamaForCausalLM.from_pretrained(model_name_hf).to(torch.bfloat16).to(device))
    else:
        raise ValueError(f"Unsupported model: {args.model}. Only Llama and OPT models are supported.")

    tokenizer = AutoTokenizer.from_pretrained(model_name_hf)

    cur_idx = 0
    for dataset in datasets:
        dataset_name = os.path.basename(dataset).split('.')[0]
        
        # Load dataset
        prompts, answers, output_indices = load_datasets(
            dataset_path=dataset,
            tokenizer=tokenizer
        )

        # Nested loops: budget → method
        for cur_budget in budget_list:
            for cur_method in methods:
                cur_idx += 1
                config = load_configs(model_name_hf.split("/")[1], cur_method, cur_budget)

                # Warm-up
                for p in prompts[:10]:
                    model.init_cache(config)
                    _ = model.generate(
                        p.to(device),
                        max_new_tokens=output_indices[0].numel(),
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False
                    )

                # Evaluate with current configuration
                results = evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    answers=answers,
                    output_indices=output_indices,
                    device=device,
                    desc=f"dataset={dataset_name}, method={cur_method}, budget={cur_budget}, cfg={cur_idx}/{max_len}",
                    init_cache_fn=model.init_cache,
                    compression_config=config
                )

                # Print results
                print(f"Config {cur_idx}/{max_len} | dataset={dataset_name}, method={cur_method}, budget={cur_budget}")
                print(
                    f"  ROUGE-1: {results['rouge1']:.4f}, ROUGE-2: {results['rouge2']:.4f}, ROUGE-L: {results['rougeL']:.4f}\n"
                    f"  Throughput: {results['throughput']:.2f} toks/s\n"
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Llama predictions with various budgets.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--dataset", type=str, nargs='+', default=["datasets/cnn_dailymail-3shot.jsonl"])
    parser.add_argument("--budget", type=int, nargs='+', default=[100])
    parser.add_argument("--method", type=str, nargs='+', default=["h2o"])
    args = parser.parse_args()
    main(args)