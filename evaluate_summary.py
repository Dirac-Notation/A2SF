import os
import torch
import json
import argparse

from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm

from utils import load_configs
from utils_real_drop import KVLlamaForCausalLM, KVOPTForCausalLM, OptimalLlamaForCausalLM, MaskedLlamaForCausalLM

def load_datasets(
    dataset_path: str,
    tokenizer
):
    with open(dataset_path, "r") as f:
        datalines = f.readlines()
        articles = []
        
        for dataline in datalines:
            articles.append(json.loads(dataline))

    inputs = []
    answers = []
    output_indices = []

    for data in tqdm(articles, desc="Tokenizing"):
        input = data["article"]
        answer = data["summary_gt"]
        
        if "llama" in args.model.lower():
            input = f"[INST]{input}[/INST]"
        
        input_data = tokenizer(input, return_tensors="pt")
        output_ids = tokenizer(answer, return_tensors="pt").input_ids

        inputs.append(input_data)
        answers.append(answer)
        output_indices.append(output_ids)
    
    num_input_ids = sum([input_data.input_ids.numel() for input_data in inputs])/len(inputs)
    num_output_ids = sum([output_ids.numel() for output_ids in output_indices])/len(output_indices)
    
    print(f"Average input ids length : {num_input_ids:.2f}")
    print(f"Average output ids length : {num_output_ids:.2f}")
    
    return inputs, answers, output_indices

def evaluate_model(
    model,
    tokenizer,
    inputs,
    answers,
    output_indices,
    device,
    dataset_name,
    model_name,
    budget,
    method,
    desc="Generating",
    init_cache_fn=None,
    compression_config=None
):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.eos_token_id
    predictions = []
    throughput_samples = []

    # Create directory for saving results
    os.makedirs("result_txt/summary", exist_ok=True)
    result_file = f"result_txt/summary/{dataset_name}_{model_name}_{budget}_{method}.jsonl"

    # Open file in write mode to overwrite any existing content
    with open(result_file, 'w', encoding='utf-8') as f:
        for idx, input_data in enumerate(tqdm(inputs, desc=desc)):
            # Convert input data to proper format
            input_ids = input_data.input_ids.to(device)
            attention_mask = input_data.attention_mask.to(device)

            # Initialize cache if needed
            if init_cache_fn and compression_config:
                init_cache_fn(compression_config)

            # GPU timing events
            torch.cuda.synchronize()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()

            # Generate with proper input format and explicit pad_token_id
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=output_indices[idx].numel(),
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
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

            # Save to JSONL file
            result_entry = {
                "generated_summary": pred_text,
                "reference_summary": answers[idx],
                "input_length": input_ids.shape[1],
                "output_length": output_indices[idx].numel()
            }
            f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

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
    model2path = json.load(open("config/model2path.json", "r"))
    model_path = model2path[args.model]

    if "masked" in args.model.lower():
        model = (MaskedLlamaForCausalLM.from_pretrained(model_path).to(torch.float16).to(device))
    elif "llama" in args.model.lower():
        model = (KVLlamaForCausalLM.from_pretrained(model_path).to(torch.float16).to(device))
    elif "opt" in args.model.lower():
        model = (KVOPTForCausalLM.from_pretrained(model_path).to(torch.float16).to(device))
    else:
        raise ValueError(f"Unsupported model: {args.model}. Only Llama and OPT models are supported.")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    cur_idx = 0
    for dataset in datasets:
        dataset_name = os.path.basename(dataset).split('.')[0]
        
        # Load dataset
        inputs, answers, output_indices = load_datasets(
            dataset_path=dataset,
            tokenizer=tokenizer
        )

        # Nested loops: budget → method
        for cur_budget in budget_list:
            for cur_method in methods:
                cur_idx += 1
                config = load_configs(args.model, cur_method, cur_budget, tokenizer)

                # Evaluate with current configuration
                results = evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=inputs,
                    answers=answers,
                    output_indices=output_indices,
                    device=device,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    budget=cur_budget,
                    method=cur_method,
                    desc=f"{model_name} - dataset={dataset_name}, method={cur_method}, budget={cur_budget}, cfg={cur_idx}/{max_len}",
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
    parser.add_argument("--model", type=str, default="llama2")
    parser.add_argument("--dataset", type=str, nargs='+', default=["datasets/cnn_dailymail-3shot.jsonl"])
    parser.add_argument("--budget", type=int, nargs='+', default=[100])
    parser.add_argument("--method", type=str, nargs='+', default=["h2o"])
    args = parser.parse_args()
    main(args)