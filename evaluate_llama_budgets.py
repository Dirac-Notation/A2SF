import os
import torch
import json
import argparse
from transformers import AutoTokenizer

from utils import load_datasets, evaluate_model
from utils_real_drop.kv_llama import LlamaForCausalLM
from utils_real_drop.kv_opt import OPTForCausalLM

def main(args):
    # Initialize budget and hyperparameter lists
    sb_list = args.select_budget
    rb_list = args.recent_budget
    randb_list = args.random_budget
    sbud_list = args.streaming_budget
    ff_list = args.forgetting_factor
    rm_list = args.random_method

    # Check and extend list lengths
    lengths = [len(sb_list), len(rb_list), len(randb_list), len(sbud_list)]
    max_len = max(lengths)
    cfg_len = max_len * len(ff_list) * len(rm_list)
    for l in lengths:
        if l != 1 and l != max_len:
            raise ValueError("All budget lists (including streaming) must have the same length or contain only one element.")
    if len(sb_list) == 1: sb_list *= max_len
    if len(rb_list) == 1: rb_list *= max_len
    if len(randb_list) == 1: randb_list *= max_len
    if len(sbud_list) == 1: sbud_list *= max_len

    # Prepare device, model, and tokenizer
    device = f"cuda:{args.gpu}"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load appropriate model based on model name
    if "llama" in args.model_name.lower():
        model = (LlamaForCausalLM.from_pretrained(args.model_name).to(torch.bfloat16).to(device))
    elif "opt" in args.model_name.lower():
        model = (OPTForCausalLM.from_pretrained(args.model_name).to(torch.bfloat16).to(device))
    else:
        raise ValueError(f"Unsupported model: {args.model_name}. Only Llama and OPT models are supported.")

    # Load dataset
    prompts, answers, output_indices = load_datasets(
        dataset_path=args.datasets,
        tokenizer=tokenizer
    )

    # Prepare output directory
    dataset_name = os.path.splitext(os.path.basename(args.datasets))[0]
    output_dir = "output_text"
    os.makedirs(output_dir, exist_ok=True)

    # Nested loops: forgetting_factor → random_method → budget configs
    for ff in ff_list:
        for rm in rm_list:
            for idx in range(max_len):
                cur_sb = sb_list[idx]
                cur_rb = rb_list[idx]
                cur_rand = randb_list[idx]
                cur_sbud = sbud_list[idx]

                # Warm-up
                for p in prompts[:10]:
                    model.init_cache(
                        use_compression=False,
                        select_budget=cur_sb,
                        recent_budget=cur_rb,
                        random_budget=cur_rand,
                        streaming_budget=cur_sbud,
                        random_method=rm,
                        forgetting_factor=ff
                    )
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
                    desc=f"sel={cur_sb},rec={cur_rb},ran={cur_rand},str={cur_sbud},ff={ff},rm={rm},cfg{idx+1}/{cfg_len}",
                    init_cache_fn=model.init_cache,
                    cache_params={
                        'use_compression': True,
                        'select_budget': cur_sb,
                        'recent_budget': cur_rb,
                        'random_budget': cur_rand,
                        'streaming_budget': cur_sbud,
                        'random_method': rm,
                        'forgetting_factor': ff
                    }
                )

                # Print results
                print(
                    f"Config {idx+1}/{max_len} | "
                    f"select={cur_sb}, recent={cur_rb}, random={cur_rand}, "
                    f"streaming={cur_sbud}, ff={ff}, rm={rm}"
                )
                print(
                    f"  ROUGE-1: {results['rouge1']:.4f}, "
                    f"ROUGE-2: {results['rouge2']:.4f}, "
                    f"ROUGE-L: {results['rougeL']:.4f}"
                )
                print(f"  Throughput: {results['throughput']:.2f} toks/s\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Llama predictions with various budgets.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--datasets", type=str, default="fewshot_data/cnn_dailymail-3shot.jsonl")
    parser.add_argument("--select_budget", type=int, nargs='+', default=[100])
    parser.add_argument("--recent_budget", type=int, nargs='+', default=[100])
    parser.add_argument("--random_budget", type=int, nargs='+', default=[0])
    parser.add_argument("--streaming_budget", type=int, nargs='+', default=[0])
    parser.add_argument("--forgetting_factor", type=float, nargs='+', default=[1.0])
    parser.add_argument("--random_method", type=str, nargs='+', default=["att"])
    args = parser.parse_args()
    main(args)