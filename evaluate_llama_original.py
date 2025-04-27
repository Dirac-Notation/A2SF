import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import load_datasets, evaluate_model

def main(args):
    device = f"cuda:{args.gpu}"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name) \
               .to(torch.bfloat16).to(device)

    # Load prompts, answers, and output indices
    prompts, answers, output_indices = load_datasets(
        dataset_path=args.datasets,
        tokenizer=tokenizer
    )

    # Warm-up
    for p in prompts[:10]:
        _ = model.generate(
            p.to(device),
            max_new_tokens=output_indices[0].numel(),
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    # Evaluate model
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        answers=answers,
        output_indices=output_indices,
        device=device
    )

    # Print results
    dataset_name = os.path.splitext(os.path.basename(args.datasets))[0]
    print(
        f"Config 1/1 | model={args.model_name}, "
        f"dataset={dataset_name}"
    )
    print(
        f"  ROUGE-1: {results['rouge1']:.4f}, "
        f"ROUGE-2: {results['rouge2']:.4f}, "
        f"ROUGE-L: {results['rougeL']:.4f}"
    )
    print(f"  Throughput: {results['throughput']:.2f} toks/s\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate base LLM predictions using ROUGE and throughput."
    )
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--datasets",   type=str,
                        default="fewshot_data/cnn_dailymail-3shot.jsonl")
    args = parser.parse_args()
    main(args)
