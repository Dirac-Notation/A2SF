import argparse

parser = argparse.ArgumentParser(description="Process dataset for few-shot learning.")
parser.add_argument('--dataset', type=str, required=True, help="abisee/cnn_dailymail, EdinburghNLP/xsum, alexfabbri/multi_news, Samsung/samsum")
parser.add_argument('--shots', type=int, nargs='+', required=True, help="Number of examples for few-shot learning.")
parser.add_argument('--use_model', action='store_true', default=False, help="Whether to use model for generation")
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Name of the model to use for generation")
parser.add_argument('--gpu', type=int, default=0, help="GPU device ID")
args = parser.parse_args()
shots = args.shots

import os
import json
from datasets import load_dataset
from tqdm import tqdm

if args.use_model:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(torch.float16).to(device)

    def generate_summary(prompt):
        if "llama" in args.model_name.lower():
            prompt = f"[INST]{prompt}[/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        return response

SYSTEM_PROMPT = 'You are a helpful assistant that can summarize the text.\n\n'
USER_PROMPT_TEMPLATE = '<text>\n{article}\n</text>\n\nSummarize the above article briefly.'

dataset_obj = (load_dataset(args.dataset, "2.0.0") if args.dataset == "abisee/cnn_dailymail" else load_dataset(args.dataset))
dataset = dataset_obj['test']

prompts = []
for entry in dataset:
    if args.dataset in ["abisee/cnn_dailymail", "EdinburghNLP/xsum", "alexfabbri/multi_news"]:
        article, highlight = list(entry.values())[:2]
    elif args.dataset in ["Samsung/samsum"]:
        article, highlight = list(entry.values())[1:3]
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    prompts.append({
        'article': article,
        'summary': highlight
    })

if args.dataset in ["abisee/cnn_dailymail", "EdinburghNLP/xsum"]:
    prompts = sorted(prompts, key=lambda x: len(x['article']))[1000:]
elif args.dataset in ["Samsung/samsum"]:
    prompts = sorted(prompts, key=lambda x: len(x['article']))[500:]
elif args.dataset in ["alexfabbri/multi_news"]:
    prompts = sorted(prompts, key=lambda x: len(x['article']))[200:]

dataset_name = args.dataset.split('/')[-1]
os.makedirs('datasets', exist_ok=True)

for shot in shots:
    model_suffix = f"_{args.model_name.split('/')[-1]}" if args.use_model else ""
    output_path = f'datasets/{dataset_name}-{shot}shot{model_suffix}.jsonl'
    with open(output_path, 'w', encoding='utf-8') as fout:
        target_indices = range(shot, shot + 100 * 2, 2)
        for target_idx in tqdm(target_indices, desc=f"Processing shot={shot}"):
            if target_idx >= len(prompts):
                break
            target = prompts[target_idx]
            
            fewshot_examples = ''
            for example in prompts[:shot]:
                prompt_text = USER_PROMPT_TEMPLATE.format(article=example['article'])
                fewshot_examples += f"{prompt_text}{example['summary']}\n\n"

            final_prompt = (f"{SYSTEM_PROMPT}{fewshot_examples}{USER_PROMPT_TEMPLATE.format(article=target['article'])}")

            if args.use_model:
                summary = generate_summary(final_prompt)
            else:
                summary = target['summary']

            record = {
                'article': final_prompt,
                'summary_gt': summary
            }
            json.dump(record, fout, ensure_ascii=False)
            fout.write('\n')