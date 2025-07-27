import argparse
import random
import os
import json

from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Process dataset for few-shot learning.")
parser.add_argument('--dataset', type=str, required=True, help="abisee/cnn_dailymail, EdinburghNLP/xsum, alexfabbri/multi_news, Samsung/samsum")
parser.add_argument('--shots', type=int, nargs='+', required=True, help="Number of examples for few-shot learning.")

args = parser.parse_args()
shots = args.shots

# Set random seed for reproducibility
random.seed(42)

SYSTEM_PROMPT = 'You are a helpful assistant that can summarize the text.\n\n'
USER_PROMPT_TEMPLATE = '<text>\n{article}\n</text>\n\nSummarize the key points from the text above. Provide a direct response.'

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

# Randomly sample prompts instead of sorting by length
if args.dataset in ["abisee/cnn_dailymail", "EdinburghNLP/xsum"]:
    prompts = random.sample(prompts, len(prompts) - 1000)
elif args.dataset in ["Samsung/samsum"]:
    prompts = random.sample(prompts, len(prompts) - 500)
elif args.dataset in ["alexfabbri/multi_news"]:
    prompts = random.sample(prompts, len(prompts) - 200)

dataset_name = args.dataset.split('/')[-1]
os.makedirs('datasets', exist_ok=True)

for shot in shots:
    output_path = f'datasets/{dataset_name}-{shot}shot.jsonl'
    with open(output_path, 'w', encoding='utf-8') as fout:
        # Use first 'shot' examples as few-shot examples
        fewshot_examples = prompts[:shot]
        
        # Randomly sample targets from the remaining prompts
        remaining_prompts = prompts[shot:]
        num_targets = min(100, len(remaining_prompts))
        target_prompts = random.sample(remaining_prompts, num_targets)
        
        for target in tqdm(target_prompts, desc=f"Processing shot={shot}"):
            fewshot_text = ''
            for example in fewshot_examples:
                prompt_text = USER_PROMPT_TEMPLATE.format(article=example['article'])
                fewshot_text += f"{prompt_text}{example['summary']}\n\n"

            final_prompt = (f"{SYSTEM_PROMPT}{fewshot_text}{USER_PROMPT_TEMPLATE.format(article=target['article'])}")

            record = {
                'article': final_prompt,
                'summary_gt': target['summary']
            }
            json.dump(record, fout, ensure_ascii=False)
            fout.write('\n')