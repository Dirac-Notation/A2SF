import argparse
import random
import os
import json

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description="Process dataset for few-shot learning.")
parser.add_argument('--dataset', type=str, required=True, help="abisee/cnn_dailymail, EdinburghNLP/xsum, alexfabbri/multi_news, Samsung/samsum")
parser.add_argument('--shots', type=int, nargs='+', required=True, help="Number of examples for few-shot learning.")

args = parser.parse_args()
shots = args.shots

# Set random seed for reproducibility
random.seed(42)

SYSTEM_PROMPT = 'You are a helpful assistant that can summarize the text.\n\n'
USER_PROMPT_TEMPLATE = '<text>\n{article}\n</text>\n\nSummarize the key points from the text above. Provide a direct response.'

# Load Llama2 tokenizer for token counting
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
MIN_TOKENS = 1000
MAX_TOKENS = 3000  # Changed to 1000 tokens
FEWSHOT_SAMPLES = 100  # Number of few-shot samples to create

dataset_obj = (load_dataset(args.dataset, "2.0.0", trust_remote_code=True) if args.dataset == "abisee/cnn_dailymail" else load_dataset(args.dataset, trust_remote_code=True))
dataset = dataset_obj['test']

prompts = []
print("Loading and filtering dataset...")
for entry in tqdm(dataset, desc="Processing dataset entries"):
    if args.dataset in ["abisee/cnn_dailymail", "EdinburghNLP/xsum", "alexfabbri/multi_news"]:
        article, highlight = list(entry.values())[:2]
    elif args.dataset in ["Samsung/samsum"]:
        article, highlight = list(entry.values())[1:3]
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    # Count tokens for article and summary
    article_tokens = len(tokenizer.encode(article))
    summary_tokens = len(tokenizer.encode(highlight))
    
    # Only include if total tokens (input + output) is within limit
    if article_tokens + summary_tokens >= MIN_TOKENS and article_tokens + summary_tokens <= MAX_TOKENS:
        prompts.append({
            'article': article,
            'summary': highlight,
            'article_tokens': article_tokens,
            'summary_tokens': summary_tokens
        })

# Filter prompts by token count and randomly sample
print(f"Total prompts after token filtering: {len(prompts)}")

# Randomly sample prompts for few-shot learning
# We need FEWSHOT_SAMPLES + max(shots) prompts to create 100 samples for each shot configuration
max_shot = max(shots) if shots else 0
required_prompts = FEWSHOT_SAMPLES + max_shot

if len(prompts) >= required_prompts:
    print(f"Randomly sampling {required_prompts} prompts (100 + {max_shot} for few-shot examples)...")
    prompts = random.sample(prompts, required_prompts)
    print(f"✓ Randomly sampled {required_prompts} prompts")
else:
    print(f"⚠️  Warning: Only {len(prompts)} prompts available, using all of them")
    print(f"   This will create {len(prompts) - max_shot} samples instead of 100")

dataset_name = args.dataset.split('/')[-1]
os.makedirs('datasets', exist_ok=True)

print(f"\nCreating few-shot datasets for {dataset_name}...")
for shot_idx, shot in enumerate(shots):
    output_path = f'datasets/{dataset_name}-{shot}shot.jsonl'
    with open(output_path, 'w', encoding='utf-8') as fout:
        # Use first 'shot' examples as few-shot examples
        fewshot_examples = prompts[:shot]
        
        # Create 100 few-shot samples
        num_samples = FEWSHOT_SAMPLES
        
        # Create exactly 100 few-shot samples
        for i in tqdm(range(num_samples), desc=f"Creating {shot}-shot samples", position=0, leave=False):
            # Use the (shot + i)th prompt as target
            target_idx = shot + i
            target = prompts[target_idx]
            
            # Build few-shot examples text
            fewshot_text = ''
            for example in fewshot_examples:
                prompt_text = USER_PROMPT_TEMPLATE.format(article=example['article'])
                fewshot_text += f"{prompt_text}{example['summary']}\n\n"

            # Create final prompt
            final_prompt = (f"{SYSTEM_PROMPT}{fewshot_text}{USER_PROMPT_TEMPLATE.format(article=target['article'])}")
            
            # Count total tokens in the final prompt
            total_input_tokens = len(tokenizer.encode(final_prompt))
            total_output_tokens = target['summary_tokens']
            
            # Create record (always save, since we already filtered by individual article+summary length)
            record = {
                'article': final_prompt,
                'summary_gt': target['summary'],
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'total_tokens': total_input_tokens + total_output_tokens
            }
            json.dump(record, fout, ensure_ascii=False)
            fout.write('\n')
        
        print(f"✓ Created {output_path} with {num_samples} samples")