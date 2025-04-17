import os
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration: define prompt templates for clarity and reusability
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = '''<<SYS>>
You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible while ensuring that your responses are safe, unbiased, and positive.
If a question is nonsensical or factually incoherent, explain why rather than providing incorrect information.
If you don't know the answer, state that clearly instead of making up an answer.
<</SYS>>'''

USER_PROMPT_TEMPLATE = '''[INST] Article: {article}
Q: Summarize the above article briefly. [/INST]
A: '''

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Process dataset for few-shot learning.")
parser.add_argument('--dataset', type=str, required=True,
                    help="Name of the dataset (e.g., alexfabbri/multi_news, abisee/cnn_dailymail, EdinburghNLP/xsum)")
parser.add_argument('--shots', type=int, nargs='+', required=True,
                    help="Number of examples for few-shot learning.")
args = parser.parse_args()
shots = args.shots

# -----------------------------------------------------------------------------
# Load dataset (use version 2.0.0 for cnn_dailymail)
# -----------------------------------------------------------------------------
dataset_obj = (load_dataset(args.dataset, "2.0.0")
               if args.dataset == "abisee/cnn_dailymail"
               else load_dataset(args.dataset))
dataset = dataset_obj['test']

# Prepare base prompts (article and reference summary)
prompts = []
for entry in dataset:
    article, highlight = list(entry.values())[:2]
    prompts.append({
        'article': article,
        'summary': highlight
    })

# Sort by article length and skip the first 1000 shortest examples
prompts = sorted(prompts, key=lambda x: len(x['article']))[1000:]

# Extract dataset name for output filenames
dataset_name = args.dataset.split('/')[-1]

# Create output directory
os.makedirs('fewshot_data', exist_ok=True)

# Generate few-shot files
for shot in shots:
    output_path = f'fewshot_data/{dataset_name}-{shot}shot.jsonl'
    with open(output_path, 'w', encoding='utf-8') as fout:
        # For each target example after the initial skip
        for target in tqdm(prompts[shot:shot+100], desc=f"Processing shot={shot}"):
            # Build few-shot examples from the first 'shot' items
            fewshot_examples = ''
            for example in prompts[:shot]:
                # Format each example using the same user prompt template
                prompt_text = USER_PROMPT_TEMPLATE.format(article=example['article'])
                fewshot_examples += f"{prompt_text}{example['summary']}\n\n"

            # Build the final prompt combining system, few-shot examples, and current article
            final_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"{fewshot_examples}"
                f"{USER_PROMPT_TEMPLATE.format(article=target['article'])}"
            )

            # Write to JSONL file
            record = {
                'article': final_prompt,
                'summary_gt': target['summary']
            }
            json.dump(record, fout, ensure_ascii=False)
            fout.write('\n')