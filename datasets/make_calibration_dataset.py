import json
import random
import os
from pathlib import Path
from datetime import datetime

def get_data_group(dataset):
    for group, datasets in data_group.items():
        if dataset in datasets:
            return group
    return None

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

data_group = {
    "Code Complete": ["repobench-p", "lcc"],
    "Few Shot": ["trec", "triviaqa", "samsum", "lsht"],
    "Single-doc QA": ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
    "Multi-doc QA": ["hotpotqa", "2wikimqa", "musique", "dureader"],
    "Summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "Passage Retrieval": ["passage_retrieval_en", "passage_retrieval_zh", "passage_count"],
}

longbench_dir = Path("datasets/longbench")

output_dir = Path("datasets")
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "calibration_dataset.jsonl"

all_converted_data = []
total_converted = 0

for dataset in datasets:
    group = get_data_group(dataset)
    print(f"Processing \"{group}\" - \"{dataset}\"")
    
    temp_path = longbench_dir / f"{dataset}.jsonl"
    
    file_data = []
    
    with open(temp_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            data = json.loads(line)
            file_data.append(data)
    
    print(f"  - \"{dataset}\": {len(file_data)} data found")
    
    # random sample 2 data
    selected_data = random.sample(file_data, 2)

    for item in selected_data:
        article = item['input_prompt']
        input_tokens = len(article.split())
        
        # create converted data
        converted_item = {
            "group": group,
            "dataset": dataset,
            "article": article,
            "input_tokens": input_tokens,
        }
        
        all_converted_data.append(converted_item)
    
    print(f"  - \"{dataset}\": {len(selected_data)} data converted")
    total_converted += len(selected_data)

# save all data to one file
with open(output_file, 'w', encoding='utf-8') as f:
    for item in all_converted_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nTotal {total_converted} data converted")
print(f"All data saved to {output_file}")