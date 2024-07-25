import re

def extract_dataset_acc(text):
    lines = text.split('\n')
    results = {}
    current_key = ''
    for idx, line in enumerate(lines):
        if '====' in line:
            current_key = lines[idx+1]
            results[current_key] = []
        elif '|acc     |' in line:
            parts = re.split(r'\|', line)
            dataset = parts[1].strip()
            acc = parts[6].strip()
            results[current_key].append((dataset, acc))
    return results

with open("1shot_result.txt", "r") as f:
    text = f.read()

results = extract_dataset_acc(text)

dataset_results = {}
for key in results:
    for dataset, acc in results[key]:
        if dataset not in dataset_results:
            dataset_results[dataset] = []
        dataset_results[dataset].append((key, acc))

for dataset in dataset_results:
    print(dataset)
    for key, acc in dataset_results[dataset]:
        print(f"{key} {acc}")
    print()
