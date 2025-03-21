import os
import json
import argparse

from datasets import load_dataset
from tqdm import tqdm

def count_line(text):
    lines = text.split(". ")
    
    return len(lines)

parser = argparse.ArgumentParser(description="Process dataset for few-shot learning.")
parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset. alexfabbri/multi_news / abisee/cnn_dailymail / EdinburghNLP/xsum")
parser.add_argument('--shots', type=int, nargs="+", required=True, help="Number of examples for few-shot learning.")

args = parser.parse_args()

dataset_path = args.dataset_path # alexfabbri/multi_news / abisee/cnn_dailymail / EdinburghNLP/xsum
shots = args.shots

if dataset_path != "abisee/cnn_dailymail":
    dataset = load_dataset(dataset_path)
else:
    dataset = load_dataset(dataset_path, "2.0.0")
dataset = dataset["test"]

prompts = []

for line in dataset:
    article, highlight = [item for item in line.values()][:2]
    num_line = count_line(highlight)
    
    prompt = f"[INST] Article: {article}\nQ: Summarize the above article briefly in {num_line} sentences. [/INST]\nA: "
    text_dict = {"a": prompt, "b": highlight}
    prompts.append(text_dict)

prompts = sorted(prompts, key=lambda x: len(x["a"]))
prompts = prompts[200:]

dataset_name = dataset_path.split("/")[1]

os.makedirs("fewshot_data", exist_ok=True)
for shot in shots:
    with open(f"fewshot_data/{dataset_name}-{shot}shot.jsonl", "w") as file:
        for line in tqdm(prompts[shot:100+shot]):
            tmp_text = "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
            
            for examples in prompts[:shot]:
                article, highlight = examples.values()
                tmp_text += article
                tmp_text += highlight
                tmp_text += "\n\n"
            tmp_text += line["a"]
            
            json.dump({"article": tmp_text, "summary_gt": line["b"]}, file)
            file.write("\n")