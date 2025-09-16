import json
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

with open("datasets/calibration_dataset.jsonl", "r") as f:
    articles = []
    for line in f:
        line_data = json.loads(line)
        article = line_data["article"]
        articles.append(article)

for article in articles:
    input_ids = tokenizer(article, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    # output = model(input_ids, output_attentions=True)
    output = model(input_ids)
    
    import pdb; pdb.set_trace()