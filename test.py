import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def wiki_preprocess(ds):
    combined_texts = []
    i = 0
    while i < len(ds):
        if ds[i] == '':
            i += 1
            continue
        
        title = ds[i]
        i += 2
        
        content = []
        while i < len(ds) and ds[i] != '':
            content.append(ds[i])
            i += 1

        combined_texts.append({"text": title + "".join(content)})
    return combined_texts

model_name = "meta-llama/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

ds_test = ds["test"]["text"]

ds_test = wiki_preprocess(ds_test)

inputs = tokenizer