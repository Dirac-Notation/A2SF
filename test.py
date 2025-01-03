import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from scipy.optimize import curve_fit
from tqdm import tqdm

def fitting(x, a, b):
    return a*x + b

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
        
        if len(content) == 0:
            continue
            
        combined_texts.append({"text": title + "".join(content)})
    return combined_texts

device = "cuda:1"

model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "facebook/opt-2.7b"

model = AutoModelForCausalLM.from_pretrained(model_name).half().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

ds_test = ds["test"]["text"]

ds_test = wiki_preprocess(ds_test)

for idx in range(4):
    text = ""
    for shot in range(3):
        text += ds_test[3*idx + shot]["text"] + "\n"
    texts = tokenizer.tokenize(text)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():
        outputs = model(
            inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            output_attentions=True
        )

    attentions = [attention.cpu() for attention in outputs.attentions]

    for layer in tqdm(range(32)):
        for head in range(32):
            data = attentions[layer][0,head].to(torch.float)
            num_tokens = int(data.size(-1)*0.95)
            y = torch.empty(num_tokens)
            for i in range(int(num_tokens)):
                tmp = data.diagonal(offset=-i).sort()[0]
                y[i] = tmp[:int(tmp.size(0)*0.95)].mean()
            y = np.log(y.numpy() + 1e-10)
            x = np.arange(0, y.size)
            params, covariance = curve_fit(fitting, x, y, p0=[0.0, 0.0])
            a_est, b_est = params
            os.makedirs(f"tmp/prompt_{idx}/{layer}", exist_ok=True)
            
            plt.figure(figsize=(12,6))
            
            plt.subplot(1,2,1)
            plt.imshow(data.abs().pow(1/3), cmap="Blues")
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(1,2,2)
            plt.title(f"{a_est:.3f} / {b_est:.3f}")
            plt.plot(x, y)
            plt.plot(x, fitting(x, a_est, b_est))
            plt.savefig(f"tmp/prompt_{idx}/{layer}/{head}.png")
            plt.close()
            
            plt.tight_layout()