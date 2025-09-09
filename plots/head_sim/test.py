import torch
import json
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set larger font sizes for better readability
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14

def jaccard_similarity(a, b):
    return len(a.intersection(b)) / len(a.union(b))

def head_sim(attention_maps):
    num_layers = len(attention_maps)
    num_heads = attention_maps[0].size(1)
    
    head_set = [[] for _ in range(num_layers)]
    for layer in range(num_layers):
        for head in range(num_heads):
            att = attention_maps[layer][0, head]
            score = att.sum(dim=0)
            idx = set(score.topk(int(score.size(0) * 0.1), dim=0).indices.tolist())
            head_set[layer].append(idx)
    
    result = torch.zeros(num_layers, num_heads, num_heads)
    for i in range(num_layers):
        for j in range(num_heads):
            for k in range(num_heads):
                result[i,j,k] = jaccard_similarity(head_set[i][j], head_set[i][k])

    return result

dataset_path = "plots/head_sim/xsum-5shot.jsonl"

with open(dataset_path, "r") as f:
    datalines = f.readlines()
    articles = []
    
    for dataline in datalines:
        articles.append(json.loads(dataline))

# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "facebook/opt-6.7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(torch.float16).to("cuda")

max_token_length = model.config.max_position_embeddings

inputs = []
for article in articles:
    input_ids = tokenizer(article["article"], return_tensors="pt").input_ids
    if input_ids.numel() > max_token_length:
        input_ids = torch.cat([input_ids[:, :max_token_length//2], input_ids[:, -max_token_length//2:]], dim=1)
    inputs.append(input_ids)

num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads
results = torch.zeros(num_layers, num_heads, num_heads)

with torch.no_grad():
    for input_ids in tqdm(inputs):
        outputs = model(input_ids.to("cuda"), output_attentions=True)
        attention_maps = outputs.attentions
        results +=head_sim(attention_maps)
        del outputs, attention_maps
        torch.cuda.empty_cache()
results /= len(inputs)

for i in range(num_layers):
    plt.figure(figsize=(12, 10))
    plt.imshow(results[i].cpu().numpy(), cmap='Blues', aspect='equal')
    plt.colorbar()
    plt.xlabel('Head Index')
    plt.ylabel('Head Index')
    plt.title(f'Layer {i}')
    plt.tight_layout()
    plt.savefig(f'plots/head_sim/head_sim_{i}.png')
    plt.close()