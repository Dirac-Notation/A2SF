import json
from pandas.core.window.rolling import Window
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from a2sf_search import make_sentence_exp, get_punctuation_ids

MAX_LENGTH = 1024

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

import os

# longbench_dir = "result_txt/longbench"
# files = os.listdir(longbench_dir)

# for file in files:
#     if not file.endswith('.jsonl'):
#         continue
        
#     with open(os.path.join(longbench_dir, file), "r") as f:
#         lines = f.readlines()

#     dataset = [json.loads(line) for line in lines]

#     buffer = []
#     for data in dataset:
#         input_ids = tokenizer.encode(data["input_prompt"], return_tensors="pt")
#         input_ids = torch.cat([input_ids[:,:MAX_LENGTH//2], input_ids[:,-MAX_LENGTH//2:]], dim=1)

#         sentence_exp = make_sentence_exp(input_ids, get_punctuation_ids(tokenizer))
#         num_sentence = (sentence_exp == 0).float().sum().item()
#         buffer.append(num_sentence)

#     print(f"{file}: {sum(buffer)/len(buffer)}")

with open("result_txt/longbench/lcc.jsonl", "r") as f:
    lines = f.readlines()

dataset = [json.loads(line) for line in lines]

batch_input_ids = []
for i in range(4):
    input_ids = tokenizer.encode(dataset[i]["input_prompt"], return_tensors="pt").to(model.device)
    input_ids = torch.cat([input_ids[:,:MAX_LENGTH//2], input_ids[:,-MAX_LENGTH//2:]], dim=1)
    batch_input_ids.append(input_ids)

batch_input_ids = torch.cat(batch_input_ids, dim=0)

output_ids = model.generate(batch_input_ids, max_new_tokens=1, min_new_tokens=1)
output = model(output_ids, output_attentions=True)

# h2o = 64
# snap = 16

# for attention in output.attentions:
#     h2o_score = attention[:,:,-h2o:,:-h2o].sum(dim=2)
#     snap_score = attention[:,:,-snap:,:-snap].sum(dim=2)
    
#     h2o_index = h2o_score.topk(k=128-h2o).indices.sort().values
#     snap_index = snap_score.topk(k=128-snap).indices.sort().values

#     h2o_result = torch.cat([attention[:,:,-1,:].gather(dim=2, index=h2o_index), attention[:,:,-1,-h2o:]], dim=2)
#     snap_result = torch.cat([attention[:,:,-1,:].gather(dim=2, index=snap_index), attention[:,:,-1,-snap:]], dim=2)

#     print(h2o_result.sum(dim=2).mean().item())
#     print(snap_result.sum(dim=2).mean().item())
#     print()

def sim(a, b):
    # a, b의 shape: (4, 32, 64)
    # 각 배치와 헤드에 대해 Jaccard similarity 계산
    batch_size, num_heads, k = a.shape
    
    similarities = []
    for i in range(batch_size):
        for j in range(num_heads):
            # a[i,j]와 b[i,j]는 각각 길이 64의 인덱스 배열
            set_a = set(a[i,j].cpu().numpy())
            set_b = set(b[i,j].cpu().numpy())
            
            # 교집합과 합집합 계산
            intersection = len(set_a.intersection(set_b))
            union = len(set_a.union(set_b))
            
            # Jaccard similarity: 교집합 / 합집합
            if union == 0:
                similarity = 0.0
            else:
                similarity = intersection / union
            
            similarities.append(similarity)
    
    # 전체 평균 반환
    return sum(similarities) / len(similarities)

for window in [5,10,20,30,40,50,60,70,80,90,100]:
    print(f"window: {window}")
    for attention in output.attentions:
        a_score = attention[:,:,-window-1:-1,:-64].sum(dim=2)
        b_score = attention[:,:,-1,:-64]
        
        a_index = a_score.topk(k=64).indices.sort().values
        b_index = b_score.topk(k=64).indices.sort().values

        print(sim(a_index, b_index))

    print()