import json
import torch

from tqdm import tqdm

def load_datasets(
    dataset_path: str,
    tokenizer
):
    with open(dataset_path, "r") as f:
        prompts = []
        answers = []
        
        datalines = f.readlines()
        articles = []
        
        for dataline in datalines:
            articles.append(json.loads(dataline))
        
    for data in tqdm(articles, desc="Tokenizing"):
        input = data["article"]
        answer = data["summary_gt"]
        
        input_ids = tokenizer(input, return_tensors="pt").input_ids

        prompts.append(input_ids)
        answers.append(answer)
    
    return prompts, answers

def sim(
    vec_1: torch.Tensor,
    vec_2: torch.Tensor,
    k: int = 128
):
    if vec_1.dim() != 1 or vec_2.dim() != 1:
        assert "must dim 1"
    
    set_1 = set(vec_1.topk(k).indices.tolist())
    set_2 = set(vec_2.topk(k).indices.tolist())
    
    return len(set_1&set_2)/len(set_1|set_2)

def diff(
    vec_1: torch.Tensor,
    vec_2: torch.Tensor,
    vec_3: torch.Tensor,
    k: int = 128
):
    if vec_1.dim() != 1 or vec_2.dim() != 1 or vec_3.dim() != 1:
        assert "must dim 1"
    
    set_1 = set(vec_1.topk(k).indices.tolist())
    set_2 = set(vec_2.topk(k).indices.tolist())
    set_3 = set(vec_3.topk(25).indices.tolist())
    
    union_all = set_1&set_2&set_3
    union_12 = set_1&set_2 - union_all
    union_13 = set_1&set_3 - union_all
    union_23 = set_2&set_3 - union_all

    num_union_all = len(union_all)
    num_union_12 = len(union_12)
    num_union_13 = len(union_13)
    num_union_23 = len(union_23)

    try:
        score_union_all = vec_1[torch.tensor(list(union_all))].sum().item()
    except:
        score_union_all = 0

    try:
        score_union_12 = vec_1[torch.tensor(list(union_12))].sum().item()
    except:
        score_union_12 = 0

    try:
        score_union_13 = vec_1[torch.tensor(list(union_13))].sum().item()
    except:
        score_union_13 = 0
    
    return num_union_12, num_union_13, num_union_23, num_union_all, score_union_12, score_union_13, score_union_all, vec_1[torch.tensor(list(set_1))].sum()