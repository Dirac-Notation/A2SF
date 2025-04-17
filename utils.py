import json
import torch

from tqdm import tqdm

def load_datasets(
    dataset_path: str,
    tokenizer
):
    with open(dataset_path, "r") as f:
        datalines = f.readlines()
        articles = []
        
        for dataline in datalines:
            articles.append(json.loads(dataline))

    prompts = []
    answers = []
    output_indices = []

    for data in tqdm(articles, desc="Tokenizing"):
        input = data["article"]
        answer = data["summary_gt"]
        
        input_ids = tokenizer(input, return_tensors="pt").input_ids
        output_ids = tokenizer(answer, return_tensors="pt").input_ids

        prompts.append(input_ids)
        answers.append(answer)
        output_indices.append(output_ids)
    
    num_input_ids = sum([prompt.numel() for prompt in prompts])/len(prompts)
    num_output_ids = sum([output_ids.numel() for output_ids in output_indices])/len(output_indices)
    
    print(f"Average input ids length : {num_input_ids:.2f}")
    print(f"Average output ids length : {num_output_ids:.2f}")
    
    return prompts, answers, output_indices

def sim(
    vec_1: torch.Tensor,
    vec_2: torch.Tensor,
    k: int = 20,
):
    if vec_1.dim() != 1 or vec_2.dim() != 1:
        assert "must dim 1"

    index_set = vec_2.topk(k).indices
    
    return (vec_1.index_select(dim=0, index=index_set)).sum()