import json
import torch
from typing import List

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")

def encode_context(text, max_tokens) -> torch.Tensor:
    tokens = tokenizer.encode(text, add_special_tokens=False)

    batched_tokens = []
    for i in range(len(tokens), 0, -max_tokens):
        if i < max_tokens:
            batched_tokens.append(tokens[:i])
        else:    
            batched_tokens.append(tokens[i-max_tokens:i])
    batched_tokens.reverse()

    batched_text = []
    for tokens in batched_tokens:
        batched_text.append(tokenizer.decode(tokens))

    # Encode using sentence transformer
    with torch.no_grad():
        embedding = sentence_transformer.encode(
            batched_text,
            batch_size=len(batched_text),
            convert_to_tensor=True,
            normalize_embeddings=True
        )
    return embedding

def load_data():
    with open("datasets/training_data.json", "r") as f:
        data = json.load(f)
    return data

def similarity(embedding):
    cross_cosine_sim = torch.matmul(embedding, embedding.transpose(0, 1))
    v1_norm = torch.norm(embedding, dim=1, keepdim=True)
    v2_norm = torch.norm(embedding, dim=1, keepdim=True)
    cross_norm = torch.matmul(v1_norm, v2_norm.transpose(0, 1))
    cross_cosine_sim = cross_cosine_sim / cross_norm
    return cross_cosine_sim

data = load_data()

for item in data:
    embedding = encode_context(item["input_prompt"], 512)
    print(similarity(embedding)[-1].shape)
    break