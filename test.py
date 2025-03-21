import torch

from transformers import AutoTokenizer

from utils import load_datasets
from utils_real_drop.kv_llama import LlamaForCausalLM

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).half().to("cuda")
model.init_cache()

dataset = load_datasets(dataset_path="fewshot_data/cnn_dailymail-0shot.jsonl", tokenizer=tokenizer)

input_ids = dataset[0].to(model.device)

print(tokenizer.decode(model.generate(input_ids, max_new_tokens=64).flatten()[input_ids.numel():].tolist()))