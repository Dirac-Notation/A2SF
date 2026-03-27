import torch
import json
import os
import numpy as np
import random

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils_real_drop import KVLlamaForCausalLM

class CompressionConfig(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model(model_name):
    """
    Load model and tokenizer based on model name.
    
    Args:
        model_name (str): Name of the model (e.g., 'llama2', 'llama3', 'opt', 'qwen2')
    
    Returns:
        tuple: (model, tokenizer)
    """
    model2path = json.load(open("config/model2path.json", "r"))
    model_path = model2path[model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if "llama" in model_name.lower():
        model = KVLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}. Only Llama and OPT models are supported.")
    
    model = model.eval()
    
    return model, tokenizer