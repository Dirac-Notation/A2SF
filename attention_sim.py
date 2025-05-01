import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import get_prompt, make_optimal_mask, make_a2sf_mask

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()
device = f"cuda:{args.gpu}"

prompt = get_prompt()

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

with torch.no_grad():
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs, output_attentions=True)
    
    attn_probs = outputs.attentions
    attention_maps = torch.stack(outputs.attentions).squeeze(1)
