import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

model_path = "togethercomputer/Llama-2-7B-32K-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, skip_special_tokens=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

queries = {}

keies = {}

values = {}

hook_handles = []

def make_hook(layer_idx, attn_module, op_type):
    num_heads = getattr(attn_module, "num_heads", None)
    head_dim  = getattr(attn_module, "head_dim", None)
    rotary    = getattr(attn_module, "rotary_emb", None)

    def hook_fn(module, inputs, output):
        tensors = output
        
        bsz, seqlen, _ = tensors.shape

        tensors = tensors.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)

        if op_type == "v":
            values[layer_idx] = tensors.detach().to("cpu")
            return
            
        position_ids = torch.arange(seqlen, device=tensors.device)[None,:].to(tensors.device)

        cos, sin = rotary(tensors, seqlen)
        tensors_rope, _ = apply_rotary_pos_emb(tensors, tensors, cos, sin, position_ids)

        if op_type == "q":
            queries[layer_idx] = tensors_rope.detach().to("cpu")
        elif op_type == "k":
            keies[layer_idx] = tensors_rope.detach().to("cpu")

    return hook_fn

def get_skew_matrix(matrix):
    _, _, V = torch.svd(matrix)
    return V

def plot_matrix(matrix, save_path):
    matrix = matrix.T.abs()
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    x = torch.arange(matrix.shape[0])
    y = torch.arange(matrix.shape[1])
    x, y = torch.meshgrid(x, y, indexing="ij")

    ax.plot_surface(x, y, matrix, cmap="viridis")

    ax.set_xlabel("Channel")
    ax.set_ylabel("Token ID")
    ax.set_zlabel("Absolute Value")

    plt.savefig(save_path)
    plt.close()

for layer_idx in range(model.config.num_hidden_layers):
    attn = model.model.layers[layer_idx].self_attn
        
    handle_q = attn.q_proj.register_forward_hook(make_hook(layer_idx, attn, "q"))
    handle_k = attn.k_proj.register_forward_hook(make_hook(layer_idx, attn, "k"))
    handle_v = attn.v_proj.register_forward_hook(make_hook(layer_idx, attn, "v"))
    
    hook_handles.append(handle_q)
    hook_handles.append(handle_k)
    hook_handles.append(handle_v)

text = "You are a helpful assistant that can summarize the text.\n\n<text>\nWe have all been there. Wanting to get online but just out of wifi coverage. However in central China, Mother Nature recently appeared to lend a helping hand by giving locals what could be the world's largest 'Wi-Fi cloud'. A cloud formation resembling the Wi-Fi symbol appeared recently in the sky of Xiangtan city in Hunan Province, the People's Daily Online reports. God's router? A cloud formation resembling the Wi-Fi symbol appeared in central China last week . The global Wi-Fi symbol has one dot and three curved lines radiating from it"

input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

with torch.no_grad():
    outputs = model(input_ids)

for layer_idx in tqdm(range(model.config.num_hidden_layers)):
    query = queries[layer_idx]
    key = keies[layer_idx]
    value = values[layer_idx]

    for head_idx in range(model.config.num_attention_heads):
        save_dir = os.path.join(DIR_PATH, "pre_skewing", f"layer{layer_idx}", f"head{head_idx}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        query_head = query[0, head_idx].float()
        key_head = key[0, head_idx].float()
        value_head = value[0, head_idx].float()

        plot_matrix(query_head, os.path.join(save_dir, "query.png"))
        plot_matrix(key_head, os.path.join(save_dir, "key.png"))
        plot_matrix(value_head, os.path.join(save_dir, "values.png"))

        save_dir = os.path.join(DIR_PATH, "post_skewing", f"layer{layer_idx}", f"head{head_idx}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        qk_skew_matrix = get_skew_matrix(key_head)
        v_skew_matrix = get_skew_matrix(value_head)
        
        query_head_skew = query_head @ qk_skew_matrix
        key_head_skew = key_head @ qk_skew_matrix
        value_head_skew = value_head @ v_skew_matrix

        plot_matrix(query_head_skew, os.path.join(save_dir, "query.png"))
        plot_matrix(key_head_skew, os.path.join(save_dir, "key.png"))
        plot_matrix(value_head_skew, os.path.join(save_dir, "values.png"))

for h in hook_handles:
    h.remove()