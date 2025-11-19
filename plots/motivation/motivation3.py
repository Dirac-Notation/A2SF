import torch
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm
from transformers import AutoTokenizer, AutoModelForCausalLM

def compare(tensor_a, tensor_b):
    result = 0
    for i in range(tensor_a.size(0)):
        for j in range(tensor_a.size(1)):
            set_a = set(tensor_a[i,j].cpu().tolist())
            set_b = set(tensor_b[i,j].cpu().tolist())
            result += len(set_a & set_b) / len(set_a | set_b)
    return result / (tensor_a.size(0) * tensor_a.size(1))

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

text = """You are a highly knowledgeable AI assistant specialized in computational science and large language model optimization. Your task is to reason analytically and provide detailed technical explanations based on the provided contextual information. When answering, focus on clarity, correctness, and relevance to the given data.

In transformer-based LLMs, the KV (Key-Value) cache stores previously computed attention key and value tensors for each layer to avoid recomputation during autoregressive decoding. This significantly reduces inference time, especially for long sequences. However, memory usage grows linearly with sequence length and the number of layers, motivating research on cache compression and eviction strategies.

In classical mechanics, the motion of a particle is determined by Newton’s second law: F = m·a. When analyzing systems with variable mass or external fields, the law can be extended using generalized coordinates and Lagrangian mechanics. The total energy of the system is often conserved if no non-conservative forces are present.

Chemical reactions involve the rearrangement of atoms and the breaking or forming of chemical bonds. Reaction rates depend on temperature, pressure, and the activation energy barrier. Catalysts function by providing an alternative reaction pathway with a lower activation energy, increasing the rate without being consumed.

Let f(x) be a continuous differentiable function. Optimization methods such as gradient descent find local minima by iteratively updating x ← x − η∇f(x), where η is the learning rate. Convergence depends on η and the curvature of f(x). In high-dimensional spaces, saddle points can slow down convergence.

Large Language Models (LLMs) use multi-head self-attention to model dependencies between tokens. During inference, caching previous activations improves efficiency but limits flexibility for context updates. Recent research explores adaptive KV cache pruning, mixed-precision storage, and retrieval-augmented methods to optimize both performance and memory efficiency.

Based on the information above, explain how KV cache compression could be applied to reduce the memory footprint during long-sequence inference without significantly degrading model accuracy."""

input_ids = tokenizer(text, return_tensors="pt").input_ids

with torch.no_grad():
    output_ids = model.generate(input_ids, max_new_tokens=128, do_sample=False)
    outputs = model(output_ids, output_attentions=True)

prompt_length = input_ids.shape[1]
attention_maps = torch.stack(outputs.attentions, dim=0).squeeze(1)

prompt_part = attention_maps[:,:,:prompt_length,:prompt_length]
generation_part = attention_maps[:,:,prompt_length:,:]

plt.imshow(prompt_part.mean(dim=(0,1)).cpu().numpy(), cmap='Blues', norm=LogNorm())
plt.xticks([])
plt.yticks([])
plt.savefig("plots/motivation/motivation3.png")

answer = generation_part.sum(dim=2)[:,:,:prompt_length-5].topk(k=35).indices

hist_data = answer.cpu().numpy().flatten()

plt.figure(figsize=(8, 6))
plt.hist(hist_data, bins=40, range=(0, 399), color='royalblue', edgecolor='black', alpha=0.85)
plt.xlabel("Token Index")
plt.ylabel("Frequency")
plt.title("Answer", fontsize=20)
plt.tight_layout()
plt.savefig("plots/motivation/motivation3_answer.png")
plt.close()

h2o = prompt_part.sum(dim=2)[:,:,:prompt_length-5].topk(k=35).indices

hist_data = h2o.cpu().numpy().flatten()

plt.figure(figsize=(8, 6))
plt.hist(hist_data, bins=40, range=(0, 399), color='royalblue', edgecolor='black', alpha=0.85)
plt.xlabel("Token Index")
plt.ylabel("Frequency")
plt.title(f"H2O (similarity: {compare(answer, h2o):.2f})", fontsize=20)
plt.tight_layout()
plt.savefig("plots/motivation/motivation3_h2o.png")
plt.close()

snap = prompt_part[:,:,-16:,:].sum(dim=2)[:,:,:prompt_length-5].topk(k=35).indices

hist_data = snap.cpu().numpy().flatten()

plt.figure(figsize=(8, 6))
plt.hist(hist_data, bins=40, range=(0, 399), color='royalblue', edgecolor='black', alpha=0.85)
plt.xlabel("Token Index")
plt.ylabel("Frequency")
plt.title(f"SNAP (similarity: {compare(answer, snap):.2f})", fontsize=20)
plt.tight_layout()
plt.savefig("plots/motivation/motivation3_snap.png")
plt.close()

tova = prompt_part[:,:,-1:,:].sum(dim=2)[:,:,:prompt_length-5].topk(k=35).indices

hist_data = tova.cpu().numpy().flatten()

plt.figure(figsize=(8, 6))
plt.hist(hist_data, bins=40, range=(0, 399), color='royalblue', edgecolor='black', alpha=0.85)
plt.xlabel("Token Index")
plt.ylabel("Frequency")
plt.title(f"TOVA (similarity: {compare(answer, tova):.2f})", fontsize=20)
plt.tight_layout()
plt.savefig("plots/motivation/motivation3_tova.png")
plt.close()

exponents = torch.arange(prompt_length-1, -1, -1).to(prompt_part.device)
a2sf = (prompt_part * (0.8**exponents.view(1,1,-1,1))).sum(dim=2)[:,:,:prompt_length-5].topk(k=35).indices

hist_data = a2sf.cpu().numpy().flatten()

plt.figure(figsize=(8, 6))
plt.hist(hist_data, bins=40, range=(0, 399), color='royalblue', edgecolor='black', alpha=0.85)
plt.xlabel("Token Index")
plt.ylabel("Frequency")
plt.title(f"A2SF (similarity: {compare(answer, a2sf):.2f})", fontsize=20)
plt.tight_layout()
plt.savefig("plots/motivation/motivation3_a2sf.png")
plt.close()

linear = torch.maximum(prompt_part - 0.01*exponents.view(1,1,-1,1), torch.zeros_like(prompt_part)).sum(dim=2)[:,:,:prompt_length-5].topk(k=35).indices

hist_data = linear.cpu().numpy().flatten()

plt.figure(figsize=(8, 6))
plt.hist(hist_data, bins=40, range=(0, 399), color='royalblue', edgecolor='black', alpha=0.85)
plt.xlabel("Token Index")
plt.ylabel("Frequency")
plt.title(f"Linear (similarity: {compare(answer, linear):.2f})", fontsize=20)
plt.tight_layout()
plt.savefig("plots/motivation/motivation3_linear.png")
plt.close()