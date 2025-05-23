import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

def split_seq(seq, patterns=[29889, 869]):
    segments = []
    start = 0
    length = seq.size(0)
    i = 0

    while i < length - 1:
        if int(seq[i]) in patterns:
            segments.append(seq[start : i + 1])
            start = i + 1
        i += 1

    if start < length:
        segments.append(seq[start : length])

    return segments

def calculate_sentence_attention(attention_maps, sentence_segments):
    num_sentences = len(sentence_segments)
    sentence_attention = np.zeros((num_sentences, num_sentences))
    
    # Calculate average attention between sentences
    for i in range(num_sentences):
        for j in range(num_sentences):
            if i >= j:  # Only calculate attention for current and future sentences
                start_i = 0 if i == 0 else sum(len(s) for s in sentence_segments[:i])
                end_i = start_i + len(sentence_segments[i])
                start_j = 0 if j == 0 else sum(len(s) for s in sentence_segments[:j])
                end_j = start_j + len(sentence_segments[j])
                
                # Average attention across all layers and heads
                attention = attention_maps[:, :, start_i:end_i, start_j:end_j].mean() * 100
                sentence_attention[i, j] = 2*attention.item() if i == j else attention.item()
    
    return sentence_attention

def plot_attention_heatmap(attention_matrix, save_path):
    plt.figure(figsize=(10, 8))
    im = plt.imshow(attention_matrix, cmap='Blues', norm=LogNorm())
    cbar = plt.colorbar(im, label='Attention Score')
    cbar.ax.tick_params(labelsize=25)
    cbar.set_label('Attention Score', fontsize=28)
    
    # Add text annotations
    for i in range(attention_matrix.shape[0]):
        for j in range(attention_matrix.shape[1]):
            plt.text(j, i, f'{attention_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=25)
    
    # Move x-axis to top
    ax = plt.gca()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    plt.xlabel("Key Sentence", fontsize=28)
    plt.ylabel("Query Sentence", fontsize=28)
    plt.xticks(fontsize=25) 
    plt.yticks(fontsize=25)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_full_attention_heatmap(attention_maps, sentence_segments, save_path):
    # Calculate cumulative lengths for sentence boundaries
    cum_lengths = [0]
    for seg in sentence_segments:
        cum_lengths.append(cum_lengths[-1] + len(seg))
    
    # Average attention across all layers and heads
    full_attention = attention_maps.mean(dim=(0, 1)).cpu().numpy()
    
    plt.figure(figsize=(12, 10))
    im = plt.imshow(full_attention, cmap='Blues', aspect='auto', norm=LogNorm())
    
    # Add colorbar with correct fontsize setting
    cbar = plt.colorbar(im)
    cbar.set_label('Attention Score', fontsize=28)
    cbar.ax.tick_params(labelsize=25)
    
    # Add sentence boundary lines
    for length in cum_lengths[1:-1]:  # Skip first and last boundaries
        plt.axhline(y=length-0.5, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=length-0.5, color='red', linestyle='--', alpha=0.5)
    
    # Move x-axis to top
    ax = plt.gca()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    plt.xlabel("Key Token", fontsize=28)
    plt.ylabel("Query Token", fontsize=28)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

dataset_path = "datasets/cnn_dailymail-0shot.jsonl"

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(torch.float16).to(device)

with open(dataset_path, "r") as f:
    datalines = f.readlines()
    articles = []
    
    for dataline in datalines:
        articles.append(json.loads(dataline))

# Process articles from 0 to 100
for article_idx in tqdm(range(100)):
    # Create directory for this article
    article_dir = f"plots/sentences/article_{article_idx}"
    os.makedirs(article_dir, exist_ok=True)
    
    input = f"[INST]{articles[article_idx]['article']}[/INST]"
    input_ids = tokenizer(input, return_tensors="pt").input_ids

    with torch.no_grad():
        outputs = model(input_ids.to(device), output_attentions=True)
    attention_maps = torch.cat(outputs.attentions, dim=0)

    # After getting attention_maps, add:
    sentence_segments = split_seq(input_ids[0])
    sentence_attention = calculate_sentence_attention(attention_maps, sentence_segments)
    
    # Save plots in article-specific directory
    plot_attention_heatmap(sentence_attention, f"{article_dir}/sentence_attention_heatmap.png")
    plot_full_attention_heatmap(attention_maps, sentence_segments, f"{article_dir}/full_attention_heatmap.png")
    
    # Save sentence segments to a text file
    with open(f"{article_dir}/sentences.txt", "w") as f:
        for idx, text in enumerate(sentence_segments):
            f.write(f"{idx}: {tokenizer.decode(text)}\n")