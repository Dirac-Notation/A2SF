import sys
import os
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

MODEL_CONFIGS = [
    {
        "name": "llama2",
        "path": "meta-llama/Llama-2-7b-chat-hf",
        "plot_dir": "plots/attention_map/llama2"
    },
    {
        "name": "llama3",
        "path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "plot_dir": "plots/attention_map/llama3"
    }
]

parser = argparse.ArgumentParser(description="Plot attention maps for specified layers")
parser.add_argument('--gpu', type=int, default=0, help="GPU device number to use")
parser.add_argument('--layers', type=int, nargs='+', default=[0, 16, 31], help="List of layer indices to plot")
parser.add_argument('--article_idx', type=int, default=0, help="Index of the article to process")
args = parser.parse_args()
device = f"cuda:{args.gpu}"

def plot_attention_map(attention_map, title, save_path):
    plt.figure(figsize=(12, 10))
    
    # Create the main plot with LogNorm
    im = plt.imshow(attention_map.cpu().numpy(), cmap='Blues', aspect='auto', norm=LogNorm())
    
    # Add colorbar with proper formatting
    cbar = plt.colorbar(im)
    cbar.set_label('Attention Score', fontsize=22)
    cbar.ax.tick_params(labelsize=20)
    
    # Set labels and title
    plt.xlabel('Key Token Position', fontsize=22)
    plt.ylabel('Query Token Position', fontsize=22)
    plt.title(title, fontsize=24, pad=20)
    
    # Set tick parameters
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_model(model_config):
    print(f"Processing model: {model_config['name']}")
    
    # Load dataset
    dataset_path = "datasets/cnn_dailymail-0shot.jsonl"
    with open(dataset_path, "r") as f:
        datalines = f.readlines()
        articles = []
        for dataline in datalines:
            articles.append(json.loads(dataline))
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config['path'])
    model = AutoModelForCausalLM.from_pretrained(model_config['path']).to(torch.float16).to(device)
    
    # Generate text and get attention maps
    with torch.inference_mode():
        input = f"[INST]{articles[args.article_idx]['article']}[/INST]"
        input_ids = tokenizer(input, return_tensors="pt").input_ids.to(device)
        outputs = model(input_ids, output_attentions=True)
    
    attention_maps = torch.stack(outputs.attentions).squeeze(1)  # [num_layers, num_heads, seq_len, seq_len]
    del model, tokenizer, outputs
    torch.cuda.empty_cache()
    
    # Create directory for plots
    os.makedirs(f"{model_config['plot_dir']}/attention", exist_ok=True)
    
    # Calculate and save overall averaged attention map (across all layers and heads)
    overall_averaged = attention_maps.mean(dim=(0, 1))  # [seq_len, seq_len]
    plot_attention_map(
        overall_averaged,
        'Overall Average Attention Map',
        f"{model_config['plot_dir']}/attention/overall_averaged.png"
    )
    
    # Calculate and save layer-wise averaged attention maps
    layer_averaged = attention_maps.mean(dim=1)  # [num_layers, seq_len, seq_len]
    for layer_idx in tqdm(args.layers, desc="Processing layers"):
        plot_attention_map(
            layer_averaged[layer_idx],
            f'Layer {layer_idx} Average Attention Map',
            f"{model_config['plot_dir']}/attention/layer{layer_idx}_averaged.png"
        )

# Execute the code
for model_config in MODEL_CONFIGS:
    os.makedirs(model_config['plot_dir'], exist_ok=True)
    process_model(model_config)
    torch.cuda.empty_cache()