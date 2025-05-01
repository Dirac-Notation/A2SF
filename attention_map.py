import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import get_prompt, make_optimal_mask, make_a2sf_mask

GENERATION_LENGTH = 800
LOCAL_WINDOW_SIZE = 100
NUM_SELECTED_TOKENS = 100
MODEL_CONFIGS = [
    {
        "name": "llama2-chat",
        "path": "meta-llama/Llama-2-7b-chat-hf",
        "plot_dir": "plots/llama2-chat"
    },
    {
        "name": "opt",
        "path": "facebook/opt-6.7b",
        "plot_dir": "plots/opt"
    },
    {
        "name": "llama",
        "path": "huggyllama/llama-7b",
        "plot_dir": "plots/llama"
    }
]

parser = argparse.ArgumentParser(description="Model generation with RoPE cache settings via CLI")
parser.add_argument('--gpu', type=str, default="1", help="GPU device number to use")
args = parser.parse_args()
device = f"cuda:{args.gpu}"

def plot_first_token_attention(attention_maps, scores_h2o, scores_a2sf, prompt_length, save_dir):
    num_tokens_to_show = 200
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))
    
    ax1.bar(range(num_tokens_to_show), attention_maps[0,0,prompt_length,:num_tokens_to_show].cpu().numpy())
    ax1.set_title('Original Attention Distribution\nFirst Generated Token')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Attention Score')
    
    ax2.bar(range(num_tokens_to_show), scores_h2o[0,0,0,:num_tokens_to_show].cpu().numpy())
    ax2.set_title('H2O Attention Distribution\nFirst Generated Token')
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Accumulative\nAttention Weight')
    
    ax3.bar(range(num_tokens_to_show), scores_a2sf[0,0,0,:num_tokens_to_show].cpu().numpy())
    ax3.set_title('A2SF Attention Distribution\nFirst Generated Token')
    ax3.set_xlabel('Token Position')
    ax3.set_ylabel('Accumulative\nAttention Weight')
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(range(0, num_tokens_to_show+1, 20))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/first_token_attention_distribution.png', bbox_inches='tight')
    plt.close()

def plot_attention_maps(original_map, h2o_map, a2sf_map, layer_idx, head_idx, save_path):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
    
    original_map = torch.pow(original_map, 1/3)
    h2o_map = torch.pow(h2o_map, 1/3)
    a2sf_map = torch.pow(a2sf_map, 1/3)
    
    im1 = ax1.imshow(original_map.cpu().numpy(), cmap='Blues', vmin=0.0, vmax=1.0)
    ax1.set_title(f'Original Attention Map\nLayer {layer_idx}, Head {head_idx}')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Generation Step')
    
    im2 = ax2.imshow(h2o_map.cpu().numpy(), cmap='Blues', vmin=0.0, vmax=1.0)
    ax2.set_title(f'H2O Attention Map\nLayer {layer_idx}, Head {head_idx}')
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Generation Step')
    
    im3 = ax3.imshow(a2sf_map.cpu().numpy(), cmap='Blues', vmin=0.0, vmax=1.0)
    ax3.set_title(f'A2SF Attention Map\nLayer {layer_idx}, Head {head_idx}')
    ax3.set_xlabel('Token Position')
    ax3.set_ylabel('Generation Step')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_attention_heatmap_comparison(answer, h2o, a2sf, save_dir):
    heatmap_h2o = ((answer * h2o).sum(dim=3)/answer.norm(dim=3)/h2o.norm(dim=3)).mean(dim=2)
    heatmap_a2sf = ((answer * a2sf).sum(dim=3)/answer.norm(dim=3)/a2sf.norm(dim=3)).mean(dim=2)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot H2O heatmap
    im1 = ax1.imshow(heatmap_h2o.cpu().to(torch.float32).numpy(), cmap='Blues_r', vmin=0.0, vmax=1.0)
    ax1.set_title('H2O Attention Heatmap')
    ax1.set_xlabel('Head')
    ax1.set_ylabel('Layer')
    plt.colorbar(im1, ax=ax1)
    
    # Plot A2SF heatmap
    im2 = ax2.imshow(heatmap_a2sf.cpu().to(torch.float32).numpy(), cmap='Blues_r', vmin=0.0, vmax=1.0)
    ax2.set_title('A2SF Attention Heatmap')
    ax2.set_xlabel('Head')
    ax2.set_ylabel('Layer')
    plt.colorbar(im2, ax=ax2)
    
    # Plot difference heatmap (A2SF - H2O)
    difference = heatmap_a2sf.cpu().to(torch.float32).numpy() - heatmap_h2o.cpu().to(torch.float32).numpy()
    im3 = ax3.imshow(difference, cmap='RdBu', vmin=-0.15, vmax=0.15)
    ax3.set_title('Difference (A2SF - H2O)\nBlue: A2SF better, Red: H2O better')
    ax3.set_xlabel('Head')
    ax3.set_ylabel('Layer')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/attention_heatmap_comparison.png')
    plt.close()

def plot_layer_similarities(h2o_similarity, a2sf_similarity, optimal_similarity, num_layers, save_dir):
    h2o_layer_means = h2o_similarity.mean(axis=1)
    a2sf_layer_means = a2sf_similarity.mean(axis=1)
    optimal_layer_means = optimal_similarity.mean(axis=1)
    
    plt.figure(figsize=(10, 6))
    x = range(num_layers)
    width = 0.25
    
    plt.bar([i - width for i in x], h2o_layer_means, width, label='H2O', color='blue', alpha=0.7)
    plt.bar([i for i in x], a2sf_layer_means, width, label='A2SF', color='orange', alpha=0.7)
    plt.bar([i + width for i in x], optimal_layer_means, width, label='Optimal', color='green', alpha=0.7)
    
    plt.xlabel('Layer')
    plt.ylabel('Average Cosine Similarity')
    plt.title('Layer-wise Average Attention Map Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.65, 1.02)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/layer_wise_average_similarity.png')
    plt.close()

def plot_layer_similarities_over_steps(h2o_similarity, a2sf_similarity, optimal_similarity, save_dir):
    """
    Plot similarity over generation steps for each layer as subplots.
    
    Args:
        h2o_similarity: Array of shape [num_layers, num_steps]
        a2sf_similarity: Array of shape [num_layers, num_steps]
        optimal_similarity: Array of shape [num_layers, num_steps]
        save_dir: Directory to save the plot
    """
    num_layers = h2o_similarity.shape[0]
    num_steps = h2o_similarity.shape[1]
    
    # Calculate overall average for each layer
    h2o_overall_avg = np.mean(h2o_similarity, axis=1)  # [num_layers]
    a2sf_overall_avg = np.mean(a2sf_similarity, axis=1)  # [num_layers]
    optimal_overall_avg = np.mean(optimal_similarity, axis=1)  # [num_layers]
    
    # Calculate grid dimensions for subplots
    grid_size = int(np.ceil(np.sqrt(num_layers)))
    
    # Create figure with subplots and higher DPI
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(24, 24))
    axes = axes.flatten()
    
    # Plot each layer in its own subplot
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        
        # Plot step-by-step similarity
        ax.plot(h2o_similarity[layer_idx], color='blue', label='H2O', alpha=0.7, linewidth=1.5)
        ax.plot(a2sf_similarity[layer_idx], color='orange', label='A2SF', alpha=0.7, linewidth=1.5)
        ax.plot(optimal_similarity[layer_idx], color='green', label='Optimal', alpha=0.7, linewidth=1.5)
        
        # Plot overall average as dashed lines without adding to legend
        ax.axhline(y=h2o_overall_avg[layer_idx], color='blue', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=a2sf_overall_avg[layer_idx], color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=optimal_overall_avg[layer_idx], color='green', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add average values as text
        ax.text(0.02, 0.98, f'H2O Avg: {h2o_overall_avg[layer_idx]:.3f}', 
                transform=ax.transAxes, color='blue', verticalalignment='top')
        ax.text(0.02, 0.94, f'A2SF Avg: {a2sf_overall_avg[layer_idx]:.3f}', 
                transform=ax.transAxes, color='orange', verticalalignment='top')
        ax.text(0.02, 0.90, f'Optimal Avg: {optimal_overall_avg[layer_idx]:.3f}', 
                transform=ax.transAxes, color='green', verticalalignment='top')
        
        ax.set_title(f'Layer {layer_idx}', fontsize=12, pad=10)
        ax.set_xlabel('Generation Step', fontsize=10)
        ax.set_ylabel('Average Cosine Similarity', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        # Dynamically adjust y-axis limits to the data range for this layer
        combined = np.concatenate([
            h2o_similarity[layer_idx],
            a2sf_similarity[layer_idx],
            optimal_similarity[layer_idx]
        ])
        y_min, y_max = combined.min(), combined.max()
        margin = (y_max - y_min) * 0.1 if (y_max > y_min) else 0.05
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Hide empty subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/layer_similarities_over_steps.png', bbox_inches='tight', dpi=300)
    plt.close()

def process_model(model_config):
    print(f"Processing model: {model_config['name']}")
    
    prompt = get_prompt()
    
    tokenizer = AutoTokenizer.from_pretrained(model_config['path'])
    model = AutoModelForCausalLM.from_pretrained(model_config['path']).to(torch.float16).to(device)
    
    with torch.inference_mode():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        past_key_values = None
        
        outputs = model(input_ids, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        
        for i in tqdm(range(GENERATION_LENGTH), desc="Token generation"):
            next_token_scores = next_token_logits
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=1)
            
            outputs = model(next_tokens.unsqueeze(-1), past_key_values=past_key_values, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
    
        outputs = model(input_ids, output_attentions=True)
    
    attention_maps = torch.stack(outputs.attentions).squeeze(1) #.to(torch.float32)
    del model, tokenizer, outputs, past_key_values
    torch.cuda.empty_cache()
    
    prompt_length = input_ids.size(1) - GENERATION_LENGTH
    optimal_attention_maps = make_optimal_mask(attention_maps, prompt_length, NUM_SELECTED_TOKENS)
    h2o_attention_maps = make_a2sf_mask(attention_maps, prompt_length, LOCAL_WINDOW_SIZE, NUM_SELECTED_TOKENS, forgetting_factor=1.00)
    a2sf_attention_maps = make_a2sf_mask(attention_maps, prompt_length, LOCAL_WINDOW_SIZE, NUM_SELECTED_TOKENS, forgetting_factor=0.99)

    # plot_first_token_attention(attention_maps, h2o_attention_maps, a2sf_attention_maps, prompt_length, model_config['plot_dir'])
    
    answer = attention_maps[:,:,-GENERATION_LENGTH:,:]
    optimal = optimal_attention_maps[:,:,-GENERATION_LENGTH:,:]
    h2o = h2o_attention_maps[:,:,-GENERATION_LENGTH:,:]
    a2sf = a2sf_attention_maps[:,:,-GENERATION_LENGTH:,:]
    
    optimal_similarity = (answer * optimal).sum(dim=3) / (answer.norm(dim=3) * optimal.norm(dim=3))
    h2o_similarity = (answer * h2o).sum(dim=3) / (answer.norm(dim=3) * h2o.norm(dim=3))
    a2sf_similarity = (answer * a2sf).sum(dim=3) / (answer.norm(dim=3) * a2sf.norm(dim=3))
    
    optimal_similarity = optimal_similarity.mean(dim=1).cpu().numpy()
    h2o_similarity = h2o_similarity.mean(dim=1).cpu().numpy()
    a2sf_similarity = a2sf_similarity.mean(dim=1).cpu().numpy()
    
    os.makedirs(f"{model_config['plot_dir']}/attention", exist_ok=True)
    selected_layers = [0, 16, 31]
    
    plot_attention_heatmap_comparison(answer, h2o, a2sf, model_config['plot_dir'])
    
    # Add the new plot for layer similarities over steps
    plot_layer_similarities_over_steps(h2o_similarity, a2sf_similarity, optimal_similarity, model_config['plot_dir'])
    
    for layer_idx in tqdm(selected_layers, desc="Processing layers"):
        os.makedirs(f"{model_config['plot_dir']}/attention/layer{layer_idx}", exist_ok=True)
        for head_idx in range(attention_maps.size(1)):
            save_path = f"{model_config['plot_dir']}/attention/layer{layer_idx}/head{head_idx}.png"
            plot_attention_maps(
                optimal_attention_maps[layer_idx, head_idx, -GENERATION_LENGTH:, :],
                h2o_attention_maps[layer_idx, head_idx, -GENERATION_LENGTH:, :],
                a2sf_attention_maps[layer_idx, head_idx, -GENERATION_LENGTH:, :],
                layer_idx, head_idx, save_path
            )
    
    plot_layer_similarities(h2o_similarity, a2sf_similarity, optimal_similarity, len(h2o_similarity), model_config['plot_dir'])

# Execute the code sequentially
for model_config in MODEL_CONFIGS:
    os.makedirs(model_config['plot_dir'], exist_ok=True)
    process_model(model_config)