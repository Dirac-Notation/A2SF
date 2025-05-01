import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_dataset(file_path):
    """Load the needle-in-haystack dataset from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def evaluate_model(model, tokenizer, dataset, device):
    """Evaluate the model on the needle-in-haystack task."""
    results = defaultdict(list)
    
    for sample in tqdm(dataset, desc="Evaluating samples"):
        # Get the prompt and expected answer
        prompt = sample["prompt"]
        expected_answer = sample["answer"]
        needle_position = sample["needle_position"]
        total_tokens = sample["total_tokens"]
        
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.0,
                do_sample=False,
            )

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the model's answer (everything after the prompt)
        model_answer = response[len(prompt):].strip()

        # Check if the model found the correct password
        is_correct = expected_answer in model_answer
        
        # Record the result
        results[(total_tokens, needle_position)].append({
            "expected": expected_answer,
            "model_answer": model_answer,
            "is_correct": is_correct
        })
    
    return results

def calculate_metrics(results):
    """Calculate accuracy metrics for each needle position and context length."""
    metrics = {}
    
    for (total_tokens, position), samples in results.items():
        correct_count = sum(1 for sample in samples if sample["is_correct"])
        total_count = len(samples)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        metrics[(total_tokens, position)] = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count
        }
    
    return metrics

def create_heatmap(metrics, output_file):
    """Create a heatmap visualization of the results."""
    # Extract unique context lengths and positions
    context_lengths = sorted(list(set(k[0] for k in metrics.keys())))
    positions = sorted(list(set(k[1] for k in metrics.keys())))
    
    # Create a 2D array for the heatmap
    heatmap_data = np.zeros((len(positions), len(context_lengths)))
    
    # Fill the heatmap data
    for i, position in enumerate(positions):
        for j, length in enumerate(context_lengths):
            if (length, position) in metrics:
                heatmap_data[i, j] = metrics[(length, position)]["accuracy"]
    
    # Create the heatmap using matplotlib
    plt.figure(figsize=(12, 8))
    
    # Create the heatmap
    im = plt.imshow(heatmap_data, cmap='RdYlGn', vmin=0.0, vmax=1.0)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Accuracy')
    
    # Set axis labels and title
    plt.xlabel('Context Length (tokens)')
    plt.ylabel('Needle Position (%)')
    plt.title('Needle-in-Haystack Performance (Original)')
    
    # Set tick labels
    plt.xticks(np.arange(len(context_lengths)), context_lengths)
    plt.yticks(np.arange(len(positions)), positions)
    
    # Add text annotations
    for i in range(len(positions)):
        for j in range(len(context_lengths)):
            if (context_lengths[j], positions[i]) in metrics:
                text = plt.text(j, i, f"{heatmap_data[i, j]:.2f}", 
                               ha="center", va="center", color="black")
    
    plt.tight_layout()
    
    # Save the heatmap
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf", 
                        help="Name of the model to use")
    parser.add_argument("--dataset_path", type=str, default="datasets/needle_dataset_Llama-2-7b-hf.jsonl",
                        help="Path to the dataset file")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU number to use (default: 0)")
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} samples")
    
    # Evaluate model
    print("Starting evaluation...")
    results = evaluate_model(model, tokenizer, dataset, device)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Create heatmap
    os.makedirs("plots", exist_ok=True)
    output_file = "plots/needle_heatmap_original.png"
    print(f"Creating heatmap visualization: {output_file}")
    create_heatmap(metrics, output_file)
    
    # Print summary
    print("\nSummary of results:")
    print("Position | Accuracy | Correct/Total")
    print("-" * 40)
    for position in sorted(set(k[1] for k in metrics.keys())):
        # Calculate average accuracy across all context lengths for this position
        position_samples = [(k[0], v) for k, v in metrics.items() if k[1] == position]
        total_correct = sum(sample[1]["correct_count"] for sample in position_samples)
        total_samples = sum(sample[1]["total_count"] for sample in position_samples)
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
        print(f"{position:3.2f} | {avg_accuracy:.2%} | {total_correct}/{total_samples}")

if __name__ == "__main__":
    main()
