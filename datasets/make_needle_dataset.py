import json
import os
import random
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
from datasets import load_dataset
import numpy as np

def find_similar_length_sentences(sentences_with_lengths, target_length, tolerance=10):
    """Find sentences with length within tolerance of target_length.
    If no suitable sentence is found, recursively split the target length and combine sentences."""
    
    # First try to find a single sentence that matches
    suitable_sentences = []
    for sentence, length in sentences_with_lengths:
        if abs(length - target_length) <= tolerance:
            suitable_sentences.append(sentence)
    
    if suitable_sentences:
        return random.choice(suitable_sentences)
    
    # If no suitable sentence found, combine multiple 10-token sentences
    num_sentences_needed = (target_length + 9) // 10  # Round up division
    selected_sentences = []
    
    # Get all sentences with exactly 10 tokens
    ten_token_sentences = [s for s, l in sentences_with_lengths if l == 10]
    
    if not ten_token_sentences:
        return None
    
    # Randomly select the required number of sentences
    selected_sentences = random.sample(ten_token_sentences, num_sentences_needed)
    
    # Combine the sentences
    return " ".join(selected_sentences)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Name of the model to use (e.g., 'meta-llama/Llama-2-7b-hf')")
    args = parser.parse_args()
    
    # Create fewshot_data directory if it doesn't exist
    os.makedirs("datasets", exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Define the needle text (password-like format)
    needle = "Password is 12345. "
    
    # Define user prompt
    user_prompt = "Find the password in the above text. The password"

    # Load dataset and tokenize all sentences
    print("Loading and tokenizing dataset...")
    dataset = load_dataset("bookcorpus", split="train", streaming=True)
    
    # Pre-fetch and tokenize sentences
    sentences_with_lengths = []
    dataset_iterator = iter(dataset)
    for _ in tqdm(range(100000)):  # Fetch 100,000 sentences
        try:
            text = next(dataset_iterator)["text"]
            # Split into sentences (simple split by period)
            sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
            for sentence in sentences:
                length = len(tokenizer.encode(sentence))
                if length == 10:  # Only keep sentences with exactly 10 tokens
                    sentences_with_lengths.append((sentence, length))
        except StopIteration:
            break
    
    print(f"Found {len(sentences_with_lengths)} sentences with exactly 10 tokens")
    
    # Calculate token lengths of prompts and needle
    user_tokens = tokenizer.encode(user_prompt)
    needle_tokens = tokenizer.encode(needle)
    prompt_tokens_length = len(user_tokens) + len(needle_tokens)
    
    # Define target total token lengths and positions
    target_lengths = list(range(200, 4001, 200))  # 200 to 4000 in 200-token increments
    positions = list(range(5, 105, 5))  # 5% to 100% in 5% increments
    
    # Extract model name for filename
    model_name_for_file = args.model_name.split('/')[-1]
    
    # Create a single output file for all samples
    output_file = f"datasets/needle_test_{model_name_for_file}.jsonl"
    
    # Generate samples for all context lengths
    all_samples = []
    
    for target_total_tokens in tqdm(target_lengths, desc="Generating samples for different lengths"):
        # Calculate available space for context
        available_context_tokens = target_total_tokens - prompt_tokens_length
        
        # Generate 5 samples for each position
        for position in positions:
            for _ in range(5):  # 5 samples per position
                # Calculate required lengths before and after needle
                before_length = int((position / 100) * available_context_tokens)
                after_length = available_context_tokens - before_length
                
                # Find suitable sentences
                before_sentence = find_similar_length_sentences(sentences_with_lengths, before_length)
                after_sentence = find_similar_length_sentences(sentences_with_lengths, after_length)
                
                if before_sentence is None or after_sentence is None:
                    print(f"Warning: Could not find suitable sentences for length {target_total_tokens} at position {position}")
                    continue
                
                # Combine all parts
                full_text = f"{before_sentence} {needle} {after_sentence}\n\n{user_prompt}"
                
                # Create the sample dictionary
                sample = {
                    "prompt": full_text,
                    "needle": needle,
                    "needle_position": position,
                    "max_tokens": target_total_tokens,
                    "answer": "12345"  # The expected answer is just the password
                }
                
                all_samples.append(sample)
    
    # Save all samples to a single JSONL file
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(all_samples)} samples to {output_file}")

if __name__ == "__main__":
    main()
