import json
import os
import random
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Name of the model to use (e.g., 'meta-llama/Llama-2-7b-hf')")
    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs("datasets", exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Define user prompt
    user_prompt = "Identify the password hidden in the text above.\nPassword: "

    # Collect sentences by token length
    print("Loading dataset and collecting sentences...")
    dataset = load_dataset("bookcorpus", split="train", streaming=True)
    sentences_by_length = {length: [] for length in range(8, 13)}
    for example in tqdm(dataset, desc="Collecting sentences"):
        text = example["text"]
        sentences = [s.strip() + " ." for s in text.split(".") if s.strip()]
        for s in sentences:
            token_len = len(tokenizer.encode(s, add_special_tokens=False))
            if 8 <= token_len <= 12 and len(sentences_by_length[token_len]) < 1000:
                sentences_by_length[token_len].append(s)
        if all(len(lst) >= 1000 for lst in sentences_by_length.values()):
            break
    counts = ", ".join(f"{l}:{len(lst)}" for l, lst in sentences_by_length.items())
    print(f"Collected sentences counts: {counts}")

    # Flatten list of sentences
    all_sentences = [s for lst in sentences_by_length.values() for s in lst]

    # Define total token lengths and needle positions
    target_lengths = list(range(200, 3801, 200))
    positions = [i / 100 for i in range(5, 101, 5)]

    # Prepare output file
    model_file_name = args.model_name.split('/')[-1]
    output_file = f"datasets/needle_dataset_{model_file_name}.jsonl"

    all_samples = []
    for total_tokens in tqdm(target_lengths, desc="Generating samples"):
        # approximate number of sentences
        total_sentences = total_tokens // 10
        for position in positions:
            num_before = int(total_sentences * position)
            num_after = total_sentences - num_before
            for _ in range(5):  # 5 samples per position
                # Generate a new random 5-digit number for each sample
                random_password = str(random.randint(10000, 99999))
                needle = f"Password is {random_password} ."
                before = random.sample(all_sentences, num_before) if num_before > 0 else []
                after = random.sample(all_sentences, num_after) if num_after > 0 else []
                context = " ".join(before + [needle] + after)
                prompt = f"{context}\n{user_prompt}"
                sample = {
                    "prompt": prompt,
                    "answer": random_password,
                    "total_tokens": total_tokens,
                    "needle_position": position
                }
                all_samples.append(sample)

    # Write to JSONL file
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_samples)} samples to {output_file}")

    # Analyze token lengths and save to file
    analysis_file = f"datasets/token_length_analysis_{model_file_name}.txt"
    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("Sample Index | Target Tokens | Actual Tokens | Difference\n")
        f.write("-" * 60 + "\n")
        for i, sample in enumerate(tqdm(all_samples, desc="Analyzing token lengths")):
            prompt = sample["prompt"]
            target_tokens = sample["total_tokens"]
            actual_tokens = len(tokenizer.encode(prompt))
            difference = actual_tokens - target_tokens
            f.write(f"{i:11d} | {target_tokens:12d} | {actual_tokens:12d} | {difference:+12d}\n")
    print(f"Token length analysis saved to {analysis_file}")


if __name__ == "__main__":
    main()
