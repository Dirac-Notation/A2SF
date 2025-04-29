import json
from transformers import AutoTokenizer
from tqdm import tqdm

def main():
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Open the dataset file
    with open("fewshot_data/needle_test_Llama-2-7b-hf.jsonl", "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    
    # Open output file
    with open("token_length_analysis.txt", "w", encoding="utf-8") as f:
        f.write("Sample Index | Target Tokens | Actual Tokens | Difference\n")
        f.write("-" * 60 + "\n")
        
        # Analyze each sample
        for i, sample in enumerate(tqdm(samples, desc="Analyzing token lengths")):
            prompt = sample["prompt"]
            target_tokens = sample["max_tokens"]
            
            # Get actual token length
            actual_tokens = len(tokenizer.encode(prompt))
            
            # Calculate difference
            difference = actual_tokens - target_tokens
            
            # Write to file
            f.write(f"{i:11d} | {target_tokens:12d} | {actual_tokens:12d} | {difference:+12d}\n")
            
            # Print progress every 100 samples
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} samples...")
    
    print("Analysis complete! Results saved to token_length_analysis.txt")

if __name__ == "__main__":
    main() 