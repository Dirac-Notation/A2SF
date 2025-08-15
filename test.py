import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import json
import torch
import glob

from pandas.core.window.rolling import Window
from transformers import AutoTokenizer, AutoModelForCausalLM

from a2sf_search import make_sentence_exp, get_punctuation_ids

MAX_LENGTH = 1000

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Get all jsonl files in result_txt/longbench
jsonl_files = glob.glob("result_txt/longbench/*.jsonl")

# List to store selected sentences and track which datasets were used
selected_sentences = []
used_datasets = []

for file_path in jsonl_files:
    # Get filename without path and extension
    filename = os.path.basename(file_path).split('.')[0]
    
    # Read jsonl file
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            input_prompt = data['input_prompt']
            
            # Tokenize and get length
            tokens = tokenizer(input_prompt, return_tensors='pt')
            length = len(tokens.input_ids[0])
            
            # Check if length is between 1000-2000
            if 1000 <= length <= 2000:
                selected_sentences.append({
                    'dataset': filename,
                    'input_prompt': input_prompt,
                    'token_length': length
                })
                used_datasets.append(filename)
                print(f"Selected sentence from {filename} with length {length}")
                break  # Only take one sentence per dataset

# Process selected sentences to get attention maps
print(f"\n=== Processing selected sentences ===")
for sentence_data in selected_sentences:
    input_prompt = sentence_data['input_prompt']
    original_length = sentence_data['token_length']
    
    # Tokenize the input
    tokens = tokenizer(input_prompt, return_tensors='pt')
    
    # Take first 500 and last 500 tokens to make it exactly 1000 tokens
    first_500 = tokens.input_ids[0][:500]
    last_500 = tokens.input_ids[0][-500:]
    combined_tokens = torch.cat([first_500, last_500]).unsqueeze(0).to(model.device)
    
    # Get attention map from model
    with torch.no_grad():
        outputs = model(combined_tokens, output_attentions=True)
        attention_maps = outputs.attentions  # This contains attention maps from all layers
    
    attention_maps = torch.stack(attention_maps, dim=0)
    
    h2o_scores = attention_maps.sum(dim=3)
    h2o_selected_tokens = h2o_scores[:,:,:,:-50].topk(50, dim=3).indices.to(torch.float16)
    
    snapkv_scores = attention_maps[:,:,:,-16:,:].sum(dim=3)
    snapkv_selected_tokens = snapkv_scores[:,:,:,:-50].topk(50, dim=3).indices.to(torch.float16)
    
    import pdb; pdb.set_trace()

# Print summary of used datasets
print(f"\n=== Summary ===")
print(f"Total datasets with sentences in 1000-2000 token range: {len(used_datasets)}")
print(f"Used datasets: {used_datasets}")

# Print selected sentences info
if selected_sentences:
    print(f"\nSelected {len(selected_sentences)} sentences:")
    for sentence_data in selected_sentences:
        print(f"- {sentence_data['dataset']}: original {sentence_data['token_length']} tokens, processed to {sentence_data['processed_token_length']} tokens")
        print(f"  Attention maps shape: {len(sentence_data['attention_maps'])} layers")
else:
    print("\nNo sentences found in the 1000-2000 token range.")