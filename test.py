import torch
import json

from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm

from utils import load_datasets
from utils_real_drop.kv_llama import LlamaForCausalLM

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).half().to("cuda:2")

prompts, answers, output_indices = load_datasets(dataset_path="fewshot_data/cnn_dailymail-3shot.jsonl", tokenizer=tokenizer)
predictions = []

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
eos_token_id = tokenizer.eos_token_id

for idx, prompt in enumerate(tqdm(prompts)):
    model.init_cache(
        use_compression=True,
        select_budget=100,
        recent_budget=100,
        forgetting_factor=0.0
    )

    input_ids = prompt.to(model.device)

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=output_indices[idx].numel()+10,
        eos_token_id=eos_token_id,
        do_sample=False
    )

    predictions.append(tokenizer.decode(generated_ids[0,input_ids.numel():], skip_special_tokens=True))

rouge_scores = {
    'rouge1': [],
    'rouge2': [],
    'rougeL': []
}

with open("results.jsonl", "w") as f:
    for i in range(len(predictions)):
        prediction = predictions[i]
        reference = answers[i]

        score = scorer.score(reference, prediction)
        
        json.dump({"output": prediction}, f)
        f.write("\n")
        
        rouge_scores['rouge1'].append(score['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(score['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(score['rougeL'].fmeasure)

avg_rouge1 = sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1'])
avg_rouge2 = sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2'])
avg_rougeL = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])

print(f"Average ROUGE-1: {avg_rouge1:.4f}")
print(f"Average ROUGE-2: {avg_rouge2:.4f}")
print(f"Average ROUGE-L: {avg_rougeL:.4f}")