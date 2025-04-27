import json
import torch
from rouge_score import rouge_scorer

from tqdm import tqdm

def load_datasets(
    dataset_path: str,
    tokenizer
):
    with open(dataset_path, "r") as f:
        datalines = f.readlines()
        articles = []
        
        for dataline in datalines:
            articles.append(json.loads(dataline))

    prompts = []
    answers = []
    output_indices = []

    for data in tqdm(articles, desc="Tokenizing"):
        input = data["article"]
        answer = data["summary_gt"]
        
        input_ids = tokenizer(input, return_tensors="pt").input_ids
        output_ids = tokenizer(answer, return_tensors="pt").input_ids

        prompts.append(input_ids)
        answers.append(answer)
        output_indices.append(output_ids)
    
    num_input_ids = sum([prompt.numel() for prompt in prompts])/len(prompts)
    num_output_ids = sum([output_ids.numel() for output_ids in output_indices])/len(output_indices)
    
    print(f"Average input ids length : {num_input_ids:.2f}")
    print(f"Average output ids length : {num_output_ids:.2f}")
    
    return prompts, answers, output_indices

def sim(
    vec_1: torch.Tensor,
    vec_2: torch.Tensor,
    k: int = 20,
):
    if vec_1.dim() != 1 or vec_2.dim() != 1:
        assert "must dim 1"

    index_set = vec_2.topk(k).indices
    
    return (vec_1.index_select(dim=0, index=index_set)).sum()

def evaluate_model(
    model,
    tokenizer,
    prompts,
    answers,
    output_indices,
    device,
    desc="Generating",
    init_cache_fn=None,
    cache_params=None
):
    """
    Common evaluation function for both original and budget-based evaluation.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        prompts: List of input prompts
        answers: List of ground truth answers
        output_indices: List of output token indices
        device: Device to run evaluation on
        desc: Description for progress bar
        init_cache_fn: Optional function to initialize cache (for budget-based evaluation)
        cache_params: Optional parameters for cache initialization
    """
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    eos_token_id = tokenizer.eos_token_id

    predictions = []
    throughput_samples = []

    for idx, prompt in enumerate(tqdm(prompts, desc=desc)):
        input_ids = prompt.to(device)

        # Initialize cache if needed
        if init_cache_fn and cache_params:
            init_cache_fn(**cache_params)

        # GPU timing events
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

        # Generate
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=output_indices[idx].numel(),
            eos_token_id=eos_token_id,
            do_sample=False
        )

        # Record timing
        end_evt.record()
        torch.cuda.synchronize()
        elapsed = start_evt.elapsed_time(end_evt) / 1000.0  # ms → s
        toks = gen_ids.shape[1] - input_ids.shape[1]
        throughput_samples.append(toks / elapsed if elapsed > 0 else 0)

        # Decode
        pred_text = tokenizer.decode(
            gen_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True
        )
        predictions.append(pred_text)

    # Calculate ROUGE scores
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for ref, pred in zip(answers, predictions):
        score = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(score[key].fmeasure)

    # Calculate averages
    avg_r1 = sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1'])
    avg_r2 = sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2'])
    avg_rL = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])
    avg_tp = sum(throughput_samples) / len(throughput_samples)

    return {
        'predictions': predictions,
        'rouge1': avg_r1,
        'rouge2': avg_r2,
        'rougeL': avg_rL,
        'throughput': avg_tp
    }