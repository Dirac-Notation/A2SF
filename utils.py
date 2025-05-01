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

def get_prompt():
    prompt_list = [
        "The Schrödinger equation is a partial differential equation that governs the wave function of a non-relativistic quantum-mechanical system.",
        "Its discovery was a significant landmark in the development of quantum mechanics.",
        "It is named after Erwin Schrödinger, an Austrian physicist, who postulated the equation in 1925 and published it in 1926, forming the basis for the work that resulted in his Nobel Prize in Physics in 1933.",
        "Conceptually, the Schrödinger equation is the quantum counterpart of Newton's second law in classical mechanics.",
        "Given a set of known initial conditions, Newton's second law makes a mathematical prediction as to what path a given physical system will take over time.",
        "The Schrödinger equation gives the evolution over time of the wave function, the quantum-mechanical characterization of an isolated physical system.",
        "The equation was postulated by Schrödinger based on a postulate of Louis de Broglie that all matter has an associated matter wave.",
        "The equation predicted bound states of the atom in agreement with experimental observations.",
        "The Schrödinger equation is not the only way to study quantum mechanical systems and make predictions.",
        "Other formulations of quantum mechanics include matrix mechanics, introduced by Werner Heisenberg, and the path integral formulation, developed chiefly by Richard Feynman.",
        "When these approaches are compared, the use of the Schrödinger equation is sometimes called \"wave mechanics\".",
        "The equation given by Schrödinger is nonrelativistic because it contains a first derivative in time and a second derivative in space, and therefore space and time are not on equal footing.",
        "Paul Dirac incorporated special relativity and quantum mechanics into a single formulation that simplifies to the Schrödinger equation in the non-relativistic limit.",
        "This is the Dirac equation, which contains a single derivative in both space and time.",
        "Another partial differential equation, the Klein–Gordon equation, led to a problem with probability density even though it was a relativistic wave equation.",
        "The probability density could be negative, which is physically unviable.",
        "This was fixed by Dirac by taking the so-called square root of the Klein–Gordon operator and in turn introducing Dirac matrices.",
        "In a modern context, the Klein–Gordon equation describes spin-less particles, while the Dirac equation describes spin-1/2 particles."
    ]
    return " ".join(prompt_list)

def make_optimal_mask(attention_maps, prompt_length, select_budget):
    optimal_maps = attention_maps.clone()
    
    for i in range(attention_maps.size(2) - prompt_length):
        current_pos = prompt_length + i
        
        # Select top-k tokens
        selected_scores = optimal_maps[:,:,[current_pos],:].topk(k=2*select_budget, dim=3).indices
        
        # Create and apply mask
        mask = torch.zeros_like(optimal_maps[:,:,[current_pos],:], device=attention_maps.device)
        mask[:,:,:,current_pos:] = 1
        mask.scatter_(3, selected_scores, 1)
        
        # Apply mask and normalize
        optimal_maps[:,:,[current_pos],:] = optimal_maps[:,:,[current_pos],:] * mask
        divider = optimal_maps[:,:,current_pos,:].sum(dim=2, keepdim=True)
        optimal_maps[:,:,current_pos,:] = optimal_maps[:,:,current_pos,:] / divider
    
    return optimal_maps

def make_a2sf_mask(attention_maps, prompt_length, recent_budget, select_budget, forgetting_factor=1.00):
    a2sf_maps = attention_maps.clone()
    
    exponent = (forgetting_factor**torch.arange(prompt_length-1,-1,-1, device=attention_maps.device)).view(1,1,-1,1)
    scores = (a2sf_maps[:,:,:prompt_length,:] * exponent).sum(dim=2, keepdim=True)
    
    for i in range(attention_maps.size(2)-prompt_length-1):
        current_pos = prompt_length + i
        window_start = current_pos - recent_budget
        
        # Select top-k tokens within the window
        selected_scores = scores[:,:,:,:window_start].topk(k=select_budget, dim=3).indices
        
        # Create and apply mask
        mask = torch.zeros_like(scores, device=attention_maps.device)
        mask[:,:,:,window_start:] = 1
        mask.scatter_(3, selected_scores, 1)
        
        # Apply mask and normalize
        a2sf_maps[:,:,current_pos+1:,:] = a2sf_maps[:,:,current_pos+1:,:] * mask
        divider = a2sf_maps[:,:,current_pos+1,:].sum(dim=2, keepdim=True)
        a2sf_maps[:,:,current_pos+1,:] = a2sf_maps[:,:,current_pos+1,:] / divider
        
        scores = scores * mask
        scores = scores + a2sf_maps[:,:,current_pos+1,:].unsqueeze(2)
    
    return a2sf_maps