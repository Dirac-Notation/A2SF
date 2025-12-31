import os
import json
import torch
from tqdm import tqdm
import argparse
from typing import List

from utils import load_model, set_seed, CompressionConfig

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument('--model', type=str, required=True, choices=["llama", "llama2", "llama3", "opt"])
    parser.add_argument('--budget', type=int, default=128, help="Total budget for KV cache compression")
    parser.add_argument('--rbo_p', type=float, default=0.95, help="RBO persistence parameter")
    parser.add_argument('--data_path', type=str, default="datasets/training_data.jsonl", help="Path to training data")
    return parser.parse_args(args)

def load_training_data(data_path: str) -> List[dict]:
    """Load training data from JSONL file"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def calculate_rbo(list1: List[int], list2: List[int], p: float) -> float:
    """
    두 리스트 간의 Rank-Biased Overlap (RBO)를 계산합니다.
    list1, list2: 순위가 매겨진 요소들의 리스트 (앞쪽일수록 중요도 높음)
    p: persistence parameter (0 < p < 1)
    """
    # 비교 깊이 설정 (더 긴 리스트 기준)
    k = max(len(list1), len(list2))
    
    rbo_score = 0.0
    weight = 1.0
    
    # 누적된 요소를 추적하기 위한 집합
    seen1 = set()
    seen2 = set()
    
    for d in range(1, k + 1):
        # d번째 순위(인덱스 d-1)의 요소 가져오기
        item1 = list1[d-1] if d-1 < len(list1) else None
        item2 = list2[d-1] if d-1 < len(list2) else None
        
        if item1 is not None: seen1.add(item1)
        if item2 is not None: seen2.add(item2)
        
        # 현재 깊이 d까지의 교집합 개수 계산 (Agreement)
        # RBO의 핵심: 단순히 현재 위치가 같은지가 아니라, 현재 깊이까지의 집합이 얼마나 겹치는지 확인
        current_overlap = len(seen1.intersection(seen2))
        agreement = current_overlap / d
        
        # 가중치 적용하여 점수 합산
        rbo_score += agreement * weight
        weight *= p
    
    # 정규화 (extrapolated RBO가 아닌 표준 수식 사용)
    return rbo_score * (1 - p)

def compute_accuracy_score(model, selected_indices: List[List[int]], context_length: int, rbo_p: float) -> float:
    """
    모델이 선택한 인덱스와 정답 인덱스를 비교하여 RBO 기반 정확도 점수를 계산합니다.
    """
    similarity_score = 0.0
    
    num_layers = len(model.model.layers)
    num_heads = model.config.num_attention_heads

    for layer_idx, layer in enumerate(model.model.layers):
        # 모델이 선택한 인덱스 가져오기
        model_selected_indices_tensor = layer.self_attn.past_key_value.selected_indices
        if model_selected_indices_tensor is None:
            # 선택된 인덱스가 없으면 (아직 압축이 일어나지 않았으면) 스킵
            continue
        
        model_selected_indices = model_selected_indices_tensor.squeeze(0).cpu()
        # 정답 인덱스 가져오기
        answer_selected_indices = torch.tensor(selected_indices[layer_idx])
        
        # KV Heads 확장을 위한 처리
        num_key_value_heads = layer.self_attn.num_key_value_groups
        model_selected_indices = model_selected_indices.unsqueeze(1).expand(-1, num_key_value_heads, -1).reshape(answer_selected_indices.size(0), -1)
        
        for head_idx in range(num_heads):
            # 모델 리스트 구성: 선택된 인덱스
            model_list = model_selected_indices[head_idx].tolist()
            
            # 정답 리스트 구성
            answer_list = answer_selected_indices[head_idx].tolist()
            
            # RBO 계산
            similarity_score += calculate_rbo(model_list, answer_list, rbo_p)
    
    similarity_score /= (num_layers * num_heads)
    
    return similarity_score

def prepare_prompt(prompt: str, dataset: str, tokenizer, max_length: int) -> str:
    """Prepare prompt for model input"""
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    
    if len(tokenized_prompt) > max_length:
        half = int(max_length / 2)
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    
    if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        if "llama" in tokenizer.name_or_path.lower() or "llama" in str(type(tokenizer)).lower():
            prompt = f"[INST]{prompt}[/INST]"
    
    return prompt

def process_with_observation_window(model, tokenizer, training_data, max_length, budget, num_layers, observation_window, rbo_p):
    """Process all samples with a specific observation window and return RBO scores"""
    # Create SnapKV compression config
    config = CompressionConfig()
    config.compression_method = "snap"
    config.total_budget = budget
    config.layerwise_ratios = [1.0 for i in range(num_layers)]
    config.local_ratios = 0.125
    config.observation_window = observation_window
    config.method = "snap"
    
    rbo_scores = []
    
    for sample in training_data:
        prompt = sample["input_prompt"]
        selected_indices = sample["selected_indices"]  # Ground truth indices
        dataset = sample.get("dataset", None)
        
        # Prepare prompt
        prepared_prompt = prepare_prompt(prompt, dataset, tokenizer, max_length)
        
        # Tokenize input
        input_tensor = tokenizer(prepared_prompt, truncation=False, return_tensors="pt")
        input_ids = input_tensor.input_ids.to(model.device)
        context_length = input_ids.size(1)
        
        # Initialize cache with SnapKV
        model.init_cache(config)
        
        # Run model forward pass
        with torch.no_grad():
            model(input_ids)
        
        # Compute RBO score
        rbo_score = compute_accuracy_score(model, selected_indices, context_length, rbo_p)
        rbo_scores.append(rbo_score)
    
    return rbo_scores

def main():
    set_seed(42)
    args = parse_args()
    
    # Observation window values to test
    observation_windows = [1, 2, 4, 8, 16, 32, 64, 128, 512, 1024, 4096, 8192]
    
    # Set GPU environment
    gpus = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    
    # Load model configuration
    model_name = args.model
    model_name = model_name.split("_")[0].lower()
    
    # Load max length
    with open("config/model2maxlen.json", "r") as f:
        model2maxlen = json.load(f)
    max_length = model2maxlen[model_name]
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, args.gpus)
    print("Model loaded successfully!")
    
    # Get number of layers from model config
    num_layers = model.config.num_hidden_layers
    print(f"Number of layers: {num_layers}")
    
    # Load training data
    print(f"Loading training data from: {args.data_path}")
    training_data = load_training_data(args.data_path)
    print(f"Loaded {len(training_data)} training samples")
    
    print(f"\nProcessing samples with SnapKV:")
    print(f"  Budget: {args.budget}")
    print(f"  Observation windows: {observation_windows}")
    print(f"  RBO p: {args.rbo_p}")
    print("=" * 80)
    
    # Process each observation window
    results = {}
    for obs_window in tqdm(observation_windows, desc="Processing observation windows"):
        rbo_scores = process_with_observation_window(
            model, tokenizer, training_data, max_length, 
            args.budget, num_layers, obs_window, args.rbo_p
        )
        
        avg_rbo = sum(rbo_scores) / len(rbo_scores) if rbo_scores else 0.0
        min_rbo = min(rbo_scores) if rbo_scores else 0.0
        max_rbo = max(rbo_scores) if rbo_scores else 0.0
        
        results[obs_window] = {
            "avg_rbo": avg_rbo,
            "min_rbo": min_rbo,
            "max_rbo": max_rbo,
            "all_scores": rbo_scores
        }
    
    # Print results summary
    print("\n" + "=" * 80)
    print("Results Summary:")
    print("=" * 80)
    print(f"{'Observation Window':<25} {'Average RBO':<15} {'Min RBO':<15} {'Max RBO':<15}")
    print("-" * 80)
    
    for obs_window in observation_windows:
        result = results[obs_window]
        print(f"{obs_window:<25} {result['avg_rbo']:<15.4f} {result['min_rbo']:<15.4f} {result['max_rbo']:<15.4f}")
    
    print("=" * 80)
    
    # Find best observation window
    best_obs_window = max(observation_windows, key=lambda x: results[x]['avg_rbo'])
    best_result = results[best_obs_window]
    
    print(f"\nBest Observation Window: {best_obs_window}")
    print(f"  Average RBO: {best_result['avg_rbo']:.4f}")
    print(f"  Min RBO: {best_result['min_rbo']:.4f}")
    print(f"  Max RBO: {best_result['max_rbo']:.4f}")
    print("=" * 80)

if __name__ == '__main__':
    main()

