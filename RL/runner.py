import torch
import json
import os
from typing import Dict, Any, List, Tuple, Set
from dataclasses import dataclass
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_model, set_seed, CompressionConfig
from .config import A2SFRLConfig

@dataclass
class ModelResult:
    reward: float
    inference_time: float

class A2SFModelRunner:
    def __init__(self, config: A2SFRLConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.rbo_p = config.rbo_p
        
        self.model, self.tokenizer = load_model(config.model_name, config.gpus)
        
        with open("config/model2maxlen.json", "r") as f:
            self.model2maxlen = json.load(f)
        
        self.max_length = self.model2maxlen[config.model_name]
    
    def prepare_prompt(self, prompt: str, dataset: str) -> Tuple[torch.Tensor, List[str]]:
        tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > self.max_length:
            half = int(self.max_length / 2)
            prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                    self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            if "llama" in self.config.model_name:
                prompt = f"[INST]{prompt}[/INST]"
        
        return prompt
    
    def run_with_compression(
        self, 
        prompt: str, 
        a: float,
        b: float,
        selected_indices: List[int],
        dataset: str = None
    ) -> ModelResult:
        start_time = time.time()
        
        input_tensor = self.tokenizer(prompt, truncation=False, return_tensors="pt")
        
        input_ids = input_tensor.input_ids.to(self.model.device)
        context_length = input_ids.size(1)
        
        compression_config = self._create_compression_config(a, b)
        
        self.model.init_cache(compression_config)
        
        with torch.no_grad():
            self.model(input_ids)
        
        inference_time = time.time() - start_time
        
        reward = self._compute_accuracy_score(selected_indices, context_length, self.rbo_p)
        
        return ModelResult(
            reward=reward,
            inference_time=inference_time
        )
    
    def _create_compression_config(self, a: float, b: float) -> Dict[str, Any]:
        base_config = CompressionConfig()
        
        base_config.compression_method = "sigmoid"
        base_config.total_budget = 128
        base_config.layerwise_ratios = [1.0 for i in range(32)]
        base_config.local_ratios = 0.125
        base_config.a = a
        base_config.b = b
        
        return base_config

    @staticmethod
    def calculate_rbo(list1: List[int], list2: List[int], p: float) -> float:
        """
        두 리스트 간의 Rank-Biased Overlap (RBO)를 계산합니다.
        list1, list2: 순위가 매겨진 요소들의 리스트 (앞쪽일수록 중요도 높음)
        p: persistence parameter (0 < p < 1)
        """
        # 비교 깊이 설정 (더 긴 리스트 기준)
        k = max(len(list1), len(list2))
        
        overlap = 0
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
    
    def _compute_accuracy_score(self, selected_indices: List[int], context_length: int, rbo_p: float) -> float:
        similarity_score = 0.0
        
        num_layers = len(self.model.model.layers)
        num_heads = self.model.config.num_attention_heads

        for layer_idx, layer in enumerate(self.model.model.layers):
            # 모델이 선택한 인덱스 가져오기
            model_selected_indices = layer.self_attn.past_key_value.selected_indices.squeeze(0).cpu()
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
                similarity_score += self.calculate_rbo(model_list, answer_list, rbo_p)
            
        similarity_score /= (num_layers * num_heads)
        
        return similarity_score