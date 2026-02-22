import torch
import json
import os
from typing import Dict, Any, List, Tuple, Set, Optional
from dataclasses import dataclass
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_model, set_seed, CompressionConfig
from .main import A2SFRLConfig

@dataclass
class ModelResult:
    reward: float
    inference_time: float

class A2SFModelRunner:
    def __init__(self, config: A2SFRLConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.model, self.tokenizer = load_model(config.model_name)
        
        self.num_layers = self.model.config.num_hidden_layers
        self.num_attention_heads = self.model.config.num_attention_heads
        self.num_kv_heads = self.model.config.num_key_value_heads
        self.gqa_group_size = self.num_attention_heads // self.num_kv_heads
    
    def run_with_compression(
        self,
        prompt: str,
        a: float,
        b: float,
        token_budget: int,
        answer_indices: List[List[List[int]]],
        dataset: str = None,
    ) -> ModelResult:
        start_time = time.time()
        
        input_tensor = self.tokenizer(prompt, truncation=False, return_tensors="pt")
        input_ids = input_tensor.input_ids.to(self.model.device)
        attention_mask = input_tensor.attention_mask.to(self.model.device)
        
        compression_config = self._create_compression_config(a, b, token_budget)
        self.model.init_cache(compression_config)
        
        with torch.no_grad():
            self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract selected_indices from each layer cache after prefill
        model_selected = []
        for layer_idx in range(self.num_layers):
            layer_cache = self.model.layer_caches[layer_idx]
            model_selected.append(layer_cache.selected_indices)
        
        inference_time = time.time() - start_time
        # print(f"Inference time: {inference_time}")
        
        reward = self._compute_jaccard_reward(answer_indices, model_selected)
        
        return ModelResult(reward=reward, inference_time=inference_time)
    
    def _compute_jaccard_reward(
        self,
        answer_indices: List[List[List[int]]],
        model_selected: List[Optional[torch.Tensor]],
    ) -> float:
        """
        Compute average Jaccard similarity between ground-truth answer_indices
        and model's selected_indices across all (layer, kv_head) pairs.
        
        answer_indices: [num_layers][num_attention_heads][128]
        model_selected: list of tensors (1, num_kv_heads, select_budget) or None per layer
        """
        jaccard_scores = []
        
        for layer_idx in range(self.num_layers):
            sel = model_selected[layer_idx]
            if sel is None:
                continue
            
            # sel: (1, num_kv_heads, select_budget) -> (num_kv_heads, select_budget)
            sel = sel[0]
            
            for kv_head_idx in range(self.num_kv_heads):
                # Union of ground-truth indices across attention heads in this GQA group
                gt_set = set()
                for h in range(kv_head_idx * self.gqa_group_size,
                               (kv_head_idx + 1) * self.gqa_group_size):
                    gt_set.update(answer_indices[layer_idx][h])
                
                pred_set = set(sel[kv_head_idx].cpu().tolist())
                
                intersection = len(gt_set & pred_set)
                union = len(gt_set | pred_set)
                jaccard = intersection / union if union > 0 else 1.0
                jaccard_scores.append(jaccard)
        
        if not jaccard_scores:
            return 0.0
        return sum(jaccard_scores) / len(jaccard_scores)
    
    def _create_compression_config(self, a: float, b: float, token_budget: int) -> Dict[str, Any]:
        base_config = CompressionConfig()
        
        base_config.compression_method = "sigmoid"
        base_config.total_budget = token_budget
        base_config.layerwise_ratios = [1.0 for _ in range(self.num_layers)]
        base_config.local_ratios = 0.125
        base_config.a = float(a)
        base_config.b = float(b)
        
        return base_config
