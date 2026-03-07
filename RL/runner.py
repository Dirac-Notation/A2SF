import torch
import json
import os
import math
import re
from typing import Dict, Any, List, Tuple, Set, Optional
from collections import Counter
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
        generation_length: int,
        reference_text: Optional[str] = None,
        dataset: str = None,
    ) -> ModelResult:
        start_time = time.time()
        
        input_tensor = self.tokenizer(prompt, truncation=False, return_tensors="pt")
        input_ids = input_tensor.input_ids.to(self.model.device)
        attention_mask = input_tensor.attention_mask.to(self.model.device)
        
        compression_config = self._create_compression_config(a, b, token_budget)
        self.model.init_cache(compression_config)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=generation_length,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract selected_indices from each layer cache after prefill
        model_selected = []
        for layer_idx in range(self.num_layers):
            layer_cache = self.model.layer_caches[layer_idx]
            model_selected.append(layer_cache.selected_indices)
        
        inference_time = time.time() - start_time
        # print(f"Inference time: {inference_time}")
        
        token_similarity = self._compute_jaccard_reward(answer_indices, model_selected)
        rouge1_score = self._compute_rouge1_f1(generated_text, reference_text)
        # reward = self._geometric_mean(token_similarity, rouge1_score)
        reward = rouge1_score
        
        return ModelResult(reward=reward, inference_time=inference_time)

    def _geometric_mean(self, x: float, y: float) -> float:
        x = max(0.0, float(x))
        y = max(0.0, float(y))
        return math.sqrt(x * y)
    
    def _compute_jaccard_reward(
        self,
        answer_indices: List[List[List[int]]],
        model_selected: List[Optional[torch.Tensor]],
    ) -> float:
        """
        Compute average Jaccard similarity between ground-truth answer_indices
        and model's selected_indices across all (layer, query_head) pairs.
        
        In GQA, query heads in the same group share the same KV head's
        selected indices, so each query head is compared individually
        against its corresponding KV head's prediction.
        
        answer_indices: [num_layers][num_attention_heads][top_k]
        model_selected: list of tensors (1, num_kv_heads, select_budget) or None per layer
        """
        jaccard_scores = []
        
        for layer_idx in range(self.num_layers):
            sel = model_selected[layer_idx]
            
            # sel: (1, num_kv_heads, select_budget) -> (num_kv_heads, select_budget)
            sel = sel[0]
            
            for q_head_idx in range(self.num_attention_heads):
                kv_head_idx = q_head_idx // self.gqa_group_size
                gt_set = set(answer_indices[layer_idx][kv_head_idx][0])
                pred_set = set(sel[kv_head_idx].cpu().tolist())
                intersection = len(gt_set & pred_set)
                union = len(gt_set | pred_set)
                jaccard = intersection / union if union > 0 else 1.0
                jaccard_scores.append(jaccard)
        
        if not jaccard_scores:
            return 0.0
        return sum(jaccard_scores) / len(jaccard_scores)

    def _compute_rouge1_f1(self, prediction: str, reference: Optional[str]) -> float:
        """
        Compute ROUGE-1 F1 score with simple whitespace/punctuation tokenization.
        """
        if reference is None:
            return 0.0

        pred_tokens = self._tokenize_for_rouge(prediction)
        ref_tokens = self._tokenize_for_rouge(reference)

        if not pred_tokens or not ref_tokens:
            return 0.0

        pred_counts = Counter(pred_tokens)
        ref_counts = Counter(ref_tokens)
        overlap = sum((pred_counts & ref_counts).values())
        if overlap == 0:
            return 0.0

        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _tokenize_for_rouge(self, text: Optional[str]) -> List[str]:
        if not text:
            return []
        return re.findall(r"\w+", text.lower())
    
    def _create_compression_config(self, a: float, b: float, token_budget: int) -> Dict[str, Any]:
        base_config = CompressionConfig()
        
        base_config.compression_method = "sigmoid"
        base_config.total_budget = token_budget
        base_config.layerwise_ratios = [1.0 for _ in range(self.num_layers)]
        base_config.local_ratios = 0.125
        base_config.a = float(a)
        base_config.b = float(b)
        
        return base_config
