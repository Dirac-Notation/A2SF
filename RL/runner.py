import torch
import os
from typing import Dict, Any
from dataclasses import dataclass
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_model, CompressionConfig
from .main import A2SFRLConfig
from longbench_eval import (
    qa_f1_score,
    qa_f1_zh_score,
    rouge_score,
    rouge_zh_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

METRIC_FN_REGISTRY = {
    "qa_f1_score": qa_f1_score,
    "qa_f1_zh_score": qa_f1_zh_score,
    "rouge_score": rouge_score,
    "rouge_zh_score": rouge_zh_score,
    "classification_score": classification_score,
    "retrieval_score": retrieval_score,
    "retrieval_zh_score": retrieval_zh_score,
    "count_score": count_score,
    "code_sim_score": code_sim_score,
}

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
        self.debug_shapes = os.environ.get("A2SF_DEBUG_SHAPES", "0") == "1"
    
    def run_with_compression(
        self,
        prompt: str,
        a: float,
        b: float,
        token_budget: int,
        generation_length: int,
        answers: list,
        all_classes: list,
        metric_type: str,
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
                max_new_tokens=int(generation_length),
                num_beams=1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]
            pred = self.tokenizer.decode(output_ids[input_ids.size(-1):], skip_special_tokens=True)
            metric_fn = METRIC_FN_REGISTRY.get(metric_type, qa_f1_score)
            reward = 0.0
            if answers:
                for gt in answers:
                    reward = max(
                        reward,
                        float(metric_fn(pred, gt, all_classes=all_classes or [])),
                    )
            if self.debug_shapes:
                print("[A2SF_DEBUG] dataset:", dataset)
                print("[A2SF_DEBUG] metric:", metric_type)
                print("[A2SF_DEBUG] reward:", reward)
        inference_time = time.time() - start_time

        return ModelResult(reward=reward, inference_time=inference_time)
    
    def _create_compression_config(self, a: float, b: float, token_budget: int) -> Dict[str, Any]:
        base_config = CompressionConfig()
        
        base_config.compression_method = "sigmoid"
        base_config.total_budget = token_budget
        base_config.layerwise_ratios = [1.0 for _ in range(self.num_layers)]
        base_config.local_ratios = 0.125
        base_config.a = float(a)
        base_config.b = float(b)
        
        return base_config
