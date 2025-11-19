import torch
import json
import os
from typing import Dict, Any, List, Tuple
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

class A2SFRunner:
    def __init__(self, config: A2SFRLConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
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
        forgetting_factor: float,
        selected_indices: List[int],
        dataset: str = None
    ) -> ModelResult:
        start_time = time.time()
        
        input_tensor = self.tokenizer(prompt, truncation=False, return_tensors="pt")
        
        input_ids = input_tensor.input_ids.to(self.model.device)
        context_length = input_ids.size(1)
        
        compression_config = self._create_compression_config(forgetting_factor)
        
        self.model.init_cache(compression_config)
        
        with torch.no_grad():
            self.model(input_ids)
        
        inference_time = time.time() - start_time
        
        reward = self._compute_accuracy_score(selected_indices, context_length)
        
        return ModelResult(
            reward=reward,
            inference_time=inference_time
        )
    
    def _create_compression_config(self, forgetting_factor: float) -> Dict[str, Any]:
        base_config = CompressionConfig()
        
        base_config.compression_method = "a2sf"
        base_config.total_budget = 128
        base_config.layerwise_ratios = [1.0 for i in range(32)]
        base_config.local_ratios = 0.125
        base_config.forgetting_factors = [forgetting_factor for i in range(32)]
        
        return base_config
    
    def _compute_accuracy_score(self, selected_indices: List[int], context_length: int) -> float:
        similarity_score = 0.0
        
        num_layers = len(self.model.model.layers)
        num_heads = self.model.config.num_attention_heads
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            model_selected_indices = layer.self_attn.past_key_value.selected_indices.squeeze(0).cpu()
            answer_selected_indices = torch.tensor(selected_indices[layer_idx])
            
            num_key_value_heads = layer.self_attn.num_key_value_groups
            model_selected_indices = model_selected_indices.unsqueeze(1).expand(-1, num_key_value_heads, -1).reshape(answer_selected_indices.size(0), -1)
            
            
            for head_idx in range(num_heads):
                model_set = set(model_selected_indices[head_idx].tolist() + list(range(context_length-16, context_length)))
                answer_set = set(answer_selected_indices[head_idx].tolist())
                similarity_score += len(model_set & answer_set) / len(model_set | answer_set)
            
        similarity_score /= num_layers * num_heads
        
        return similarity_score
    
    def load_training_data(self) -> List[Dict[str, Any]]:
        training_data_path = "datasets/training_data.json"
        
        with open(training_data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        # for data in training_data:
        #     data["input_prompt"] = self.prepare_prompt(data["input_prompt"], data["dataset"])
        
        print(f"Loaded {len(training_data)} training samples from {training_data_path}")
        return training_data
