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
from rouge import Rouge

@dataclass
class ModelResult:
    reward: float
    inference_time: float
    generated_text: str

class A2SFModelRunner:
    def __init__(self, config: A2SFRLConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.model, self.tokenizer = load_model(config.model_name)
        
        with open("config/model2maxlen.json", "r") as f:
            self.model2maxlen = json.load(f)
        
        self.max_length = self.model2maxlen[config.model_name]
        
        # Initialize Rouge scorer for ROUGE score calculation
        self.rouge_scorer = Rouge()
    
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
        generation_length: int,
        token_budget: int,
        answer: str,
        dataset: str = None,
    ) -> ModelResult:
        start_time = time.time()
        
        input_tensor = self.tokenizer(prompt, truncation=False, return_tensors="pt")
        
        input_ids = input_tensor.input_ids.to(self.model.device)
        attention_mask = input_tensor.attention_mask.to(self.model.device)
        context_length = input_ids.size(1)
        
        compression_config = self._create_compression_config(forgetting_factor, token_budget)
        
        self.model.init_cache(compression_config)
        
        # Generate text with compression
        generated_text_compressed = self._generate_text(input_ids, attention_mask, max_new_tokens=generation_length)
        
        inference_time = time.time() - start_time
        
        # Compute similarity between full cache and compressed cache generated texts
        reward = self._compute_text_similarity(answer, generated_text_compressed)
        
        return ModelResult(
            reward=reward,
            inference_time=inference_time,
            generated_text=generated_text_compressed
        )
    
    def _generate_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int) -> str:
        """Generate text using the model with current cache configuration"""
        context_length = input_ids.size(1)
        
        with torch.no_grad():
            # Generate using model.generate() (includes forward pass)
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]
        
        # Decode only the generated part (excluding the input prompt)
        generated_text = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
        
        return generated_text
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute reward based on ROUGE score
        
        Args:
            text1: Reference text (ground truth answer)
            text2: Generated text (prediction)
        
        Returns:
            ROUGE-based reward score in [0, 1]
        """
        if not text1 or not text2:
            return 0.0
        
        # Compute ROUGE Score
        rouge_score = self._compute_rouge_score(text1, text2)
        
        # Reward is simply the ROUGE score
        return rouge_score
    
    def _compute_rouge_score(self, text1: str, text2: str) -> float:
        """Compute ROUGE-L F1 score between two texts"""
        # Rouge expects [prediction], [reference] format
        # text1 is reference (ground truth), text2 is prediction (generated)
        scores = self.rouge_scorer.get_scores([text2], [text1], avg=True)
        rouge_l_f1 = scores["rouge-l"]["f"]
        return float(rouge_l_f1)
    
    def _create_compression_config(self, forgetting_factor: float, token_budget: int) -> Dict[str, Any]:
        base_config = CompressionConfig()
        
        num_layers = self.model.config.num_hidden_layers
        base_config.compression_method = "a2sf"
        base_config.total_budget = token_budget
        base_config.layerwise_ratios = [1.0 for _ in range(num_layers)]
        base_config.local_ratios = 0.125
        # Single global forgetting_factor shared by all layers
        base_config.forgetting_factor = float(forgetting_factor)
        
        return base_config
