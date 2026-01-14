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
from rouge_score import rouge_scorer

@dataclass
class ModelResult:
    reward: float
    inference_time: float
    generated_text: str

class A2SFModelRunner:
    def __init__(self, config: A2SFRLConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        self.model, self.tokenizer = load_model(config.model_name, config.gpus)
        
        with open("config/model2maxlen.json", "r") as f:
            self.model2maxlen = json.load(f)
        
        self.max_length = self.model2maxlen[config.model_name]
        
        # Initialize ROUGE scorer for text similarity
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
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
        generated_text_full: str,  # Full cache generated text (baseline)
        dataset: str = None
    ) -> ModelResult:
        start_time = time.time()
        
        input_tensor = self.tokenizer(prompt, truncation=False, return_tensors="pt")
        
        input_ids = input_tensor.input_ids.to(self.model.device)
        attention_mask = input_tensor.attention_mask.to(self.model.device)
        context_length = input_ids.size(1)
        
        compression_config = self._create_compression_config(a, b)
        
        self.model.init_cache(compression_config)
        
        # Generate text with compression
        generated_text_compressed = self._generate_text(input_ids, attention_mask)
        
        inference_time = time.time() - start_time
        
        # Compute similarity between full cache and compressed cache generated texts
        reward = self._compute_text_similarity(generated_text_full, generated_text_compressed)
        
        return ModelResult(
            reward=reward,
            inference_time=inference_time,
            generated_text=generated_text_compressed
        )
    
    def _generate_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int = 16) -> str:
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
        """Compute ROUGE-L F1 score between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Compute ROUGE scores
        scores = self.rouge_scorer.score(text1, text2)
        
        # Use ROUGE-L F1 score as the similarity metric
        # ROUGE-L considers longest common subsequence
        rouge_l_f1 = scores['rougeL'].fmeasure
        
        return rouge_l_f1
    
    def _create_compression_config(self, a: float, b: float) -> Dict[str, Any]:
        base_config = CompressionConfig()
        
        base_config.compression_method = "sigmoid"
        base_config.total_budget = 128
        base_config.layerwise_ratios = [1.0 for i in range(32)]
        base_config.local_ratios = 0.125
        base_config.a = a
        base_config.b = b
        
        return base_config
