import torch
import json
import os
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import time
from rouge_score import rouge_scorer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_model, CompressionConfig
from .config import A2SFRLConfig

@dataclass
class ModelResult:
    prediction: str
    accuracy_score: float
    metrics: Dict[str, Any]
    inference_time: float

class A2SFRunner:
    def __init__(self, config: A2SFRLConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        self.model, self.tokenizer = load_model(config.model_name, config.gpus)
        
        with open("config/dataset2maxlen.json", "r") as f:
            self.dataset2maxlen = json.load(f)
        
        with open("config/model2maxlen.json", "r") as f:
            self.model2maxlen = json.load(f)
        
        self.max_length = self.model2maxlen[config.model_name]
    
    def tokenize_prompt(self, prompt: str, dataset: str) -> Tuple[torch.Tensor, List[str]]:
        tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > self.max_length:
            half = int(self.max_length / 2)
            prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                    self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            if "llama" in self.config.model_name:
                prompt = f"[INST]{prompt}[/INST]"
        
        input_tensor = self.tokenizer(prompt, truncation=False, return_tensors="pt")
        token_strings = [self.tokenizer.decode([token_id]) for token_id in input_tensor.input_ids[0]]
        
        return input_tensor, token_strings
    
    def run_with_compression(self, prompt: str, task: str, forgetting_factor: float, answers: List[str], dataset: str = None) -> ModelResult:
        start_time = time.time()
        
        if dataset is None:
            dataset = task
        
        input_tensor, token_strings = self.tokenize_prompt(prompt, dataset)
        
        input_ids = input_tensor.input_ids.to(self.model.device)
        attention_mask = input_tensor.attention_mask.to(torch.bfloat16).to(self.model.device)
        
        context_length = input_ids.shape[-1]
        max_gen = self.dataset2maxlen.get(dataset, 128)
        
        compression_config = self._create_compression_config(forgetting_factor, task)
        self.model.init_cache(compression_config)
        
        with torch.inference_mode():
            if dataset == "samsum":
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    min_length=context_length + 1,
                    eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                )[0]
        
        prediction = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
        inference_time = time.time() - start_time
        
        accuracy_score = self._compute_accuracy_score(prediction, task, answers)
        
        metrics = {
            "inference_time": inference_time,
            "forgetting_factor": forgetting_factor,
            "generated_length": len(output) - context_length,
            "context_length": context_length
        }
        
        return ModelResult(
            prediction=prediction,
            accuracy_score=accuracy_score,
            metrics=metrics,
            inference_time=inference_time
        )
    
    def _create_compression_config(self, forgetting_factor: float, task: str) -> Dict[str, Any]:
        base_config = CompressionConfig()
        base_config.compression_method = "a2sf"
        base_config.total_budget = 128
        base_config.layerwise_ratios = [1.0 for i in range(32)]
        base_config.local_ratios = 0.125
        base_config.forgetting_factors = [forgetting_factor for i in range(32)]
        return base_config
    
    def _compute_accuracy_score(self, prediction: str, task: str, answers: List[str]) -> float:
        if not prediction.strip():
            return 0.0
        
        if not answers:
            raise ValueError("Ground truth answers are required for accuracy computation")
        
        if isinstance(answers, str):
            ground_truth = answers
        elif isinstance(answers, list) and len(answers) > 0:
            ground_truth = answers[0]
        else:
            raise ValueError("Invalid answers format")
        
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = scorer.score(ground_truth, prediction)
        return scores['rouge1'].fmeasure
    
    def load_training_data(self, max_samples_per_task: int = 100) -> List[Dict[str, Any]]:
        training_data_path = "datasets/training_data.json"
        
        with open(training_data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        print(f"Loaded {len(training_data)} training samples from {training_data_path}")
        return training_data