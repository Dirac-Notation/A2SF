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

from utils import load_model, set_seed, CompressionConfig
from .config import A2SFRLConfig

@dataclass
class ModelResult:
    """Result from model inference"""
    prediction: str
    accuracy_score: float
    metrics: Dict[str, Any]
    inference_time: float

class A2SFRunner:
    """Runner for A2SF model with RL integration"""
    
    def __init__(self, config: A2SFRLConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.model, self.tokenizer = load_model(config.model_name, config.gpus)
        
        # Dataset configurations are now loaded from training_data.json
        
        # Load max lengths
        with open("config/dataset2maxlen.json", "r") as f:
            self.dataset2maxlen = json.load(f)
        
        with open("config/model2maxlen.json", "r") as f:
            self.model2maxlen = json.load(f)
        
        self.max_length = self.model2maxlen[config.model_name]
    
    def load_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def tokenize_prompt(self, prompt: str, dataset: str) -> Tuple[torch.Tensor, List[str]]:
        """
        Tokenize prompt and return tokens and token strings
        
        Args:
            prompt: Input prompt
            dataset: Dataset name
            
        Returns:
            Tuple of (token_ids, token_strings)
        """
        # Tokenize
        tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        # Truncate if too long
        if len(tokenized_prompt) > self.max_length:
            half = int(self.max_length / 2)
            prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                    self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        # Add instruction format for LLaMA models
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            if "llama" in self.config.model_name:
                prompt = f"[INST]{prompt}[/INST]"
        
        # Final tokenization
        input_tensor = self.tokenizer(prompt, truncation=False, return_tensors="pt")
        
        return input_tensor
    
    def run_with_compression(
        self, 
        prompt: str, 
        forgetting_factor: float,
        answers: List[str],
        dataset: str = None
    ) -> ModelResult:
        """
        Run model with specified compression ratio
        
        Args:
            prompt: Input prompt
            forgetting_factor: Compression ratio (0.0 to 1.0)
            answers: Ground truth answers for evaluation (required)
            dataset: Dataset name (if None, uses task as dataset name)
            
        Returns:
            ModelResult with prediction and metrics
        """
        start_time = time.time()
        
        # Tokenize prompt
        input_tensor = self.tokenize_prompt(prompt, dataset)
        
        input_ids = input_tensor.input_ids.to(self.model.device)
        attention_mask = input_tensor.attention_mask.to(torch.bfloat16).to(self.model.device)
        
        context_length = input_ids.shape[-1]
        # Get max generation length from dataset2maxlen, fallback to 128
        max_gen = self.dataset2maxlen.get(dataset, 128)
        
        # Create compression config with the RL-determined forgetting factor
        compression_config = self._create_compression_config(forgetting_factor)
        
        # Initialize cache
        self.model.init_cache(compression_config)
        
        # Generate
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
        
        # Decode prediction
        prediction = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
        
        # prediction = answers
        
        inference_time = time.time() - start_time
        
        # Compute metrics using ground truth answers
        accuracy_score = self._compute_accuracy_score(prediction, dataset, answers)
        
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
    
    def _create_compression_config(self, forgetting_factor: float) -> Dict[str, Any]:
        """
        Create compression configuration with the specified forgetting factor
        
        Args:
            forgetting_factor: Forgetting factor from RL agent (0.0 to 1.0)
            dataset: Dataset name
            
        Returns:
            Compression configuration
        """
        # Load base config
        base_config = CompressionConfig()
        
        # Add forgetting factor configuration
        base_config.compression_method = "a2sf"
        base_config.total_budget = 128
        base_config.layerwise_ratios = [1.0 for i in range(32)]
        base_config.local_ratios = 0.125
        base_config.forgetting_factors = [forgetting_factor for i in range(32)]
        
        return base_config
    
    def _compute_accuracy_score(self, prediction: str, dataset: str, answers: List[str]) -> float:
        """
        Compute accuracy score for prediction using ROUGE score
        
        Args:
            prediction: Model prediction
            dataset: Dataset name
            answers: Ground truth answers (required)
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if not prediction.strip():
            return 0.0
        
        # Ground truth answers are required
        if not answers:
            raise ValueError("Ground truth answers are required for accuracy computation")
        
        # Handle both string and list formats for answers
        if isinstance(answers, str):
            ground_truth = answers
        elif isinstance(answers, list) and len(answers) > 0:
            ground_truth = answers[0]
        else:
            raise ValueError("Invalid answers format")
        # Initialize ROUGE scorer (ROUGE-L only)
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        
        # Compute ROUGE-L score against the ground truth answer
        scores = scorer.score(ground_truth, prediction)
        # Use ROUGE-L F1 score
        rouge_1_f1 = scores['rouge1'].fmeasure
        return rouge_1_f1
    
    def load_training_data(self) -> List[Dict[str, Any]]:
        """
        Load training data from generated training_data.json
        
        Args:
            max_samples_per_dataset: Maximum number of samples per dataset (not used for training_data.json)
            
        Returns:
            List of training samples
        """
        training_data_path = "datasets/training_data.json"
        
        with open(training_data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        print(f"Loaded {len(training_data)} training samples from {training_data_path}")
        return training_data
