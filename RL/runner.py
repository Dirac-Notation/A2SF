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
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

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
        
        # Initialize BERT model for text similarity using CLS embeddings
        self.bert_model_name = "bert-base-uncased"
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
        self.bert_model.to(self.device)
        self.bert_model.eval()  # Set to evaluation mode
        self.bert_max_length = 512  # BERT's maximum input length
    
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
        answer: str,
        dataset: str = None,
    ) -> ModelResult:
        start_time = time.time()
        
        input_tensor = self.tokenizer(prompt, truncation=False, return_tensors="pt")
        
        input_ids = input_tensor.input_ids.to(self.model.device)
        attention_mask = input_tensor.attention_mask.to(self.model.device)
        context_length = input_ids.size(1)
        
        compression_config = self._create_compression_config(forgetting_factor)
        
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
        """Compute cosine similarity between BERT CLS embeddings of two texts"""
        if not text1 or not text2:
            return 0.0
        
        with torch.no_grad():
            # Tokenize and encode text1 (truncation=True automatically truncates from the end if exceeds max_length)
            encoded1 = self.bert_tokenizer(
                text1,
                max_length=self.bert_max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Tokenize and encode text2 (truncation=True automatically truncates from the end if exceeds max_length)
            encoded2 = self.bert_tokenizer(
                text2,
                max_length=self.bert_max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids1 = encoded1["input_ids"].to(self.device)
            attention_mask1 = encoded1["attention_mask"].to(self.device)
            input_ids2 = encoded2["input_ids"].to(self.device)
            attention_mask2 = encoded2["attention_mask"].to(self.device)
            
            # Get BERT CLS embeddings
            outputs1 = self.bert_model(input_ids=input_ids1, attention_mask=attention_mask1)
            cls_embedding1 = outputs1.last_hidden_state[:, 0, :]  # CLS token embedding (batch_size, hidden_size)
            
            outputs2 = self.bert_model(input_ids=input_ids2, attention_mask=attention_mask2)
            cls_embedding2 = outputs2.last_hidden_state[:, 0, :]  # CLS token embedding (batch_size, hidden_size)
            
            # Compute cosine similarity
            # Normalize embeddings
            cls_embedding1_norm = F.normalize(cls_embedding1, p=2, dim=1)
            cls_embedding2_norm = F.normalize(cls_embedding2, p=2, dim=1)
            
            # Cosine similarity: dot product of normalized vectors
            cosine_sim = torch.sum(cls_embedding1_norm * cls_embedding2_norm, dim=1)
            
            return float(cosine_sim.item())
    
    def _create_compression_config(self, forgetting_factor: float) -> Dict[str, Any]:
        base_config = CompressionConfig()
        
        num_layers = self.model.config.num_hidden_layers
        base_config.compression_method = "a2sf"
        base_config.total_budget = 128
        base_config.layerwise_ratios = [1.0 for _ in range(num_layers)]
        base_config.local_ratios = 0.125
        # Single global forgetting_factor shared by all layers
        base_config.forgetting_factor = float(forgetting_factor)
        
        return base_config
