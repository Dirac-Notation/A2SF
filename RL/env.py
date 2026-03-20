import json
import os
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from .main import A2SFRLConfig

TASK2DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "task2dataset.json",
)
with open(TASK2DATASET_PATH, "r", encoding="utf-8") as f:
    TASK_TO_DATASETS = json.load(f)

TASK_TYPE_ORDER = list(TASK_TO_DATASETS.keys()) + ["unknown"]
TASK_TYPE_TO_INDEX = {name: idx for idx, name in enumerate(TASK_TYPE_ORDER)}

# LongBench dataset name -> task type mapping (derived from config/task2dataset.json)
DATASET_TO_TASK_TYPE = {
    dataset_name.lower(): task_name
    for task_name, datasets in TASK_TO_DATASETS.items()
    for dataset_name in datasets
}


def normalize_task_type(task_type: Optional[str]) -> str:
    if task_type is None:
        return "unknown"
    key = str(task_type).strip()
    return key if key in TASK_TYPE_TO_INDEX else "unknown"


def resolve_task_type(dataset: Optional[str] = None, task_type: Optional[str] = None) -> str:
    normalized = normalize_task_type(task_type)
    if normalized != "unknown":
        return normalized
    if dataset is None:
        return "unknown"
    dataset_key = str(dataset).strip().lower()
    return DATASET_TO_TASK_TYPE.get(dataset_key, "unknown")


def task_type_to_index(task_type: Optional[str] = None, dataset: Optional[str] = None) -> int:
    resolved = resolve_task_type(dataset=dataset, task_type=task_type)
    return int(TASK_TYPE_TO_INDEX.get(resolved, TASK_TYPE_TO_INDEX["unknown"]))


class AttentionEncoder(nn.Module):
    """
    Metadata encoder for RL state construction.
    Returns a 2D feature vector:
      [ normalized_sequence_length, normalized_task_type_index ].
    """

    def __init__(
        self,
        target_model,
        target_tokenizer,
        device: str = "cpu",
        output_dim: int = 2,
        num_query_tokens: int = 16,
    ):
        super().__init__()
        del num_query_tokens  # kept for backward-compatible constructor
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.target_tokenizer = target_tokenizer
        self.output_dim = output_dim
        self.max_seq_length = float(target_model.config.max_position_embeddings)
        self.max_task_index = float(max(1, len(TASK_TYPE_ORDER) - 1))
        self.to(self.device)
        self.eval()

    def encode_context(
        self,
        text: str,
        generation_length: int,
        token_budget: int,
        task_type: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            text: Input text string
            generation_length: Unused, kept for backward compatibility
            token_budget: Unused, kept for backward compatibility
            task_type: Canonical task type string when available
            dataset: Dataset name fallback for task type resolution
        Returns:
            torch.Tensor: Feature vector of shape (2,)
        """
        del generation_length, token_budget
        tokenized = self.target_tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        seq_len = int(tokenized.input_ids.size(1))
        seq_len_feature = min(float(seq_len), self.max_seq_length) / self.max_seq_length

        task_index = float(task_type_to_index(task_type=task_type, dataset=dataset))
        task_feature = task_index / self.max_task_index

        features = torch.tensor([seq_len_feature, task_feature], device=self.device, dtype=torch.float32)
        return features

class A2SFEnv:
    """RL Environment for A2SF model (single-step / bandit)"""
    
    def __init__(self, runner, config: A2SFRLConfig):
        self.runner = runner
        self.config = config
        self.device = torch.device(config.device)
        
        # Metadata encoder used to build compact RL state features
        self.context_encoder = AttentionEncoder(
            target_model=runner.model,
            target_tokenizer=runner.tokenizer,
            device=config.device,
            output_dim=2,
            num_query_tokens=16
        )
        
        # Current episode cache
        self.current_prompt = None
        self.current_dataset = None
        self.current_target_prob_data = None
        self.current_generation_length = None
        self.current_token_budget = None
    
    def encode_to_state(
        self,
        prompt: str,
        generation_length: int,
        target_prob_data: Dict[str, torch.Tensor],
        token_budget: int,
        dataset: str = None,
        task_type: Optional[str] = None,
    ) -> torch.Tensor:
        self.current_prompt = prompt
        self.current_dataset = dataset
        self.current_target_prob_data = target_prob_data
        self.current_generation_length = generation_length
        self.current_token_budget = token_budget
        
        return self.context_encoder.encode_context(
            prompt,
            generation_length,
            token_budget,
            task_type=task_type,
            dataset=dataset,
        ).to(self.device, dtype=torch.float32)
    
    def step(self, action: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            action: tuple of (a, b) tensors for sigmoid cache parameters
        Returns:
            reward, info
        """
        a_val, b_val = action
        a_val = float(a_val.item() if isinstance(a_val, torch.Tensor) else a_val)
        b_val = float(b_val.item() if isinstance(b_val, torch.Tensor) else b_val)

        with torch.no_grad():
            result = self.runner.run_with_compression(
                prompt=self.current_prompt,
                a=a_val,
                b=b_val,
                token_budget=self.current_token_budget,
                target_prob_data=self.current_target_prob_data,
                dataset=self.current_dataset,
            )
        
        reward = torch.tensor(float(result.reward), device=self.device)
        
        info = {
            "a": a_val,
            "b": b_val,
            "reward": result.reward,
        }
        
        return reward, info
