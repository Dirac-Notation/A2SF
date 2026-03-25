import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

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
    Returns a feature vector:
      [ normalized_sequence_length, per-head entropy, per-head max-position ].
    where per-head max-position is normalized by sequence length.
    """

    def __init__(
        self,
        target_model,
        target_tokenizer,
        device: str = "cpu",
        output_dim: int = -1,
        num_query_tokens: int = 16,
    ):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.target_tokenizer = target_tokenizer
        self.output_dim = output_dim
        self.max_seq_length = float(target_model.config.max_position_embeddings)
        self.num_query_tokens = int(max(1, num_query_tokens))
        self.max_task_index = float(max(1, len(TASK_TYPE_ORDER) - 1))
        self.target_model = target_model
        first_layer = self.target_model.model.layers[0]
        self.input_layernorm = first_layer.input_layernorm
        self.self_attn = first_layer.self_attn
        self.num_heads = int(self.target_model.config.num_attention_heads)
        self.num_key_value_heads = int(self.target_model.config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = int(self.target_model.config.hidden_size)
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = self.self_attn.q_proj
        self.k_proj = self.self_attn.k_proj
        self.embed_tokens = self.target_model.model.embed_tokens
        if self.output_dim <= 0:
            # seq_len scalar + [entropy,max_pos] per head
            self.output_dim = 1 + (2 * self.num_heads)
        self.to(self.device)
        self.eval()

    @staticmethod
    def _entropy(prob: torch.Tensor) -> torch.Tensor:
        return -torch.sum(prob * torch.log(prob.clamp_min(1e-12)), dim=-1)

    def _build_first_layer_attention_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.embed_tokens.weight.device)
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states.to(dtype=self.q_proj.weight.dtype)
        seq_len = int(hidden_states.size(1))
        q_start = max(0, seq_len - self.num_query_tokens)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)

        # Match LlamaAttention layout and GQA behavior.
        bsz = q.shape[0]
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        cos, sin = self.self_attn.rotary_emb(k, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_key_value_groups)  # [B, num_heads, T, D]
        q = q[:, :, q_start:, :]  # SnapKV-style query window on top of full K.
        query_len = q.size(2)

        # attention score simulation with causal masking (absolute positions)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        query_positions = torch.arange(q_start, seq_len, device=attn_scores.device)  # [Q]
        key_positions = torch.arange(seq_len, device=attn_scores.device)  # [T]
        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)  # [Q, T]
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)  # [1, H, Q, T]

        # H2O-like cumulative attention score over queries
        acc_scores = attn_probs.sum(dim=2).squeeze(0)  # [H, T]
        normalized_scores = acc_scores / acc_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropies = self._entropy(normalized_scores)  # [H]
        entropies = entropies / torch.log(
            torch.tensor(float(seq_len), device=entropies.device, dtype=entropies.dtype).clamp_min(2.0)
        )
        max_pos = torch.argmax(acc_scores, dim=-1).to(dtype=torch.float32)  # [H]
        max_pos_norm = max_pos / max(float(seq_len - 1), 1.0)

        return torch.cat([entropies.to(torch.float32), max_pos_norm.to(torch.float32)], dim=-1)

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
        del generation_length, token_budget, task_type, dataset
        tokenized = self.target_tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        input_ids = tokenized.input_ids.to(self.device)
        seq_len = int(input_ids.size(1))
        seq_len_feature = min(float(seq_len), self.max_seq_length) / self.max_seq_length
        attention_features = self._build_first_layer_attention_features(input_ids)
        features = torch.cat(
            [
                torch.tensor([seq_len_feature], device=self.device, dtype=torch.float32),
                attention_features.to(self.device, dtype=torch.float32),
            ],
            dim=-1,
        )
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
            output_dim=-1,
            num_query_tokens=16
        )
        
        # Current episode cache
        self.current_prompt = None
        self.current_dataset = None
        self.current_answers: List[str] = []
        self.current_all_classes: List[str] = []
        self.current_metric_type: str = "qa_f1_score"
        self.current_generation_length = None
        self.current_token_budget = None
    
    def encode_to_state(
        self,
        prompt: str,
        generation_length: int,
        answers: List[str],
        all_classes: List[str],
        metric_type: str,
        token_budget: int,
        dataset: str = None,
        task_type: Optional[str] = None,
    ) -> torch.Tensor:
        self.current_prompt = prompt
        self.current_dataset = dataset
        self.current_answers = answers or []
        self.current_all_classes = all_classes or []
        self.current_metric_type = str(metric_type or "qa_f1_score")
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
                generation_length=self.current_generation_length,
                answers=self.current_answers,
                all_classes=self.current_all_classes,
                metric_type=self.current_metric_type,
                dataset=self.current_dataset,
            )

        reward_val = float(result.reward)
        reward = torch.tensor(reward_val, device=self.device)

        info = {
            "a": a_val,
            "b": b_val,
            "reward": reward_val,
        }
        
        return reward, info
