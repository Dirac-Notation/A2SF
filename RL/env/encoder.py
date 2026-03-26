import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TASK2DATASET_PATH = os.path.join(REPO_ROOT, "config", "task2dataset.json")

with open(TASK2DATASET_PATH, "r", encoding="utf-8") as f:
    TASK_TO_DATASETS = json.load(f)

TASK_TYPE_ORDER = list(TASK_TO_DATASETS.keys()) + ["unknown"]
TASK_TYPE_TO_INDEX = {name: idx for idx, name in enumerate(TASK_TYPE_ORDER)}

# LongBench dataset name -> task type mapping (derived from config/task2dataset.json)
DATASET_TO_TASK_TYPE = {
    dataset_name.lower(): task_name for task_name, datasets in TASK_TO_DATASETS.items() for dataset_name in datasets
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

    Important: Do NOT register `target_model` or any of its submodules as children of this
    module. Assigning `self.target_model = ...` would register them and then `.to(device)`
    would move the *entire* loaded model (or duplicate/move shards), breaking `device_map`
    and custom Llama wrappers (e.g. kv_llama). We keep plain references via `object.__setattr__`.
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

        # Avoid registering target_model / its layers as submodules.
        object.__setattr__(self, "target_model", target_model)
        first_layer = self.target_model.model.layers[0]
        object.__setattr__(self, "input_layernorm", first_layer.input_layernorm)
        object.__setattr__(self, "self_attn", first_layer.self_attn)

        self.num_heads = int(self.target_model.config.num_attention_heads)
        self.num_key_value_heads = int(self.target_model.config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = int(self.target_model.config.hidden_size)
        self.head_dim = self.hidden_size // self.num_heads

        object.__setattr__(self, "q_proj", self.self_attn.q_proj)
        object.__setattr__(self, "k_proj", self.self_attn.k_proj)
        object.__setattr__(self, "embed_tokens", self.target_model.model.embed_tokens)

        if self.output_dim <= 0:
            # seq_len scalar + [entropy,max_pos] per head
            self.output_dim = 1 + (2 * self.num_heads)

    @property
    def _encode_device(self) -> torch.device:
        """Device where first-layer embedding lives."""
        return self.embed_tokens.weight.device

    @staticmethod
    def _entropy(prob: torch.Tensor) -> torch.Tensor:
        return -torch.sum(prob * torch.log(prob.clamp_min(1e-12)), dim=-1)

    def _build_first_layer_attention_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self._encode_device)
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
        q = q[:, :, q_start:, :]

        # attention score simulation with causal masking (absolute positions)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        query_positions = torch.arange(q_start, seq_len, device=attn_scores.device)
        key_positions = torch.arange(seq_len, device=attn_scores.device)

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
        # generation_length / token_budget / task_type / dataset are currently kept only for API compatibility.
        del generation_length, token_budget, task_type, dataset

        tokenized = self.target_tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        enc_dev = self._encode_device
        input_ids = tokenized.input_ids.to(enc_dev)

        seq_len = int(input_ids.size(1))
        seq_len_feature = min(float(seq_len), self.max_seq_length) / self.max_seq_length
        attention_features = self._build_first_layer_attention_features(input_ids)

        features = torch.cat(
            [
                torch.tensor([seq_len_feature], device=enc_dev, dtype=torch.float32),
                attention_features.to(dtype=torch.float32),
            ],
            dim=-1,
        )

        # Decouple state vector from model shard placement.
        return features.detach().cpu()


__all__ = ["AttentionEncoder"]

