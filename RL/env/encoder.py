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


# Metric type ordering for one-hot encoding. "unknown" always last.
# longbench_eval.dataset2metric의 함수 이름들과 일치해야 한다.
METRIC_TYPE_ORDER = [
    "qa_f1_score",
    "qa_f1_zh_score",
    "rouge_score",
    "rouge_zh_score",
    "classification_score",
    "retrieval_score",
    "retrieval_zh_score",
    "count_score",
    "code_sim_score",
    "unknown",
]
METRIC_TYPE_TO_INDEX = {name: idx for idx, name in enumerate(METRIC_TYPE_ORDER)}


def metric_type_to_index(metric_type: Optional[str]) -> int:
    if metric_type is None:
        return int(METRIC_TYPE_TO_INDEX["unknown"])
    key = str(metric_type).strip()
    return int(METRIC_TYPE_TO_INDEX.get(key, METRIC_TYPE_TO_INDEX["unknown"]))


class AttentionEncoder(nn.Module):
    """
    Metadata encoder for RL state construction.

    Returns a feature vector structured as:
      [ seq_len, task_type_one_hot(T),
        tova_top1(H), tova_top2(H), tova_top3(H), tova_top4(H), tova_entropy(H),
        snap_top1(H), snap_top2(H), snap_top3(H), snap_top4(H), snap_entropy(H) ]

    - snapkv_score: cumulative attention from last 16 queries
    - tova_score: attention from the last 1 query only
    - top-k positions are normalized by (seq_len - 1)
    - entropy는 정규화된 점수 분포의 엔트로피를 log(seq_len)으로 나눈 [0, 1] 값
    - task_type_one_hot: one-hot over TASK_TYPE_ORDER (T categories incl. "unknown")
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

        self.topk = 4
        self.num_task_types = int(len(TASK_TYPE_ORDER))
        # tova/snap feature dim: (topk + 1)*H  -- top1~4 positions + entropy
        self.side_feat_per_head = self.topk + 1

        if self.output_dim <= 0:
            # [seq_len(1) + task_type_one_hot(T)] + [tova((topk+1)H), snap((topk+1)H)]
            self.output_dim = (
                1 + self.num_task_types + 2 * self.side_feat_per_head * self.num_heads
            )

    @property
    def _encode_device(self) -> torch.device:
        """Device where first-layer embedding lives."""
        return self.embed_tokens.weight.device

    @staticmethod
    def _topk_positions(acc_scores: torch.Tensor, seq_len: int, k: int) -> torch.Tensor:
        """Return normalized top-k positions from accumulated attention scores [H, T].

        Returns: [top1(H), top2(H), ..., topk(H)] flattened to (k*H,).
        """
        actual_k = min(k, acc_scores.size(-1))
        _, topk_idx = torch.topk(acc_scores, actual_k, dim=-1)  # [H, k]
        topk_norm = topk_idx.to(dtype=torch.float32) / max(float(seq_len - 1), 1.0)
        # topk_idx is [H, k] -> transpose to [k, H] then flatten to (k*H,)
        return topk_norm.t().contiguous().reshape(-1).to(torch.float32)

    @staticmethod
    def _entropy_per_head(acc_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """누적 attention score 분포의 정규화 엔트로피 per-head. [H] → log(seq_len)으로 나눠 [0, 1]."""
        prob = acc_scores / acc_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        ent = -torch.sum(prob * torch.log(prob.clamp_min(1e-12)), dim=-1)  # [H]
        denom = torch.log(
            torch.tensor(float(seq_len), device=ent.device, dtype=ent.dtype).clamp_min(2.0)
        )
        return (ent / denom).to(torch.float32)

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
        q_snap = q[:, :, q_start:, :]  # last num_query_tokens queries
        q_tova = q[:, :, -1:, :]       # last 1 query only

        # --- SnapKV score: cumulative attention from last 16 queries ---
        attn_scores_snap = torch.matmul(q_snap, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        snap_query_positions = torch.arange(q_start, seq_len, device=attn_scores_snap.device)
        key_positions = torch.arange(seq_len, device=attn_scores_snap.device)
        causal_mask_snap = key_positions.unsqueeze(0) > snap_query_positions.unsqueeze(1)
        attn_scores_snap = attn_scores_snap.masked_fill(causal_mask_snap.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_probs_snap = F.softmax(attn_scores_snap, dim=-1)
        snapkv_acc = attn_probs_snap.sum(dim=2).squeeze(0)  # [H, T]

        # --- TOVA score: attention from the last 1 query ---
        attn_scores_tova = torch.matmul(q_tova, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        tova_query_pos = torch.tensor([seq_len - 1], device=attn_scores_tova.device)
        causal_mask_tova = key_positions.unsqueeze(0) > tova_query_pos.unsqueeze(1)
        attn_scores_tova = attn_scores_tova.masked_fill(causal_mask_tova.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_probs_tova = F.softmax(attn_scores_tova, dim=-1)
        tova_acc = attn_probs_tova.sum(dim=2).squeeze(0)  # [H, T]

        tova_topk = self._topk_positions(tova_acc, seq_len, self.topk)    # (4*H,)
        snap_topk = self._topk_positions(snapkv_acc, seq_len, self.topk)  # (4*H,)
        tova_ent = self._entropy_per_head(tova_acc, seq_len)              # (H,)
        snap_ent = self._entropy_per_head(snapkv_acc, seq_len)            # (H,)

        # Return: [tova_top1..4(4H), tova_entropy(H), snap_top1..4(4H), snap_entropy(H)]
        return torch.cat([tova_topk, tova_ent, snap_topk, snap_ent], dim=-1)

    def encode_context(
        self,
        text: str,
        generation_length: int,
        token_budget: int,
        metric_type: Optional[str] = None,
        task_type: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> torch.Tensor:
        # generation_length / token_budget / metric_type are kept for API compatibility.
        del generation_length, token_budget, metric_type

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

        task_idx = task_type_to_index(task_type=task_type, dataset=dataset)
        task_one_hot = torch.zeros(self.num_task_types, device=enc_dev, dtype=torch.float32)
        task_one_hot[task_idx] = 1.0

        attention_features = self._build_first_layer_attention_features(input_ids)

        features = torch.cat(
            [
                torch.tensor([seq_len_feature], device=enc_dev, dtype=torch.float32),
                task_one_hot,
                attention_features.to(dtype=torch.float32),
            ],
            dim=-1,
        )

        # Decouple state vector from model shard placement.
        return features.detach().cpu()


__all__ = ["AttentionEncoder"]
