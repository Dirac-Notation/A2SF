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


# Metric type ordering for one-hot encoding.
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
      [ seq_len(1), metric_type_one_hot(M),
        tova_binned(num_bins * H), snap_binned(num_bins * H) ]

    - tova_score: attention from the last 1 query only
    - snapkv_score: cumulative attention from last num_query_tokens queries
    - 16 토큰 단위로 binning (합산), max_input_length까지 왼쪽 zero-padding
    """

    def __init__(
        self,
        target_model,
        target_tokenizer,
        device: str = "cpu",
        output_dim: int = -1,
        num_query_tokens: int = 16,
        max_input_length: int = 32768,
        bin_size: int = 16,
        encoder_mode: str = "bin",
        encoder_topk: int = 32,
        encoder_views: str = "both",
        include_hidden_pool: bool = False,
        hidden_pool_window: int = 0,
    ):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.target_tokenizer = target_tokenizer
        self.output_dim = output_dim
        self.max_seq_length = float(target_model.config.max_position_embeddings)
        self.num_query_tokens = int(max(1, num_query_tokens))
        self.max_input_length = int(max_input_length)
        self.bin_size = int(bin_size)
        self.num_bins = self.max_input_length // self.bin_size
        self.encoder_mode = str(encoder_mode)
        self.encoder_topk = int(encoder_topk)
        self.encoder_views = str(encoder_views)
        if self.encoder_mode not in ("bin", "topk", "stats"):
            raise ValueError(f"encoder_mode must be 'bin', 'topk', or 'stats', got {encoder_mode!r}")
        if self.encoder_views not in ("tova", "snap", "both"):
            raise ValueError(f"encoder_views must be 'tova', 'snap', or 'both', got {encoder_views!r}")
        # Architectural constraint: state features must come from layer 0 attention only,
        # because the (a, b) compression policy must be decided BEFORE layer 0's attention
        # completes. Using a deeper layer would require materialising the full KV cache of
        # all prior layers — defeating the whole compression pipeline.

        self.max_task_index = float(max(1, len(TASK_TYPE_ORDER) - 1))

        # Avoid registering target_model / its layers as submodules.
        # Layer 0 ONLY — see architectural constraint above.
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

        self.num_metric_types = int(len(METRIC_TYPE_ORDER))
        self.num_task_types = int(len(TASK_TYPE_ORDER))
        # Optional content channel: mean-pooled post-layernorm hidden states at layer 0.
        # Adds hidden_size dims to meta section (between task_oh and tova_flat).
        # hidden_pool_window: 0 = full prompt mean (legacy), N > 0 = mean over last N tokens
        # only (captures the question/instruction tail rather than diluting with long context).
        self.include_hidden_pool = bool(include_hidden_pool)
        self.hidden_pool_window = int(hidden_pool_window) if int(hidden_pool_window) > 0 else 0
        self.hidden_pool_dim = self.hidden_size if self.include_hidden_pool else 0
        # tova/snap 각각의 per-head feature 차원
        #   bin mode: num_bins                    (per-head)
        #   topk mode: encoder_topk               (per-head)
        #   stats mode: 2 * encoder_topk + 3     (top-K positions + top-K scores + entropy + mean-pos + std-pos)
        if self.encoder_mode == "stats":
            feature_per_head = 2 * self.encoder_topk + 3
        elif self.encoder_mode == "topk":
            feature_per_head = self.encoder_topk
        else:
            feature_per_head = self.num_bins
        self.side_dim = feature_per_head * self.num_heads

        if self.output_dim <= 0:
            # state = [seq_len(1), metric_oh(M), task_oh(T), hidden_pool(H?), tova_flat, snap_flat]
            # hidden_pool is included only if include_hidden_pool=True (dim = hidden_size, else 0).
            self.output_dim = (
                1 + self.num_metric_types + self.num_task_types
                + self.hidden_pool_dim + 2 * self.side_dim
            )

    @property
    def _encode_device(self) -> torch.device:
        """Device where first-layer embedding lives."""
        return self.embed_tokens.weight.device

    def _topk_attention_scores(self, acc_scores: torch.Tensor) -> torch.Tensor:
        """Per-head top-k attention scores (sorted descending).

        acc_scores: [H, T] → [H * encoder_topk] (pad with 0 if T < k).
        """
        H, T = acc_scores.shape
        k = self.encoder_topk
        if T < k:
            pad = torch.zeros(H, k - T, device=acc_scores.device, dtype=acc_scores.dtype)
            acc_scores = torch.cat([acc_scores, pad], dim=-1)
        topk_vals, _ = acc_scores.topk(k, dim=-1)  # (H, k)
        return topk_vals.reshape(-1).to(torch.float32)

    def _reduce_scores(self, acc_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        if self.encoder_mode == "stats":
            return self._stats_features(acc_scores)
        if self.encoder_mode == "topk":
            return self._topk_attention_scores(acc_scores)
        return self._bin_attention_scores(acc_scores, seq_len)

    def _stats_features(self, acc_scores: torch.Tensor) -> torch.Tensor:
        """Per-head statistics: [top-K positions (from-end, [0,1]),
                                 top-K attention scores,
                                 normalized entropy,
                                 mean position (from-end, attention-weighted),
                                 std position (from-end, attention-weighted)].

        acc_scores: (H, T) accumulated attention scores (tova: last query only; snap: last W queries)
        Returns: (H * (2K + 3)) flat, float32, with fixed dimension independent of T.
        """
        H, T = acc_scores.shape
        k = self.encoder_topk
        if T < k:
            pad = torch.zeros(H, k - T, device=acc_scores.device, dtype=acc_scores.dtype)
            acc = torch.cat([acc_scores, pad], dim=-1)
        else:
            acc = acc_scores
        T_eff = acc.size(-1)

        # Top-K scores and (indices into acc). If we padded, some indices may point to pad region (zero score).
        topk_vals, topk_idx = acc.topk(k, dim=-1)                              # (H, k)

        # Positions from end, normalized to [0, 1]: 0 = most recent, 1 = farthest.
        denom = float(max(1, T_eff - 1))
        pos_from_end_topk = (T_eff - 1 - topk_idx).float() / denom            # (H, k)

        # Distribution over positions (softmax-like via L1-normalization of nonneg attention).
        acc_pos = acc.clamp(min=0.0)
        acc_sum = acc_pos.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        p = acc_pos / acc_sum                                                  # (H, T_eff)

        # Normalized entropy in [0, 1]: H(p) / log(T_eff).
        log_T = float(max(1e-6, torch.log(torch.tensor(float(T_eff))).item()))
        ent = -(p * (p.clamp(min=1e-12)).log()).sum(dim=-1) / log_T             # (H,)

        # Position of each token from end, normalized. Shape (T_eff,).
        positions_from_end = (T_eff - 1 - torch.arange(T_eff, device=acc.device).float()) / denom
        mean_pos = (p * positions_from_end.unsqueeze(0)).sum(dim=-1)            # (H,)
        diff = positions_from_end.unsqueeze(0) - mean_pos.unsqueeze(-1)
        var_pos = (p * diff * diff).sum(dim=-1).clamp(min=0.0)
        std_pos = var_pos.sqrt()                                                # (H,)

        stats = torch.cat([
            pos_from_end_topk,                            # (H, k)
            topk_vals.to(torch.float32),                  # (H, k)
            ent.unsqueeze(-1).to(torch.float32),          # (H, 1)
            mean_pos.unsqueeze(-1).to(torch.float32),     # (H, 1)
            std_pos.unsqueeze(-1).to(torch.float32),      # (H, 1)
        ], dim=-1)                                        # (H, 2k + 3)
        return stats.reshape(-1).to(torch.float32)

    def _bin_attention_scores(self, acc_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Accumulated attention scores [H, T]를 binning하여 [H, num_bins]로 변환.

        - seq_len < max_input_length이면 왼쪽을 0으로 패딩 후 binning.
        - seq_len > max_input_length이면 오른쪽 max_input_length 만큼만 사용.
        """
        H, T = acc_scores.shape
        target_len = self.num_bins * self.bin_size

        if T > target_len:
            # 오른쪽(최근) 부분만 사용
            acc_scores = acc_scores[:, T - target_len:]
        elif T < target_len:
            # 왼쪽 zero-padding
            pad_len = target_len - T
            acc_scores = F.pad(acc_scores, (pad_len, 0), value=0.0)

        # [H, target_len] → [H, num_bins, bin_size] → sum → [H, num_bins]
        binned = acc_scores.view(H, self.num_bins, self.bin_size).sum(dim=-1)
        # Flatten to [H * num_bins]
        return binned.reshape(-1).to(torch.float32)

    def _build_first_layer_attention_features(self, input_ids: torch.Tensor):
        """Returns (attn_features, hidden_pool). attn_features is the tova+snap concat,
        hidden_pool is (hidden_size,) if include_hidden_pool else None.
        """
        input_ids = input_ids.to(self._encode_device)
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states.to(dtype=self.q_proj.weight.dtype)
        # Content channel: mean of post-layernorm hidden states over prompt tokens.
        # If hidden_pool_window > 0, only average the last N tokens (captures the
        # question/instruction tail) instead of the full prompt (which dilutes the
        # query signal across long context tokens).
        hidden_pool = None
        if self.include_hidden_pool:
            if self.hidden_pool_window > 0:
                w = min(self.hidden_pool_window, hidden_states.size(1))
                hidden_pool = hidden_states[0, -w:].mean(dim=0).to(torch.float32)
            else:
                hidden_pool = hidden_states[0].mean(dim=0).to(torch.float32)

        seq_len = int(hidden_states.size(1))
        q_start = max(0, seq_len - self.num_query_tokens)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)

        bsz = q.shape[0]
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        cos, sin = self.self_attn.rotary_emb(k, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_key_value_groups)  # [B, num_heads, T, D]
        q_snap = q[:, :, q_start:, :]  # last num_query_tokens queries
        q_tova = q[:, :, -1:, :]       # last 1 query only

        key_positions = torch.arange(seq_len, device=hidden_states.device)

        need_tova = self.encoder_views in ("tova", "both")
        need_snap = self.encoder_views in ("snap", "both")

        if need_snap:
            # --- SnapKV score ---
            attn_scores_snap = torch.matmul(q_snap, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            snap_query_positions = torch.arange(q_start, seq_len, device=attn_scores_snap.device)
            causal_mask_snap = key_positions.unsqueeze(0) > snap_query_positions.unsqueeze(1)
            attn_scores_snap = attn_scores_snap.masked_fill(causal_mask_snap.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_probs_snap = F.softmax(attn_scores_snap, dim=-1)
            snapkv_acc = attn_probs_snap.sum(dim=2).squeeze(0)  # [H, T]
            snap_feat = self._reduce_scores(snapkv_acc, seq_len)
        else:
            snap_feat = torch.zeros(self.side_dim, device=hidden_states.device, dtype=torch.float32)

        if need_tova:
            # --- TOVA score ---
            attn_scores_tova = torch.matmul(q_tova, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            tova_query_pos = torch.tensor([seq_len - 1], device=attn_scores_tova.device)
            causal_mask_tova = key_positions.unsqueeze(0) > tova_query_pos.unsqueeze(1)
            attn_scores_tova = attn_scores_tova.masked_fill(causal_mask_tova.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_probs_tova = F.softmax(attn_scores_tova, dim=-1)
            tova_acc = attn_probs_tova.sum(dim=2).squeeze(0)  # [H, T]
            tova_feat = self._reduce_scores(tova_acc, seq_len)
        else:
            tova_feat = torch.zeros(self.side_dim, device=hidden_states.device, dtype=torch.float32)

        attn_features = torch.cat([tova_feat, snap_feat], dim=-1)
        return attn_features, hidden_pool

    def encode_context(
        self,
        text: str,
        generation_length: int,
        token_budget: int,
        metric_type: Optional[str] = None,
        task_type: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> torch.Tensor:
        del generation_length, token_budget

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

        metric_idx = metric_type_to_index(metric_type)
        metric_one_hot = torch.zeros(self.num_metric_types, device=enc_dev, dtype=torch.float32)
        metric_one_hot[metric_idx] = 1.0

        task_idx = task_type_to_index(task_type=task_type, dataset=dataset)
        task_one_hot = torch.zeros(self.num_task_types, device=enc_dev, dtype=torch.float32)
        task_one_hot[task_idx] = 1.0

        attention_features, hidden_pool = self._build_first_layer_attention_features(input_ids)

        parts = [
            torch.tensor([seq_len_feature], device=enc_dev, dtype=torch.float32),
            metric_one_hot.to(enc_dev),
            task_one_hot.to(enc_dev),
        ]
        if self.include_hidden_pool and hidden_pool is not None:
            parts.append(hidden_pool.to(device=enc_dev, dtype=torch.float32))
        parts.append(attention_features.to(device=enc_dev, dtype=torch.float32))
        features = torch.cat(parts, dim=-1)
        return features.detach().cpu()


__all__ = ["AttentionEncoder"]
