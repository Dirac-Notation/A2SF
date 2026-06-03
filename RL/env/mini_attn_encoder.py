"""Encoder that uses a pretrained MiniCrossAttn instead of target-model layer-0 attention.

Drop-in replacement for AttentionEncoder. Provides the same flat-tensor interface
expected by A2SFModel:

  output_dim, num_heads, num_metric_types, num_task_types, side_dim, hidden_pool_dim,
  encode_context(text, generation_length, token_budget, metric_type, task_type, dataset)

The mini-attn module is loaded from a checkpoint and FROZEN (no_grad, eval mode).

Extra views (appended after the base mini-attn feature):
  extra_view='none'           : no extra (state_dim unchanged)
  extra_view='pre_rope_mean'  : layer-0 pre-RoPE Q·K stats, heads aggregated by mean
  extra_view='pre_rope_max'   : layer-0 pre-RoPE Q·K stats, heads aggregated by max
  extra_view='pre_rope_z_max' : layer-0 pre-RoPE Q·K, per-head z-score then max
  extra_view='snap_l0'        : layer-0 post-RoPE snap attention stats (W=16 queries)
  extra_view='position_prior' : recency-decay + sink token prior stats
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .encoder import (
    METRIC_TYPE_ORDER, TASK_TYPE_ORDER,
    metric_type_to_index, task_type_to_index,
)

EXTRA_VIEW_CHOICES = [
    'none',
    'pre_rope_mean', 'pre_rope_max', 'pre_rope_z_max',
    'snap_l0',
    'position_prior',
]


def _load_mini_attn_class():
    """Import MiniCrossAttn from script/train_mini_attn.py."""
    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "train_mini_attn", repo_root / "script" / "train_mini_attn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MiniCrossAttn


class MiniAttnEncoder(nn.Module):
    """Encoder using a pretrained MiniCrossAttn for attention-derived features.

    State layout:
      [ seq_len(1), metric_one_hot(M), task_one_hot(T),
        hidden_pool(H?),  # only if include_hidden_pool
        mini_attn_stats(side_dim),
        extra_view_stats(extra_view_dim) ]  # only if extra_view != 'none'

    side_dim = (2 * encoder_topk + 3) * mini_n_heads_exposed
    where mini_n_heads_exposed = 1 (aggregated single view).
    extra_view_dim = (2 * encoder_topk + 3) for all non-none extra views.
    """

    def __init__(
        self,
        target_model,
        target_tokenizer,
        mini_attn_ckpt_path: str,
        device: str = "cpu",
        output_dim: int = -1,
        encoder_topk: int = 16,
        include_hidden_pool: bool = False,
        hidden_pool_window: int = 0,
        max_input_length: int = 32768,
        bin_size: int = 16,
        feature_mode: str = "stats",
        endalign_vec_len: int = 256,
        endalign_bin_size: int = 128,
        endalign_sink_mask: int = 4,
        endalign_include_stats: bool = True,
        single_view: bool = False,
        include_metric_oh: bool = True,
        include_seq_len: bool = True,
        include_task_oh: bool = True,
        # Extra view
        extra_view: str = 'none',
        pre_rope_query_window: int = 16,
    ):
        super().__init__()
        if extra_view not in EXTRA_VIEW_CHOICES:
            raise ValueError(f"extra_view must be one of {EXTRA_VIEW_CHOICES}, got {extra_view!r}")

        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.target_tokenizer = target_tokenizer
        self.encoder_topk = int(encoder_topk)
        self.max_input_length = int(max_input_length)
        self.max_seq_length = float(target_model.config.max_position_embeddings)
        self.bin_size = int(bin_size)

        object.__setattr__(self, "target_model", target_model)
        object.__setattr__(self, "embed_tokens", target_model.model.embed_tokens)
        first_layer = target_model.model.layers[0]
        object.__setattr__(self, "input_layernorm", first_layer.input_layernorm)

        self.hidden_size = int(target_model.config.hidden_size)
        self.num_heads = 1
        self.num_key_value_heads = 1
        self.num_key_value_groups = 1
        self.head_dim = self.hidden_size

        self.num_metric_types = int(len(METRIC_TYPE_ORDER))
        self.num_task_types = int(len(TASK_TYPE_ORDER))

        self.include_hidden_pool = bool(include_hidden_pool)
        self.hidden_pool_window = int(hidden_pool_window) if int(hidden_pool_window) > 0 else 0
        self.hidden_pool_dim = self.hidden_size if self.include_hidden_pool else 0

        self.feature_mode = str(feature_mode)
        self.endalign_vec_len = int(endalign_vec_len)
        self.endalign_bin_size = int(endalign_bin_size)
        self.endalign_sink_mask = int(endalign_sink_mask)
        self.endalign_include_stats = bool(endalign_include_stats)
        self.single_view = bool(single_view)
        self.include_metric_oh = bool(include_metric_oh)
        self.include_seq_len = bool(include_seq_len)
        self.include_task_oh = bool(include_task_oh)

        if self.feature_mode == "stats":
            feature_per_head = 2 * self.encoder_topk + 3
            self.side_dim = feature_per_head * self.num_heads
        elif self.feature_mode == "endaligned":
            self.side_dim = self.endalign_vec_len + (3 if self.endalign_include_stats else 0)
        else:
            raise ValueError(f"unknown feature_mode {feature_mode}")
        self._features_per_view = 1 if self.single_view else 2

        self.encoder_mode = "stats"
        self.encoder_views = "both"
        self.encoder_include_hidden_pool = self.include_hidden_pool
        self.encoder_hidden_pool_window = self.hidden_pool_window
        self.encoder_bin_size = self.bin_size

        # Extra view setup
        self.extra_view = str(extra_view)
        self.pre_rope_query_window = int(pre_rope_query_window)
        self.extra_view_dim = 0
        if self.extra_view != 'none':
            self.extra_view_dim = 2 * self.encoder_topk + 3  # same as stats feature_per_head

        # For pre_rope_* and snap_l0: need layer-0 q/k projections
        if self.extra_view in ('pre_rope_mean', 'pre_rope_max', 'pre_rope_z_max', 'snap_l0'):
            object.__setattr__(self, "_qk_q_proj", first_layer.self_attn.q_proj)
            object.__setattr__(self, "_qk_k_proj", first_layer.self_attn.k_proj)
            self._qk_n_heads = int(target_model.config.num_attention_heads)
            self._qk_n_kv = int(target_model.config.num_key_value_heads)
            self._qk_groups = self._qk_n_heads // self._qk_n_kv
            self._qk_head_dim = int(target_model.config.hidden_size) // self._qk_n_heads
        if self.extra_view == 'snap_l0':
            object.__setattr__(self, "_qk_rotary_emb", first_layer.self_attn.rotary_emb)

        # Load mini-attn
        MiniCrossAttn = _load_mini_attn_class()
        ckpt = torch.load(mini_attn_ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self.mini_attn_cfg = cfg
        self.mini_attn = MiniCrossAttn(
            embed_dim=int(target_model.config.hidden_size),
            hidden=int(cfg["hidden"]),
            n_heads=int(cfg["n_heads"]),
            query_window=int(cfg["query_window"]),
            dropout=0.0,
            self_attn_layers=1,
            max_len=int(self.max_input_length),
            use_distance_bias=True,
            causal_mode=str(cfg.get("causal_mode", "bidir-causal")),
        )
        self.mini_attn.load_state_dict(ckpt["model_state_dict"])
        self.mini_attn.eval()
        for p in self.mini_attn.parameters():
            p.requires_grad_(False)
        self.mini_attn = self.mini_attn.to(self._encode_device).to(torch.float32)

        if output_dim <= 0:
            self.output_dim = (
                (1 if self.include_seq_len else 0)
                + (self.num_metric_types if self.include_metric_oh else 0)
                + (self.num_task_types if self.include_task_oh else 0)
                + self.hidden_pool_dim
                + self._features_per_view * self.side_dim
                + self.extra_view_dim
            )
        else:
            self.output_dim = int(output_dim)

    @property
    def _encode_device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def _stats_features(self, scores_1d: torch.Tensor) -> torch.Tensor:
        scores = scores_1d.unsqueeze(0)  # (1, L)
        H, T = scores.shape
        k = self.encoder_topk
        if T < k:
            pad = torch.zeros(H, k - T, device=scores.device, dtype=scores.dtype)
            acc = torch.cat([scores, pad], dim=-1)
        else:
            acc = scores
        T_eff = acc.size(-1)

        topk_vals, topk_idx = acc.topk(k, dim=-1)
        denom = float(max(1, T_eff - 1))
        pos_from_end_topk = (T_eff - 1 - topk_idx).float() / denom

        acc_pos = acc.clamp(min=0.0)
        acc_sum = acc_pos.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        p = acc_pos / acc_sum
        log_T = float(max(1e-6, torch.log(torch.tensor(float(T_eff))).item()))
        ent = -(p * (p.clamp(min=1e-12)).log()).sum(dim=-1) / log_T

        positions_from_end = (T_eff - 1 - torch.arange(T_eff, device=acc.device).float()) / denom
        mean_pos = (p * positions_from_end.unsqueeze(0)).sum(dim=-1)
        diff = positions_from_end.unsqueeze(0) - mean_pos.unsqueeze(-1)
        var_pos = (p * diff * diff).sum(dim=-1).clamp(min=0.0)
        std_pos = var_pos.sqrt()

        stats = torch.cat([
            pos_from_end_topk,
            topk_vals.to(torch.float32),
            ent.unsqueeze(-1).to(torch.float32),
            mean_pos.unsqueeze(-1).to(torch.float32),
            std_pos.unsqueeze(-1).to(torch.float32),
        ], dim=-1)
        return stats.reshape(-1).to(torch.float32)

    def _endaligned_features(self, scores_1d: torch.Tensor, plen: int,
                              include_stats: bool = True) -> torch.Tensor:
        device = scores_1d.device
        L = int(scores_1d.size(0))
        scores = scores_1d.clone().to(torch.float32)
        if self.endalign_sink_mask > 0:
            scores[:self.endalign_sink_mask] = 0.0

        out = torch.zeros(self.endalign_vec_len, device=device, dtype=torch.float32)
        distance = (L - 1) - torch.arange(L, device=device)
        bin_idx = (self.endalign_vec_len - 1) - (distance // self.endalign_bin_size)
        valid = (bin_idx >= 0) & (bin_idx < self.endalign_vec_len)
        out.scatter_add_(0, bin_idx[valid].long(), scores[valid])

        total = out.sum().clamp(min=1e-12)
        out = out / total

        log_V = float(max(1e-6, torch.log(torch.tensor(float(self.endalign_vec_len))).item()))
        p = out.clamp(min=1e-12)
        entropy = -(p * p.log()).sum() / log_V
        bin_pos_from_end = (self.endalign_vec_len - 1 - torch.arange(
            self.endalign_vec_len, device=device)).float() / max(1, self.endalign_vec_len - 1)
        mean_pos = (out * bin_pos_from_end).sum()
        var_pos = (out * (bin_pos_from_end - mean_pos) ** 2).sum().clamp(min=0)
        std_pos = var_pos.sqrt()

        if include_stats:
            return torch.cat([out, entropy.unsqueeze(0), mean_pos.unsqueeze(0), std_pos.unsqueeze(0)])
        return out

    def _build_mini_attn_features(self, input_ids: torch.Tensor, no_grad: bool = True):
        ids = input_ids.to(self._encode_device)
        with torch.no_grad():
            embeds = self.embed_tokens(ids).to(torch.float32)
        if no_grad:
            with torch.no_grad():
                scores = self.mini_attn(embeds)[0]
        else:
            scores = self.mini_attn(embeds)[0]

        hidden_pool = None
        if self.include_hidden_pool:
            with torch.no_grad():
                hp = self.input_layernorm(embeds.to(self.input_layernorm.weight.dtype))
            if self.hidden_pool_window > 0:
                w = min(self.hidden_pool_window, hp.size(1))
                hidden_pool = hp[0, -w:].mean(dim=0).to(torch.float32)
            else:
                hidden_pool = hp[0].mean(dim=0).to(torch.float32)
        return scores, hidden_pool

    # ------------------------------------------------------------------ extra views

    def _compute_extra_pre_rope(self, input_ids: torch.Tensor, agg: str) -> torch.Tensor:
        """Pre-RoPE Q·K inner product stats. agg: 'mean' | 'max' | 'z_max'."""
        dev = self._encode_device
        ids = input_ids.to(dev)
        with torch.no_grad():
            embeds = self.embed_tokens(ids).to(torch.float32)
            normed = self.input_layernorm(embeds)
            normed_cast = normed[0].to(self._qk_q_proj.weight.dtype)  # (L, d)
            q = self._qk_q_proj(normed_cast)  # (L, n_heads * head_dim)
            k = self._qk_k_proj(normed_cast)  # (L, n_kv * head_dim)

        L = ids.size(1)
        q = q.view(L, self._qk_n_heads, self._qk_head_dim).float()
        k = k.view(L, self._qk_n_kv, self._qk_head_dim).float()

        W = min(self.pre_rope_query_window, L)
        q_mean = q[-W:].mean(dim=0)  # (n_heads, head_dim)

        # Expand K for GQA: (L, n_kv, d) → (L, n_heads, d)
        k_exp = k.unsqueeze(2).expand(-1, -1, self._qk_groups, -1).reshape(
            L, self._qk_n_heads, self._qk_head_dim)

        # Per-head raw dot products: (L, n_heads)
        head_scores = (q_mean.unsqueeze(0) * k_exp).sum(-1) / (self._qk_head_dim ** 0.5)

        if agg == 'mean':
            scores = head_scores.mean(dim=-1)
        elif agg == 'max':
            scores = head_scores.max(dim=-1).values
        elif agg == 'z_max':
            mu = head_scores.mean(dim=0, keepdim=True)
            std = head_scores.std(dim=0, keepdim=True).clamp(min=1e-6)
            z = (head_scores - mu) / std
            scores = z.max(dim=-1).values
        else:
            raise ValueError(f"Unknown agg: {agg!r}")

        return self._stats_features(scores)

    def _compute_extra_snap_l0(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Post-RoPE layer-0 snap attention stats (W=pre_rope_query_window queries)."""
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        dev = self._encode_device
        ids = input_ids.to(dev)
        with torch.no_grad():
            embeds = self.embed_tokens(ids).to(torch.float32)
            normed = self.input_layernorm(embeds)
            normed_cast = normed[0].to(self._qk_q_proj.weight.dtype)
            q = self._qk_q_proj(normed_cast)
            k = self._qk_k_proj(normed_cast)

        L = ids.size(1)
        q = q.view(1, L, self._qk_n_heads, self._qk_head_dim).transpose(1, 2).float()
        k = k.view(1, L, self._qk_n_kv, self._qk_head_dim).transpose(1, 2).float()

        position_ids = torch.arange(L, device=dev).unsqueeze(0)
        cos, sin = self._qk_rotary_emb(k, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self._qk_groups)  # (1, n_heads, L, d)

        W = min(self.pre_rope_query_window, L)
        q_snap = q[:, :, -W:, :]  # (1, n_heads, W, d)

        key_pos = torch.arange(L, device=dev)
        attn = torch.matmul(q_snap, k.transpose(-2, -1)) / (self._qk_head_dim ** 0.5)
        snap_q_pos = torch.arange(L - W, L, device=dev)
        causal = key_pos.unsqueeze(0) > snap_q_pos.unsqueeze(1)  # (W, L)
        attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(attn, dim=-1)  # (1, n_heads, W, L)
        acc = attn.sum(dim=2).squeeze(0)    # (n_heads, L)
        scores = acc.mean(dim=0)             # (L,)
        return self._stats_features(scores)

    def _compute_extra_position_prior(self, seq_len: int, device) -> torch.Tensor:
        """Recency decay + sink token prior."""
        dist = (seq_len - 1) - torch.arange(seq_len, device=device, dtype=torch.float32)
        recency = torch.exp(-dist / max(1.0, seq_len * 0.1))
        sink = torch.zeros(seq_len, device=device, dtype=torch.float32)
        sink[:min(4, seq_len)] = 1.0
        scores = recency + sink
        return self._stats_features(scores)

    def _compute_extra_view(self, input_ids: torch.Tensor,
                             seq_len: int) -> torch.Tensor:
        """Dispatch to the correct extra view computation."""
        ev = self.extra_view
        if ev.startswith('pre_rope_'):
            agg = ev[len('pre_rope_'):]  # 'mean', 'max', 'z_max'
            return self._compute_extra_pre_rope(input_ids, agg)
        elif ev == 'snap_l0':
            return self._compute_extra_snap_l0(input_ids)
        elif ev == 'position_prior':
            return self._compute_extra_position_prior(seq_len, self._encode_device)
        else:
            raise ValueError(f"Unknown extra_view: {ev!r}")

    # ------------------------------------------------------------------ main encode

    def encode_context(
        self,
        text: str,
        generation_length: int,
        token_budget: int,
        metric_type: Optional[str] = None,
        task_type: Optional[str] = None,
        dataset: Optional[str] = None,
        detach: bool = True,
    ) -> torch.Tensor:
        del generation_length, token_budget
        tokenized = self.target_tokenizer(
            text, return_tensors="pt", padding=False, truncation=True,
            max_length=self.max_input_length, add_special_tokens=False,
        )
        input_ids = tokenized.input_ids.to(self._encode_device)
        seq_len = int(input_ids.size(1))
        seq_len_feature = min(float(seq_len), self.max_seq_length) / self.max_seq_length

        metric_idx = metric_type_to_index(metric_type)
        metric_one_hot = torch.zeros(self.num_metric_types,
                                     device=self._encode_device, dtype=torch.float32)
        metric_one_hot[metric_idx] = 1.0

        task_idx = task_type_to_index(task_type=task_type, dataset=dataset)
        task_one_hot = torch.zeros(self.num_task_types,
                                   device=self._encode_device, dtype=torch.float32)
        task_one_hot[task_idx] = 1.0

        scores, hidden_pool = self._build_mini_attn_features(input_ids, no_grad=detach)

        if self.feature_mode == "stats":
            mini_one = self._stats_features(scores)
        elif self.feature_mode == "endaligned":
            mini_one = self._endaligned_features(scores, plen=seq_len,
                                                  include_stats=self.endalign_include_stats)
        else:
            raise ValueError(f"unknown feature_mode {self.feature_mode}")

        if self.single_view:
            mini_feature = mini_one
        else:
            mini_feature = torch.cat([mini_one, mini_one], dim=-1)

        parts = []
        if self.include_seq_len:
            parts.append(torch.tensor([seq_len_feature],
                                       device=self._encode_device, dtype=torch.float32))
        if self.include_metric_oh:
            parts.append(metric_one_hot)
        if self.include_task_oh:
            parts.append(task_one_hot)
        if self.include_hidden_pool and hidden_pool is not None:
            parts.append(hidden_pool.to(device=self._encode_device, dtype=torch.float32))
        parts.append(mini_feature.to(device=self._encode_device, dtype=torch.float32))

        # Extra view
        if self.extra_view != 'none':
            extra_feat = self._compute_extra_view(input_ids, seq_len)
            parts.append(extra_feat.to(device=self._encode_device, dtype=torch.float32))

        features = torch.cat(parts, dim=-1)
        if detach:
            return features.detach().cpu()
        return features


__all__ = ["MiniAttnEncoder", "EXTRA_VIEW_CHOICES"]
