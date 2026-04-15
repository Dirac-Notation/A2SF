#!/usr/bin/env python3
"""RL agent 체크포인트(.pt)를 분석해서 내부 구조를 출력한다.

모델 구조가 여러 번 바뀐 상황에서, 주어진 체크포인트가 어떤 버전인지
(어떤 state layout / 어떤 head 구성 / 어떤 feature_dim 등) 한눈에 확인하기 위한 도구.

사용법:
    python script/inspect_checkpoint.py path/to/policy.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Repo root를 import path에 추가 (체크포인트에 pickle된 ModelConfig 등 복원을 위해).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch


# --- 헬퍼 --------------------------------------------------------------------


def _bytes_of(t: torch.Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def _fmt_bytes(nbytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    val = float(nbytes)
    for u in units:
        if val < 1024:
            return f"{val:.2f} {u}"
        val /= 1024
    return f"{val:.2f} TB"


def _get_shape(sd: Dict[str, torch.Tensor], key: str) -> Optional[Tuple[int, ...]]:
    t = sd.get(key)
    return tuple(t.shape) if isinstance(t, torch.Tensor) else None


def _summarize_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    total_params = 0
    total_buffers = 0
    by_prefix: Dict[str, int] = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        nb = _bytes_of(v)
        # 대략적인 param/buffer 구분: grad 정보 없이 키 이름으로 추정
        if k.startswith(("inverse_lambdas", "action_counts", "a_values", "b_values")):
            total_buffers += nb
        else:
            total_params += nb
        prefix = k.split(".")[0]
        by_prefix[prefix] = by_prefix.get(prefix, 0) + nb
    return {
        "total_params_bytes": total_params,
        "total_buffers_bytes": total_buffers,
        "by_prefix": by_prefix,
    }


# --- 구조 추론 ---------------------------------------------------------------


def _infer_architecture(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """state_dict에서 텐서 shape을 보고 architecture 파라미터를 추론한다."""
    info: Dict[str, Any] = {}

    # meta_embed.0.weight: (256, meta_in) -> meta_in = 1 + M (metric one-hot) or 1 + T (task one-hot) or 2 (legacy: seq_len + token_budget)
    meta_w = _get_shape(sd, "meta_embed.0.weight")
    if meta_w is not None:
        info["meta_embed_in"] = meta_w[1]
        info["meta_embed_hidden"] = meta_w[0]
    # tova_embed.0.weight: (256, 4H) -> H = in/4
    tova_w = _get_shape(sd, "tova_embed.0.weight")
    if tova_w is not None:
        info["tova_embed_in"] = tova_w[1]
        info["num_heads_inferred"] = int(tova_w[1]) // 4
    # snap_embed.0.weight: same as tova
    snap_w = _get_shape(sd, "snap_embed.0.weight")
    if snap_w is not None:
        info["snap_embed_in"] = snap_w[1]

    # 최종 feature_dim: meta_embed.2.weight = (feature_dim, 256)
    feat_w = _get_shape(sd, "meta_embed.2.weight")
    if feat_w is not None:
        info["feature_dim"] = feat_w[0]

    # backbone 크기: backbone.0.fc1.weight = (hidden_dim, feature_dim)
    bb_w = _get_shape(sd, "backbone.0.fc1.weight")
    if bb_w is not None:
        info["backbone_hidden"] = bb_w[0]
        info["backbone_blocks"] = sum(
            1 for k in sd.keys() if k.startswith("backbone.") and k.endswith(".fc1.weight")
        )

    # reward_heads.<metric>.weight: (num_actions, feature_dim)
    metric_names: List[str] = []
    num_actions: Optional[int] = None
    for k, v in sd.items():
        if k.startswith("reward_heads.") and k.endswith(".weight"):
            metric_names.append(k.split(".")[1])
            num_actions = int(v.shape[0])
    info["metric_heads"] = sorted(metric_names)
    info["num_metric_heads"] = len(metric_names)
    if num_actions is not None:
        info["num_actions"] = num_actions

    # sigmoid_slopes 존재 여부
    slope_keys = [k for k in sd.keys() if k.startswith("sigmoid_slopes.")]
    if slope_keys:
        slopes: Dict[str, float] = {}
        for k in slope_keys:
            name = k.split(".", 1)[1]
            slopes[name] = float(sd[k].item()) if isinstance(sd[k], torch.Tensor) else float(sd[k])
        info["sigmoid_slopes"] = slopes

    # a_values, b_values
    a = _get_shape(sd, "a_values")
    b = _get_shape(sd, "b_values")
    if a is not None:
        info["num_a_values"] = a[0]
        av = sd["a_values"].detach().cpu().tolist()
        info["a_values_preview"] = av if len(av) <= 8 else av[:4] + ["..."] + av[-2:]
    if b is not None:
        info["num_b_values"] = b[0]
        bv = sd["b_values"].detach().cpu().tolist()
        info["b_values_preview"] = bv if len(bv) <= 8 else bv[:4] + ["..."] + bv[-2:]

    # UCB covariance buffer 존재 여부
    inv_shape = _get_shape(sd, "inverse_lambdas")
    if inv_shape is not None:
        info["inverse_lambdas_shape"] = inv_shape
    # 그 외 버퍼
    ac_shape = _get_shape(sd, "action_counts")
    if ac_shape is not None:
        info["action_counts_shape"] = ac_shape

    return info


def _state_layout_guess(meta_in: Optional[int], num_metric_heads: int) -> str:
    """meta_embed 입력 차원으로부터 state layout 유형 추정."""
    if meta_in is None:
        return "unknown"
    # 후보:
    #   2 -> legacy: [seq_len, token_budget]
    #   1 + T (T=7) = 8 -> [seq_len, task_type_one_hot(7)]
    #   1 + M (M=10) = 11 -> [seq_len, metric_type_one_hot(10)]
    if meta_in == 2:
        return "legacy: [seq_len, token_budget]"
    if meta_in == 8:
        return "task_type version: [seq_len, task_type_one_hot(7)]"
    if meta_in == 11:
        return "metric_type version: [seq_len, metric_type_one_hot(10)]"
    return f"custom meta_in={meta_in}"


# --- 출력 --------------------------------------------------------------------


def print_report(path: str, ckpt: Dict[str, Any]) -> None:
    print("=" * 72)
    print(f"Checkpoint: {path}")
    print(f"File size : {_fmt_bytes(os.path.getsize(path))}")
    print("=" * 72)

    top_keys = sorted(list(ckpt.keys()))
    print(f"\n[Top-level keys] ({len(top_keys)})")
    for k in top_keys:
        v = ckpt[k]
        if isinstance(v, dict):
            print(f"  - {k}: dict({len(v)} entries)")
        elif isinstance(v, torch.Tensor):
            print(f"  - {k}: Tensor{tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  - {k}: {type(v).__name__}")

    # checkpoint type 판별
    has_opt = "optimizer_state_dict" in ckpt
    has_sched = "scheduler_state_dict" in ckpt
    has_arch = "arch_config" in ckpt
    if has_arch and not has_opt:
        ckpt_type = "FINAL (inference-only, slim)"
    elif has_opt and has_sched:
        ckpt_type = "PERIODIC (resume-capable)"
    else:
        ckpt_type = "LEGACY / partial"
    print(f"\n[Checkpoint type] {ckpt_type}")

    iteration = ckpt.get("iteration")
    epoch = ckpt.get("epoch")
    if iteration is not None:
        print(f"  iteration = {iteration}")
    if epoch is not None:
        print(f"  epoch     = {epoch}")

    # arch_config 우선
    arch_cfg = ckpt.get("arch_config")
    if isinstance(arch_cfg, dict):
        print("\n[arch_config] (saved)")
        for k, v in arch_cfg.items():
            if isinstance(v, torch.Tensor):
                preview = v.detach().cpu().tolist()
                if isinstance(preview, list) and len(preview) > 8:
                    preview = preview[:4] + ["..."] + preview[-2:]
                print(f"  {k} = Tensor{tuple(v.shape)} {preview}")
            else:
                print(f"  {k} = {v}")

    # state_dict 분석
    sd = ckpt.get("agent_state_dict") or ckpt.get("policy_state_dict")
    if not isinstance(sd, dict) or not sd:
        print("\n[WARN] agent_state_dict/policy_state_dict 없음 — 구조 추론 불가")
        return

    arch_info = _infer_architecture(sd)
    sd_summary = _summarize_state_dict(sd)

    print("\n[Inferred architecture from state_dict]")
    print(f"  state layout   : {_state_layout_guess(arch_info.get('meta_embed_in'), arch_info.get('num_metric_heads', 0))}")
    print(f"  meta_embed_in  : {arch_info.get('meta_embed_in')}")
    print(f"  tova/snap_in   : {arch_info.get('tova_embed_in')} (num_heads ≈ {arch_info.get('num_heads_inferred')})")
    print(f"  feature_dim    : {arch_info.get('feature_dim')}")
    print(f"  backbone       : {arch_info.get('backbone_blocks')} blocks, hidden={arch_info.get('backbone_hidden')}")
    print(f"  num_actions    : {arch_info.get('num_actions')}")
    if "num_a_values" in arch_info:
        print(f"  a_values ({arch_info['num_a_values']}): {arch_info.get('a_values_preview')}")
    if "num_b_values" in arch_info:
        print(f"  b_values ({arch_info['num_b_values']}): {arch_info.get('b_values_preview')}")
    heads = arch_info.get("metric_heads", [])
    print(f"  metric heads ({len(heads)}): {heads}")
    if "sigmoid_slopes" in arch_info:
        print(f"  sigmoid_slopes : (per-metric, learnable)")
        for n, v in sorted(arch_info["sigmoid_slopes"].items()):
            print(f"    {n}: {v:.3f}")
    else:
        print(f"  sigmoid_slopes : (none — uses plain sigmoid)")

    # 버퍼 존재 여부 = 학습/추론 겸용 판별
    if "inverse_lambdas_shape" in arch_info:
        shp = arch_info["inverse_lambdas_shape"]
        nbytes = 1
        for d in shp:
            nbytes *= int(d)
        print(f"  inverse_lambdas: {shp} (~{_fmt_bytes(nbytes * 4)}) -> training/resume checkpoint")
    else:
        print(f"  inverse_lambdas: not present -> inference-only checkpoint")

    print("\n[State dict byte breakdown]")
    print(f"  params-ish (excl. UCB bufs) : {_fmt_bytes(sd_summary['total_params_bytes'])}")
    print(f"  buffers (UCB, a/b, counts)  : {_fmt_bytes(sd_summary['total_buffers_bytes'])}")
    print(f"  top-level prefix sizes:")
    for prefix, nb in sorted(sd_summary["by_prefix"].items(), key=lambda x: -x[1]):
        print(f"    {prefix:<20s} {_fmt_bytes(nb)}")

    # 구버전/신버전 판정 요약
    meta_in = arch_info.get("meta_embed_in")
    has_slopes = "sigmoid_slopes" in arch_info
    print("\n[Version summary]")
    if meta_in == 2:
        print("  → Legacy checkpoint (state: [seq_len, token_budget])")
    elif meta_in == 8:
        print("  → Task-type-conditioned version (state includes task_type one-hot)")
    elif meta_in == 11:
        print("  → Metric-type-conditioned version (state includes metric_type one-hot)")
    else:
        print(f"  → Unknown variant (meta_embed_in={meta_in})")
    if has_slopes:
        print("  → Has per-metric learnable sigmoid slopes")
    print("=" * 72)


# --- main --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL agent 체크포인트 구조 분석기"
    )
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        print(f"[WARN] checkpoint는 dict가 아님: {type(ckpt)}")
        return
    print_report(args.checkpoint, ckpt)


if __name__ == "__main__":
    main()
