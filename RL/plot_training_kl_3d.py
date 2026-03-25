#!/usr/bin/env python3
"""
Training 데이터셋에 대해 Task별로 Prompt Length × (a,b) 액션 조합에 따른 KL Divergence를 3D로 시각화합니다.

- X: 프롬프트 토큰 길이
- Y: log10(b)  (sigmoid shift; a는 고정이면 축으로 b를 사용)
- Z: KL(P_teacher || P_student)  (sparse top-k 기준, runner와 동일)

출력: runs/kl_full_cache_viz/ 아래에 Task별 PNG + 선택적으로 전체 서브플롯 PNG
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — needed for 3D projection
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.main import A2SFRLConfig
from RL.runner import A2SFModelRunner
from longbench_eval import dataset2metric

# 사용자 지정 액션 그리드 (main.py 예시와 동일)
A_VALUES = [10.0]
B_VALUES = [1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0]


def collect_samples_by_task(
    train_path: str,
    max_total: Optional[int],
    max_per_task: Optional[int],
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    with open(train_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(seed)
    by_task_all: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row.get("task_type") or "unknown").strip() or "unknown"
        by_task_all[task].append(row)

    for task in by_task_all:
        rng.shuffle(by_task_all[task])
        if max_per_task is not None:
            by_task_all[task] = by_task_all[task][: max_per_task]

    flat: List[Dict[str, Any]] = []
    for samples in by_task_all.values():
        flat.extend(samples)
    rng.shuffle(flat)
    if max_total is not None:
        flat = flat[: max_total]

    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in flat:
        task = str(row.get("task_type") or "unknown").strip() or "unknown"
        out[task].append(row)
    return dict(out)


def compute_kl_for_sample(
    runner: A2SFModelRunner,
    prompt: str,
    generation_length: int,
    answers: List[str],
    all_classes: List[str],
    metric_type: str,
    dataset: str | None,
    a: float,
    b: float,
    token_budget: int,
) -> float:
    result = runner.run_with_compression(
        prompt=prompt,
        a=a,
        b=b,
        token_budget=token_budget,
        generation_length=generation_length,
        answers=answers,
        all_classes=all_classes,
        metric_type=metric_type,
        dataset=dataset,
    )
    return float(result.reward)


def plot_task_3d(
    task_name: str,
    points: List[Tuple[float, float, float]],
    out_path: str,
) -> None:
    """points: (prompt_len, log10(b), kl)"""
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xs, ys, zs, c=zs, cmap="viridis", s=12, alpha=0.65, depthshade=True)
    plt.colorbar(sc, ax=ax, shrink=0.6, label="KL divergence")
    ax.set_xlabel("Prompt length (tokens)", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(b)$", fontsize=12)
    ax.set_zlabel("KL divergence", fontsize=12)
    ax.set_title(f"KL vs prompt length & b — {task_name}\n(a={A_VALUES})", fontsize=13, pad=16)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all_tasks_subplots(
    task_to_points: Dict[str, List[Tuple[float, float, float]]],
    out_path: str,
) -> None:
    tasks = sorted(task_to_points.keys())
    n = len(tasks)
    if n == 0:
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    for i, task in enumerate(tasks):
        pts = task_to_points[task]
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        if not pts:
            ax.set_title(task)
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        sc = ax.scatter(xs, ys, zs, c=zs, cmap="viridis", s=8, alpha=0.5)
        ax.set_xlabel("Prompt len")
        ax.set_ylabel(r"$\log_{10}(b)$")
        ax.set_zlabel("KL")
        ax.set_title(task, fontsize=10)
    fig.suptitle(f"Training KL (3D) — a={A_VALUES}, b grid", fontsize=14, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot 3D KL vs prompt length and b per task (training jsonl).")
    p.add_argument("--train_data_path", type=str, default="datasets/training_data.jsonl", help="training_data.jsonl 경로")
    p.add_argument("--model", type=str, default="llama3", choices=["llama", "llama2", "llama3", "opt"])
    p.add_argument("--token_budget", type=int, default=128, help="압축 총 budget (RL trainer와 동일하게)")
    p.add_argument("--max_total_samples", type=int, default=None, help="전체에서 사용할 최대 샘플 수 (셔플 후 상한)")
    p.add_argument("--max_samples_per_task", type=int, default=None, help="task당 최대 샘플 수")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="runs/kl_full_cache_viz", help="PNG 저장 디렉터리")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.train_data_path
    if not os.path.isfile(train_path):
        raise FileNotFoundError(train_path)

    config = A2SFRLConfig(model=args.model)
    runner = A2SFModelRunner(config)
    tokenizer = runner.tokenizer

    by_task = collect_samples_by_task(
        train_path,
        max_total=args.max_total_samples,
        max_per_task=args.max_samples_per_task,
        seed=args.seed,
    )

    task_to_points: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)

    n_samples = sum(len(s) for s in by_task.values())
    total_calls = n_samples * len(A_VALUES) * len(B_VALUES)
    pbar = tqdm(total=total_calls, desc="KL compute")

    for task_name, samples in by_task.items():
        for sample in samples:
            prompt = sample.get("input_prompt", "")
            if not prompt:
                pbar.update(len(A_VALUES) * len(B_VALUES))
                continue

            enc = tokenizer(prompt, truncation=False, return_tensors="pt")
            prompt_len = float(enc.input_ids.size(1))
            dataset = sample.get("dataset")
            generation_length = int(sample.get("generation_length", 128))
            answers = sample.get("answers", [])
            all_classes = sample.get("all_classes", [])
            metric_fn = dataset2metric.get(str(dataset or "").lower())
            metric_type = metric_fn.__name__ if metric_fn is not None else "qa_f1_score"

            for a in A_VALUES:
                for b in B_VALUES:
                    try:
                        kl = compute_kl_for_sample(
                            runner=runner,
                            prompt=prompt,
                            generation_length=generation_length,
                            answers=answers,
                            all_classes=all_classes,
                            metric_type=metric_type,
                            dataset=dataset,
                            a=a,
                            b=b,
                            token_budget=args.token_budget,
                        )
                        task_to_points[task_name].append(
                            (prompt_len, float(np.log10(max(b, 1e-12))), kl)
                        )
                    except Exception as e:
                        print(f"[warn] task={task_name} a={a} b={b}: {e}")
                    pbar.update(1)

    pbar.close()

    out_dir = args.output_dir
    for task_name, pts in task_to_points.items():
        safe = task_name.replace("/", "-").replace(" ", "_")
        plot_task_3d(task_name, pts, os.path.join(out_dir, f"training_kl_3d_{safe}.png"))

    plot_all_tasks_subplots(task_to_points, os.path.join(out_dir, "training_kl_3d_all_tasks.png"))

    print(f"Saved plots under: {out_dir}")


if __name__ == "__main__":
    main()
