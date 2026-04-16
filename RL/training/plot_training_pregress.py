#!/usr/bin/env python3
"""
training_progress jsonl -> training_progress.png

`RL/training/trainer.py`에서 플로팅 로직을 분리하기 위한 전용 스크립트입니다.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

NUM_BEST = 1
BEST_LABELS = [f"best{i+1}" for i in range(NUM_BEST)]
BEST_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:NUM_BEST]


def _load_training_progress(jsonl_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _load_epoch_mrr(jsonl_path: str) -> Tuple[List[int], List[float]]:
    epochs: List[int] = []
    mrr_values: List[float] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            epochs.append(int(row["epoch"]))
            mrr_values.append(float(row["mrr"]))
    return epochs, mrr_values


def _weighted_chunk(
    vals: np.ndarray, weights: np.ndarray, w: int
) -> Tuple[np.ndarray, np.ndarray]:
    """청크 단위 가중평균. weights=0 이면 해당 iter은 skip."""
    smoothed_vals = []
    epoch_idx = []
    for chunk_id, i in enumerate(range(0, len(vals), w), start=1):
        chunk_vals = vals[i : i + w]
        chunk_w = weights[i : i + w]
        total_w = float(np.sum(chunk_w))
        if total_w <= 0:
            smoothed_vals.append(float("nan"))
        else:
            smoothed_vals.append(float(np.sum(chunk_vals * chunk_w) / total_w))
        epoch_idx.append(chunk_id)
    return np.array(epoch_idx, dtype=np.int64), np.array(smoothed_vals, dtype=np.float64)


def _smooth_chunk(vals: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    smoothed_vals = []
    epoch_idx = []
    for chunk_id, i in enumerate(range(0, len(vals), w), start=1):
        chunk_vals = vals[i : i + w]
        smoothed_vals.append(float(np.mean(chunk_vals)))
        epoch_idx.append(chunk_id)
    return np.array(epoch_idx, dtype=np.int64), np.array(smoothed_vals, dtype=np.float64)


def _collect_overall(
    rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Optional[float], Dict[str, float]]:
    """Batch-level 요약값 (best{k}_avg_reward, total_loss, batch_size, real_best refs)."""
    best_rewards: Dict[str, List[float]] = {label: [] for label in BEST_LABELS}
    total_losses: List[float] = []
    batch_sizes: List[int] = []
    ref: Optional[float] = None
    ref_by_task: Dict[str, float] = {}
    for row in rows:
        for label in BEST_LABELS:
            best_rewards[label].append(float(row.get(f"{label}_avg_reward", 0.0)))
        total_losses.append(float(row.get("total_loss", 0.0)))
        # batch_size: prefer explicit field, fallback to legacy input_seq_lengths list
        if "batch_size" in row:
            batch_sizes.append(int(row.get("batch_size", 1)))
        else:
            seq_lens = row.get("input_seq_lengths", [])
            batch_sizes.append(int(len(seq_lens)) if isinstance(seq_lens, list) else 1)
        if ref is None and "real_best_reference_avg_reward" in row:
            ref = float(row["real_best_reference_avg_reward"])
        if not ref_by_task:
            rbt = row.get("real_best_reference_avg_by_task")
            if isinstance(rbt, dict):
                ref_by_task = {str(k): float(v) for k, v in rbt.items()}
    return (
        {k: np.array(v, dtype=np.float64) for k, v in best_rewards.items()},
        np.array(total_losses, dtype=np.float64),
        np.array(batch_sizes, dtype=np.float64),
        ref,
        ref_by_task,
    )


def _collect_per_task(
    rows: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """Task별 per-iteration reward와 샘플 수를 수집.

    우선순위:
      1. `best{k}_avg_by_task` (simplified log) + `task_counts`
      2. 구버전 fallback: `best{k}_rewards` + `task_types` per-sample
    """
    task_set = set()
    for row in rows:
        # new format
        tc = row.get("task_counts")
        if isinstance(tc, dict):
            for t in tc.keys():
                task_set.add(str(t))
            continue
        # legacy format
        for t in row.get("task_types", []) or []:
            task_set.add(str(t))
    task_types = sorted(task_set)

    n_iter = len(rows)
    per_task_avg: Dict[str, Dict[str, np.ndarray]] = {
        t: {label: np.zeros(n_iter, dtype=np.float64) for label in BEST_LABELS}
        for t in task_types
    }
    per_task_cnt: Dict[str, np.ndarray] = {
        t: np.zeros(n_iter, dtype=np.float64) for t in task_types
    }

    for i, row in enumerate(rows):
        tc = row.get("task_counts")
        if isinstance(tc, dict):
            # new simplified log
            for t in task_types:
                cnt = int(tc.get(t, 0) or 0)
                if cnt > 0:
                    per_task_cnt[t][i] = float(cnt)
            for label in BEST_LABELS:
                avg_by_task = row.get(f"{label}_avg_by_task") or {}
                if not isinstance(avg_by_task, dict):
                    continue
                for t in task_types:
                    if t in avg_by_task:
                        per_task_avg[t][label][i] = float(avg_by_task[t])
            continue

        # legacy per-sample format
        row_tasks = row.get("task_types", []) or []
        for label in BEST_LABELS:
            rewards = row.get(f"{label}_rewards")
            if not isinstance(rewards, list) or len(rewards) != len(row_tasks):
                continue
            for t in task_types:
                mask = [j for j, tt in enumerate(row_tasks) if str(tt) == t]
                if not mask:
                    continue
                vals = [float(rewards[j]) for j in mask]
                per_task_avg[t][label][i] = sum(vals) / len(vals)
                per_task_cnt[t][i] = float(len(vals))

    return task_types, per_task_avg, per_task_cnt


def plot_training_progress(
    save_dir: str,
    output_path: Optional[str] = None,
    iterations_per_epoch: Optional[int] = None,
    epochs: Optional[int] = None,
) -> None:
    train_file = os.path.join(save_dir, "training_progress.jsonl")
    epoch_metric_file = os.path.join(save_dir, "training_epoch_metrics.jsonl")

    if output_path is None:
        output_path = os.path.join(save_dir, "training_progress.png")

    if not os.path.exists(train_file):
        print(f"[plot] training_progress.jsonl not found: {train_file}")
        return

    rows = _load_training_progress(train_file)
    if not rows:
        print("[plot] training_progress.jsonl is empty")
        return

    best_rewards, y_loss, w_arr, real_best_reference_avg, real_best_by_task = _collect_overall(rows)
    task_types, per_task_avg, per_task_cnt = _collect_per_task(rows)

    if iterations_per_epoch is not None:
        window_size = max(1, int(iterations_per_epoch))
    elif epochs is not None and epochs > 0:
        window_size = max(1, len(rows) // int(epochs))
    else:
        window_size = 1

    has_epoch_mrr = os.path.exists(epoch_metric_file)
    mrr_epochs: List[int] = []
    mrr_values: List[float] = []
    if has_epoch_mrr:
        mrr_epochs, mrr_values = _load_epoch_mrr(epoch_metric_file)

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 150,
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.5,
        }
    )

    # Layout: [per-task rewards ...], loss, mrr
    num_task_axes = len(task_types)
    num_rows = num_task_axes + 2  # per-task + loss + mrr
    fig_height = max(14, 4 * num_rows)
    fig, axes = plt.subplots(num_rows, 1, constrained_layout=True, figsize=(14, fig_height))
    if num_rows == 1:
        axes = [axes]

    task_axes = axes[:num_task_axes]
    ax_loss = axes[num_task_axes]
    ax_mrr = axes[num_task_axes + 1]

    # --- Per-task rewards ---
    for ax, task in zip(task_axes, task_types):
        cnt = per_task_cnt[task]
        for label, color in zip(BEST_LABELS, BEST_COLORS):
            x_epoch, y_smoothed = _weighted_chunk(
                per_task_avg[task][label], cnt, window_size
            )
            ax.plot(
                x_epoch, y_smoothed,
                marker="o", linewidth=2.5, markersize=5, color=color,
                label=f"{label.capitalize()}", zorder=5,
            )
        task_ref = real_best_by_task.get(task)
        if task_ref is not None:
            ax.axhline(
                y=float(task_ref),
                linestyle="--", linewidth=2.0, color="#222222",
                label=f"RealBest Avg ({task_ref:.3f})", zorder=4,
            )
        ax.set_title(f"Reward Curves — {task}", pad=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Reward")
        ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.legend(frameon=False, loc="best")
        ax.margins(x=0.02)

    # --- Loss ---
    x_epoch_loss, y_loss_s = _smooth_chunk(y_loss, window_size)
    ax_loss.plot(
        x_epoch_loss, y_loss_s,
        marker="o", linewidth=2.5, markersize=5, color="#8172B3",
        label="Total Loss", zorder=5,
    )
    ax_loss.set_title("Loss Curve", pad=12)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_loss.legend(frameon=False, loc="best")
    ax_loss.margins(x=0.02)

    # --- MRR ---
    if len(mrr_epochs) > 0:
        ax_mrr.plot(
            np.array(mrr_epochs, dtype=np.int64),
            np.array(mrr_values, dtype=np.float64),
            marker="o", linewidth=2.5, markersize=5, color="#64B5CD",
            label="Epoch MRR", zorder=5,
        )
        ax_mrr.legend(frameon=False, loc="best")
    else:
        ax_mrr.text(
            0.5, 0.5, "No epoch MRR data yet",
            horizontalalignment="center", verticalalignment="center",
            transform=ax_mrr.transAxes, fontsize=16, alpha=0.8,
        )
    ax_mrr.set_title("MRR Curve", pad=12)
    ax_mrr.set_xlabel("Epoch")
    ax_mrr.set_ylabel("MRR")
    ax_mrr.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_mrr.margins(x=0.02)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training progress jsonl -> png")
    parser.add_argument("--save_dir", type=str, default="runs/a2sf_rl")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--iterations_per_epoch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    plot_training_progress(
        save_dir=args.save_dir,
        output_path=args.output_path,
        iterations_per_epoch=args.iterations_per_epoch,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
