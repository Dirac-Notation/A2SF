#!/usr/bin/env python3
"""
training_progress jsonl -> training_progress.png

`RL/training/trainer.py`에서 플로팅 로직을 분리하기 위한 전용 스크립트입니다.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

NUM_BEST = 4
BEST_LABELS = [f"best{i+1}" for i in range(NUM_BEST)]
BEST_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def _load_training_progress(
    jsonl_path: str,
) -> Tuple[List[int], Dict[str, List[float]], List[float], List[int], Optional[float]]:
    iterations: List[int] = []
    best_rewards: Dict[str, List[float]] = {label: [] for label in BEST_LABELS}
    total_losses: List[float] = []
    batch_sizes: List[int] = []
    real_best_reference_avg: Optional[float] = None

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            iterations.append(int(row["iteration"]))
            for label in BEST_LABELS:
                key = f"{label}_avg_reward"
                best_rewards[label].append(float(row.get(key, 0.0)))
            total_losses.append(float(row["total_loss"]))
            # batch size는 input_seq_lengths의 길이
            seq_lens = row.get("input_seq_lengths", [])
            batch_sizes.append(int(len(seq_lens)) if isinstance(seq_lens, list) else 1)
            if real_best_reference_avg is None and "real_best_reference_avg_reward" in row:
                real_best_reference_avg = float(row["real_best_reference_avg_reward"])

    return iterations, best_rewards, total_losses, batch_sizes, real_best_reference_avg


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


def _smooth_chunk(vals: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    smoothed_vals = []
    epoch_idx = []
    for chunk_id, i in enumerate(range(0, len(vals), w), start=1):
        chunk_vals = vals[i : i + w]
        smoothed_vals.append(float(np.mean(chunk_vals)))
        epoch_idx.append(chunk_id)
    return np.array(epoch_idx, dtype=np.int64), np.array(smoothed_vals, dtype=np.float64)


def _weighted_chunk(
    vals: np.ndarray, weights: np.ndarray, w: int
) -> Tuple[np.ndarray, np.ndarray]:
    """청크 단위 가중평균. 각 iteration의 batch_size로 가중."""
    smoothed_vals = []
    epoch_idx = []
    for chunk_id, i in enumerate(range(0, len(vals), w), start=1):
        chunk_vals = vals[i : i + w]
        chunk_w = weights[i : i + w]
        total_w = float(np.sum(chunk_w))
        if total_w <= 0:
            smoothed_vals.append(float(np.mean(chunk_vals)))
        else:
            smoothed_vals.append(float(np.sum(chunk_vals * chunk_w) / total_w))
        epoch_idx.append(chunk_id)
    return np.array(epoch_idx, dtype=np.int64), np.array(smoothed_vals, dtype=np.float64)


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

    iterations, best_rewards, total_losses, batch_sizes, real_best_reference_avg = _load_training_progress(train_file)
    if len(iterations) == 0:
        print("[plot] training_progress.jsonl is empty")
        return

    if iterations_per_epoch is not None:
        window_size = max(1, int(iterations_per_epoch))
    elif epochs is not None and epochs > 0:
        window_size = max(1, len(iterations) // int(epochs))
    else:
        window_size = 1

    y_loss = np.array(total_losses, dtype=np.float64)
    w_arr = np.array(batch_sizes, dtype=np.float64)
    has_epoch_mrr = os.path.exists(epoch_metric_file)
    mrr_epochs: List[int] = []
    mrr_values: List[float] = []
    if has_epoch_mrr:
        mrr_epochs, mrr_values = _load_epoch_mrr(epoch_metric_file)

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "figure.figsize": (14, 14),
            "figure.dpi": 150,
            "font.size": 22,
            "axes.labelsize": 26,
            "axes.titlesize": 28,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.5,
        }
    )

    fig, (ax_reward, ax_loss, ax_mrr) = plt.subplots(
        3, 1, constrained_layout=True, figsize=(14, 16)
    )

    for label, color in zip(BEST_LABELS, BEST_COLORS):
        y_arr = np.array(best_rewards[label], dtype=np.float64)
        x_epoch, y_smoothed = _weighted_chunk(y_arr, w_arr, window_size)
        ax_reward.plot(
            x_epoch,
            y_smoothed,
            marker="o",
            linewidth=3,
            markersize=6,
            color=color,
            label=f"{label.capitalize()} Avg Reward",
            zorder=5,
        )

    if real_best_reference_avg is not None:
        ax_reward.axhline(
            y=float(real_best_reference_avg),
            linestyle="--",
            linewidth=2.5,
            color="#222222",
            label="RealBest Avg (Global)",
            zorder=4,
        )

    ax_reward.set_title("Reward Curves (Best1~4)", pad=20)
    ax_reward.set_xlabel("Epoch")
    ax_reward.set_ylabel("Reward")
    ax_reward.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_reward.legend(frameon=False, loc="best")
    ax_reward.margins(x=0.02)

    x_epoch_loss, y_loss_s = _smooth_chunk(y_loss, window_size)
    ax_loss.plot(
        x_epoch_loss,
        y_loss_s,
        marker="o",
        linewidth=3,
        markersize=6,
        color="#8172B3",
        label="Total Loss",
        zorder=5,
    )
    ax_loss.set_title("Loss Curve", pad=20)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_loss.legend(frameon=False, loc="best")
    ax_loss.margins(x=0.02)

    if len(mrr_epochs) > 0:
        ax_mrr.plot(
            np.array(mrr_epochs, dtype=np.int64),
            np.array(mrr_values, dtype=np.float64),
            marker="o",
            linewidth=3,
            markersize=6,
            color="#64B5CD",
            label="Epoch MRR",
            zorder=5,
        )
        ax_mrr.legend(frameon=False, loc="best")
    else:
        ax_mrr.text(
            0.5,
            0.5,
            "No epoch MRR data yet",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax_mrr.transAxes,
            fontsize=18,
            alpha=0.8,
        )
    ax_mrr.set_title("MRR Curve", pad=20)
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
