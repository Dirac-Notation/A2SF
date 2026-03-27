#!/usr/bin/env python3
"""
training_progress jsonl -> training_progress.png

`RL/training/trainer.py`에서 플로팅 로직을 분리하기 위한 전용 스크립트입니다.
"""

import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np


def _load_training_progress(
    jsonl_path: str,
) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float]]:
    iterations: List[int] = []
    best1_avg_rewards: List[float] = []
    best2_avg_rewards: List[float] = []
    worst1_avg_rewards: List[float] = []
    worst2_avg_rewards: List[float] = []
    total_losses: List[float] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            iterations.append(int(row["iteration"]))
            best1_avg_rewards.append(float(row["best1_avg_reward"]))
            best2_avg_rewards.append(float(row["best2_avg_reward"]))
            worst1_avg_rewards.append(float(row["worst1_avg_reward"]))
            worst2_avg_rewards.append(float(row["worst2_avg_reward"]))
            total_losses.append(float(row["total_loss"]))

    return (
        iterations,
        best1_avg_rewards,
        best2_avg_rewards,
        worst1_avg_rewards,
        worst2_avg_rewards,
        total_losses,
    )


def _smooth_chunk(vals: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    smoothed_vals = []
    epoch_idx = []
    for chunk_id, i in enumerate(range(0, len(vals), w), start=1):
        chunk_vals = vals[i : i + w]
        smoothed_vals.append(float(np.mean(chunk_vals)))
        epoch_idx.append(chunk_id)
    return np.array(epoch_idx, dtype=np.int64), np.array(smoothed_vals, dtype=np.float64)


def plot_training_progress(
    save_dir: str,
    output_path: Optional[str] = None,
    iterations_per_epoch: Optional[int] = None,
    epochs: Optional[int] = None,
) -> None:
    train_file = os.path.join(save_dir, "training_progress.jsonl")

    if output_path is None:
        output_path = os.path.join(save_dir, "training_progress.png")

    if not os.path.exists(train_file):
        print(f"[plot] training_progress.jsonl not found: {train_file}")
        return
    (
        iterations,
        best1_avg_rewards,
        best2_avg_rewards,
        worst1_avg_rewards,
        worst2_avg_rewards,
        total_losses,
    ) = _load_training_progress(train_file)
    if len(iterations) == 0:
        print("[plot] training_progress.jsonl is empty")
        return

    if iterations_per_epoch is not None:
        window_size = max(1, int(iterations_per_epoch))
    elif epochs is not None and epochs > 0:
        window_size = max(1, len(iterations) // int(epochs))
    else:
        window_size = 1

    y_best1 = np.array(best1_avg_rewards, dtype=np.float64)
    y_best2 = np.array(best2_avg_rewards, dtype=np.float64)
    y_worst1 = np.array(worst1_avg_rewards, dtype=np.float64)
    y_worst2 = np.array(worst2_avg_rewards, dtype=np.float64)
    y_loss = np.array(total_losses, dtype=np.float64)

    # matplotlib은 플로팅이 필요할 때만 import
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

    x_epoch, y_best1_s = _smooth_chunk(y_best1, window_size)
    _, y_best2_s = _smooth_chunk(y_best2, window_size)
    _, y_worst1_s = _smooth_chunk(y_worst1, window_size)
    _, y_worst2_s = _smooth_chunk(y_worst2, window_size)
    _, y_loss_s = _smooth_chunk(y_loss, window_size)

    fig, (ax1, ax3) = plt.subplots(2, 1, constrained_layout=True, figsize=(14, 10))

    ax1.plot(
        x_epoch,
        y_best1_s,
        marker="o",
        linewidth=2.5,
        markersize=5,
        color="#4C72B0",
        label="Best1 Avg Reward",
        zorder=5,
    )
    ax1.plot(
        x_epoch,
        y_best2_s,
        marker="o",
        linewidth=2.5,
        markersize=5,
        color="#64B5CD",
        label="Best2 Avg Reward",
        zorder=5,
    )
    ax1.plot(
        x_epoch,
        y_worst1_s,
        marker="o",
        linewidth=2.5,
        markersize=5,
        color="#DD8452",
        label="Worst1 Avg Reward",
        zorder=5,
    )
    ax1.plot(
        x_epoch,
        y_worst2_s,
        marker="o",
        linewidth=2.5,
        markersize=5,
        color="#C44E52",
        label="Worst2 Avg Reward",
        zorder=5,
    )
    ax1.set_title("Best/Worst Reward Curves", pad=20)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Reward")
    ax1.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
    ax1.legend(frameon=False, loc="best")
    ax1.margins(x=0.02)

    ax3.plot(
        x_epoch,
        y_loss_s,
        marker="o",
        linewidth=4,
        markersize=8,
        color="#C44E52",
        label="Total Loss (epoch average)",
        zorder=5,
    )
    ax3.set_title("Total Loss Over Training", pad=20)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Total Loss")
    ax3.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
    ax3.legend(frameon=False, loc="best")
    ax3.margins(x=0.02)

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

