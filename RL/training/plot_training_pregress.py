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
) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float], Optional[float]]:
    iterations: List[int] = []
    best1_avg_rewards: List[float] = []
    best2_avg_rewards: List[float] = []
    worst1_avg_rewards: List[float] = []
    worst2_avg_rewards: List[float] = []
    total_losses: List[float] = []
    real_best_reference_avg: Optional[float] = None

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
            if real_best_reference_avg is None and "real_best_reference_avg_reward" in row:
                real_best_reference_avg = float(row["real_best_reference_avg_reward"])

    return (
        iterations,
        best1_avg_rewards,
        best2_avg_rewards,
        worst1_avg_rewards,
        worst2_avg_rewards,
        total_losses,
        real_best_reference_avg,
    )


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
    (
        iterations,
        best1_avg_rewards,
        best2_avg_rewards,
        worst1_avg_rewards,
        worst2_avg_rewards,
        total_losses,
        real_best_reference_avg,
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

    y_best1_reward = np.array(best1_avg_rewards, dtype=np.float64)
    y_best2_reward = np.array(best2_avg_rewards, dtype=np.float64)
    y_worst1_reward = np.array(worst1_avg_rewards, dtype=np.float64)
    y_worst2_reward = np.array(worst2_avg_rewards, dtype=np.float64)
    y_loss = np.array(total_losses, dtype=np.float64)
    has_epoch_mrr = os.path.exists(epoch_metric_file)
    mrr_epochs: List[int] = []
    mrr_values: List[float] = []
    if has_epoch_mrr:
        mrr_epochs, mrr_values = _load_epoch_mrr(epoch_metric_file)

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

    x_epoch, y_best1_reward_s = _smooth_chunk(y_best1_reward, window_size)
    x_epoch_b2, y_best2_reward_s = _smooth_chunk(y_best2_reward, window_size)
    x_epoch_w1, y_worst1_reward_s = _smooth_chunk(y_worst1_reward, window_size)
    x_epoch_w2, y_worst2_reward_s = _smooth_chunk(y_worst2_reward, window_size)
    _, y_loss_s = _smooth_chunk(y_loss, window_size)

    fig, (ax_reward, ax_loss, ax_mrr) = plt.subplots(
        3, 1, constrained_layout=True, figsize=(14, 16)
    )

    ax_reward.plot(
        x_epoch,
        y_best1_reward_s,
        marker="o",
        linewidth=3,
        markersize=6,
        color="#4C72B0",
        label="Best1 Avg Reward",
        zorder=5,
    )
    ax_reward.plot(
        x_epoch_b2,
        y_best2_reward_s,
        marker="o",
        linewidth=3,
        markersize=6,
        color="#DD8452",
        label="Best2 Avg Reward",
        zorder=5,
    )
    ax_reward.plot(
        x_epoch_w2,
        y_worst2_reward_s,
        marker="o",
        linewidth=3,
        markersize=6,
        color="#C44E52",
        label="Worst2 Avg Reward",
        zorder=5,
    )
    ax_reward.plot(
        x_epoch_w1,
        y_worst1_reward_s,
        marker="o",
        linewidth=3,
        markersize=6,
        color="#55A868",
        label="Worst1 Avg Reward",
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

    ax_reward.set_title("Reward Curves (Best/Worst)", pad=20)
    ax_reward.set_xlabel("Epoch")
    ax_reward.set_ylabel("Reward")
    ax_reward.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_reward.legend(frameon=False, loc="best")
    ax_reward.margins(x=0.02)

    ax_loss.plot(
        x_epoch,
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
