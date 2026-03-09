#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_last_iteration_record(jsonl_path: Path) -> Dict[str, Any]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    records: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise ValueError(f"No valid records in: {jsonl_path}")

    return max(records, key=lambda x: int(x.get("iteration", -1)))


def parse_actions_a(record: Dict[str, Any]) -> np.ndarray:
    actions = record.get("eval_actions_a")
    if not isinstance(actions, list) or len(actions) == 0:
        raise ValueError("Record does not contain non-empty 'eval_actions_a'.")

    try:
        return np.array([float(v) for v in actions], dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Failed to parse eval_actions_a values: {e}") from e


def make_histogram(actions_a: np.ndarray, iteration: int, output_path: Path) -> None:
    unique_vals, counts = np.unique(actions_a, return_counts=True)
    ratios = counts / counts.sum()
    x = np.arange(len(unique_vals))
    
    plt.figure(figsize=(8, 5))
    plt.bar(x, ratios, width=0.7, edgecolor="black")
    plt.xticks(x, [f"{v:.4f}" for v in unique_vals], rotation=45, ha="right")
    plt.ylim(0.0, max(0.05, float(ratios.max()) * 1.15))

    plt.title(f"eval_actions_a Ratio (last iteration={iteration})")
    plt.xlabel("action_a")
    plt.ylabel("ratio")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw action_a ratio bar chart for the last iteration."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="runs/a2sf_rl/evaluation_progress.jsonl",
        help="Path to evaluation_progress.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/a2sf_rl/eval_actions_a_hist_last.png",
        help="Output image path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    last_record = load_last_iteration_record(input_path)
    iteration = int(last_record.get("iteration", -1))
    actions_a = parse_actions_a(last_record)

    make_histogram(actions_a, iteration, output_path)
    print(f"Saved bar chart: {output_path}")
    print(f"Last iteration: {iteration}, samples: {len(actions_a)}")


if __name__ == "__main__":
    main()
