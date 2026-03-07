import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import rcParams

from longbench_eval import data_group


Action = Tuple[float, float]  # (a, b)


def apply_plot_style() -> None:
    rcParams.update(
        {
            "font.family": "serif",
            "figure.figsize": (14, 10),
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


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Plot task-wise action ratios from LongBench RL prediction files."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory containing prediction JSONL files (e.g., result_txt/pred/llama3_sigmoid_128_RL).",
    )
    return parser.parse_args(args)


def build_dataset_to_task_map() -> Dict[str, str]:
    dataset_to_task: Dict[str, str] = {}
    for task_name, datasets in data_group.items():
        for dataset in datasets:
            dataset_to_task[dataset] = task_name
    return dataset_to_task


def action_to_label(action: Action) -> str:
    a, b = action
    return f"a={a:.3f}\nb={b:.3f}"


def load_task_action_counts(
    output_dir: str, dataset_to_task: Dict[str, str]
) -> Tuple[Dict[str, Counter], Dict[str, int], int, int]:
    task_action_counts: Dict[str, Counter] = defaultdict(Counter)
    task_total_counts: Dict[str, int] = defaultdict(int)
    loaded_files = 0
    skipped_samples = 0

    for filename in sorted(os.listdir(output_dir)):
        if not filename.endswith(".jsonl"):
            continue

        dataset = filename.rsplit(".", 1)[0]
        if dataset not in dataset_to_task:
            continue

        task_name = dataset_to_task[dataset]
        file_path = os.path.join(output_dir, filename)
        loaded_files += 1

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "a" not in row or "b" not in row:
                    skipped_samples += 1
                    continue

                # Normalize floating point noise for stable action categories.
                action = (round(float(row["a"]), 3), round(float(row["b"]), 3))
                task_action_counts[task_name][action] += 1
                task_total_counts[task_name] += 1

    return task_action_counts, task_total_counts, loaded_files, skipped_samples


def convert_to_ratios(
    task_action_counts: Dict[str, Counter],
    task_total_counts: Dict[str, int],
) -> Dict[str, Dict[Action, float]]:
    task_action_ratios: Dict[str, Dict[Action, float]] = {}

    for task_name, action_counter in task_action_counts.items():
        total = task_total_counts[task_name]
        if total == 0:
            continue
        task_action_ratios[task_name] = {
            action: count / total for action, count in action_counter.items()
        }

    return task_action_ratios


def collect_all_actions(task_action_ratios: Dict[str, Dict[Action, float]]) -> List[Action]:
    actions = set()
    for ratios in task_action_ratios.values():
        actions.update(ratios.keys())
    return sorted(actions, key=lambda x: (x[1], x[0]))


def plot_combined_task_ratios(
    task_action_ratios: Dict[str, Dict[Action, float]],
    all_actions: List[Action],
    save_path: str,
) -> None:
    tasks_in_order = [task for task in data_group.keys() if task in task_action_ratios]
    if not tasks_in_order or not all_actions:
        return

    num_tasks = len(tasks_in_order)
    num_actions = len(all_actions)

    plt.figure(figsize=(max(14, num_actions * 1.6), 10))

    bar_width = 0.8 / num_tasks
    x_base = list(range(num_actions))
    colors = [
        "#4C72B0",  # muted blue
        "#C44E52",  # muted red
        "#55A868",  # muted green
        "#8172B3",  # muted purple
        "#CCB974",  # muted yellow-brown
        "#64B5CD",  # muted cyan
    ]

    for i, task_name in enumerate(tasks_in_order):
        x_positions = [x - 0.4 + (i + 0.5) * bar_width for x in x_base]
        y_values = [task_action_ratios[task_name].get(action, 0.0) for action in all_actions]
        plt.bar(
            x_positions,
            y_values,
            width=bar_width,
            label=task_name,
            color=colors[i % len(colors)],
            zorder=5,
        )

    plt.xlabel("Action (a, b)")
    plt.ylabel("Ratio")
    plt.title("Task-wise Action Ratio Comparison", pad=20)
    plt.xticks(x_base, [action_to_label(action) for action in all_actions], rotation=0, ha="center")
    plt.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
    plt.ylim(0, 1)
    plt.legend(frameon=False, loc="best")
    plt.margins(x=0.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_task_ratios(
    task_action_ratios: Dict[str, Dict[Action, float]],
    all_actions: List[Action],
    save_dir: str,
) -> None:
    tasks_in_order = [task for task in data_group.keys() if task in task_action_ratios]

    for task_name in tasks_in_order:
        ratios = task_action_ratios[task_name]
        y_values = [ratios.get(action, 0.0) for action in all_actions]
        x_labels = [action_to_label(action) for action in all_actions]

        safe_task_name = task_name.lower().replace(" ", "_").replace("/", "_")
        save_path = os.path.join(save_dir, f"action_ratio_{safe_task_name}.png")

        plt.figure(figsize=(max(14, len(all_actions) * 1.6), 10))
        plt.bar(range(len(all_actions)), y_values, color="#4C72B0", zorder=5)
        plt.xlabel("Action (a, b)")
        plt.ylabel("Ratio")
        plt.title(f"Action Ratio - {task_name}", pad=20)
        plt.xticks(range(len(all_actions)), x_labels, rotation=0, ha="center")
        plt.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
        plt.ylim(0, 1)
        plt.margins(x=0.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


def main():
    apply_plot_style()
    args = parse_args()

    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Output directory not found: {args.output_dir}")

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    dataset_to_task = build_dataset_to_task_map()
    (
        task_action_counts,
        task_total_counts,
        loaded_files,
        skipped_samples,
    ) = load_task_action_counts(args.output_dir, dataset_to_task)

    task_action_ratios = convert_to_ratios(task_action_counts, task_total_counts)
    all_actions = collect_all_actions(task_action_ratios)

    if loaded_files == 0:
        raise RuntimeError(f"No JSONL dataset files found in: {args.output_dir}")
    if not task_action_ratios:
        raise RuntimeError(
            "No valid RL actions found. Check whether prediction JSONL files contain 'a' and 'b' fields."
        )

    combined_path = os.path.join(save_dir, "action_ratio_all_tasks.png")
    plot_combined_task_ratios(task_action_ratios, all_actions, combined_path)
    plot_per_task_ratios(task_action_ratios, all_actions, save_dir)

    print(f"Loaded dataset files: {loaded_files}")
    print(f"Skipped samples without actions: {skipped_samples}")
    print(f"Detected unique actions: {len(all_actions)}")
    print(f"Saved plots to: {save_dir}")


if __name__ == "__main__":
    main()
