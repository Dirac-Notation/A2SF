#!/usr/bin/env python3
"""
LongBench 일부만 샘플링해 RL 학습/평가용 jsonl 생성.

- 데이터셋별 train: 기본 10% (길이 균형 샘플링)
- 데이터셋별 eval: 남은 풀에서 기본 2% (길이 균형, train과 겹치지 않음)
- 샘플이 확정된 뒤, 평가(longbench_RL 등)와 동일하게 max_input_length 기준 중간 절단 적용
- generation_length: config/dataset2maxlen.json (데이터셋별, 평가와 동일 스케일)
"""
import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer

DEFAULT_SPLIT_SEED = 42
DEFAULT_SAMPLE_RATIO = 0.10
DEFAULT_LENGTH_BINS = 10
DEFAULT_EVAL_RATIO = 0.02

TASK2DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "task2dataset.json",
)
DATASET2MAXLEN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "dataset2maxlen.json",
)

with open(TASK2DATASET_PATH, "r", encoding="utf-8") as f:
    TASK_TO_DATASETS = json.load(f)

DATASET_TO_TASK = {
    dataset_name: task_name
    for task_name, datasets in TASK_TO_DATASETS.items()
    for dataset_name in datasets
}

with open(DATASET2MAXLEN_PATH, "r", encoding="utf-8") as f:
    DATASET_TO_MAXGEN = json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def list_longbench_files(longbench_dir: str) -> List[str]:
    if not os.path.isdir(longbench_dir):
        raise FileNotFoundError(f"LongBench directory not found: {longbench_dir}")
    return sorted(
        [
            os.path.join(longbench_dir, name)
            for name in os.listdir(longbench_dir)
            if name.endswith(".jsonl")
        ]
    )


def resolve_max_input_length(model_name: str) -> int:
    with open(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "model2maxlen.json",
        ),
        "r",
        encoding="utf-8",
    ) as f:
        model2maxlen = json.load(f)
    model_key = str(model_name).split("_")[0].lower()
    return int(model2maxlen.get(model_key, model2maxlen.get(model_name, 8192)))


def load_tokenizer(model_name: str):
    with open(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "model2path.json",
        ),
        "r",
        encoding="utf-8",
    ) as f:
        model2path = json.load(f)
    model_path = model2path[model_name]
    return AutoTokenizer.from_pretrained(model_path)


def truncate_middle_by_max_length(prompt: str, tokenizer, max_input_length: int) -> str:
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if tokenized_prompt.size(0) <= max_input_length:
        return prompt
    half = max_input_length // 2
    front = tokenized_prompt[:half]
    back = tokenized_prompt[-half:]
    return tokenizer.decode(front, skip_special_tokens=True) + tokenizer.decode(back, skip_special_tokens=True)


def evenly_spaced_indices(size: int, count: int) -> List[int]:
    if count <= 0:
        return []
    if count >= size:
        return list(range(size))
    if count == 1:
        return [size // 2]

    idxs = []
    for i in range(count):
        pos = round(i * (size - 1) / (count - 1))
        idxs.append(int(pos))
    return sorted(set(idxs))[:count]


def allocate_per_bin(bin_sizes: List[int], target_count: int, rng: random.Random) -> List[int]:
    num_bins = len(bin_sizes)
    if target_count <= 0 or num_bins == 0:
        return [0] * num_bins

    allocation = [0] * num_bins
    non_empty_bins = [i for i, s in enumerate(bin_sizes) if s > 0]
    if not non_empty_bins:
        return allocation

    if target_count < len(non_empty_bins):
        chosen_positions = evenly_spaced_indices(len(non_empty_bins), target_count)
        for pos in chosen_positions:
            allocation[non_empty_bins[pos]] = 1
        return allocation

    for i in non_empty_bins:
        allocation[i] = 1
    remaining = target_count - len(non_empty_bins)
    if remaining <= 0:
        return allocation

    capacities = [bin_sizes[i] - allocation[i] for i in range(num_bins)]
    total_capacity = sum(max(0, c) for c in capacities)
    if total_capacity <= 0:
        return allocation

    desired = [0.0] * num_bins
    for i in range(num_bins):
        if capacities[i] > 0:
            desired[i] = remaining * (capacities[i] / total_capacity)

    floors = [min(capacities[i], int(math.floor(desired[i]))) for i in range(num_bins)]
    for i in range(num_bins):
        allocation[i] += floors[i]
    remaining -= sum(floors)

    if remaining > 0:
        fractional = []
        for i in range(num_bins):
            if capacities[i] > floors[i]:
                fractional.append((desired[i] - floors[i], i))
        fractional.sort(key=lambda x: x[0], reverse=True)
        for _, i in fractional:
            if remaining == 0:
                break
            if allocation[i] < bin_sizes[i]:
                allocation[i] += 1
                remaining -= 1

    while remaining > 0:
        candidates = [i for i in range(num_bins) if allocation[i] < bin_sizes[i]]
        if not candidates:
            break
        chosen = rng.choice(candidates)
        allocation[chosen] += 1
        remaining -= 1

    return allocation


def length_balanced_sample(
    samples: List[Dict[str, Any]],
    ratio: float,
    num_bins: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if not samples:
        return []
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"sample_ratio must be in (0, 1], got {ratio}")

    total = len(samples)
    target = max(1, int(round(total * ratio)))
    target = min(target, total)

    sorted_samples = sorted(samples, key=lambda x: x["length"])
    actual_bins = max(1, min(num_bins, total))

    bins: List[List[Dict[str, Any]]] = []
    for b in range(actual_bins):
        start = (b * total) // actual_bins
        end = ((b + 1) * total) // actual_bins
        bins.append(sorted_samples[start:end])

    rng = random.Random(seed)
    allocation = allocate_per_bin([len(b) for b in bins], target, rng)

    selected: List[Dict[str, Any]] = []
    for bin_samples, count in zip(bins, allocation):
        if count <= 0:
            continue
        if count >= len(bin_samples):
            selected.extend(bin_samples)
            continue

        bin_sorted = sorted(bin_samples, key=lambda x: x["length"])
        chosen = evenly_spaced_indices(len(bin_sorted), count)
        selected.extend([bin_sorted[idx] for idx in chosen])

    if len(selected) > target:
        selected = sorted(selected, key=lambda x: x["length"])
        keep_idx = set(evenly_spaced_indices(len(selected), target))
        selected = [row for i, row in enumerate(selected) if i in keep_idx]
    elif len(selected) < target:
        selected_ids = {id(s) for s in selected}
        remainder = [s for s in sorted_samples if id(s) not in selected_ids]
        need = target - len(selected)
        add_idx = evenly_spaced_indices(len(remainder), need)
        selected.extend([remainder[i] for i in add_idx])

    return selected


def length_balanced_sample_by_count(
    samples: List[Dict[str, Any]],
    target_count: int,
    num_bins: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if not samples or target_count <= 0:
        return []

    total = len(samples)
    target = min(target_count, total)
    ratio = target / total

    sampled = length_balanced_sample(
        samples=samples,
        ratio=ratio,
        num_bins=num_bins,
        seed=seed,
    )

    if len(sampled) > target:
        sampled_sorted = sorted(sampled, key=lambda x: x["length"])
        keep_idx = set(evenly_spaced_indices(len(sampled_sorted), target))
        sampled = [row for i, row in enumerate(sampled_sorted) if i in keep_idx]
    elif len(sampled) < target:
        sampled_ids = {id(s) for s in sampled}
        remainder = [s for s in samples if id(s) not in sampled_ids]
        need = target - len(sampled)
        if remainder:
            add_idx = evenly_spaced_indices(len(remainder), min(need, len(remainder)))
            sampled.extend([remainder[i] for i in add_idx])

    return sampled


def load_longbench_sampled_splits(
    tokenizer,
    longbench_dir: str,
    sample_ratio: float,
    eval_ratio: float,
    num_length_bins: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_samples_all: List[Dict[str, Any]] = []
    eval_samples_all: List[Dict[str, Any]] = []
    dataset_files = list_longbench_files(longbench_dir)
    print(f"Found {len(dataset_files)} LongBench dataset files in {longbench_dir}")

    for file_path in dataset_files:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        task_type = DATASET_TO_TASK.get(dataset_name, "unknown")
        generation_length = int(DATASET_TO_MAXGEN.get(dataset_name, 128))

        dataset_samples: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"  Skip malformed line {line_idx} in {dataset_name}: {exc}")
                    continue

                prompt = row.get("input_prompt", "")
                if not prompt:
                    continue

                token_len = count_tokens(tokenizer, prompt)
                dataset_samples.append(
                    {
                        "input_prompt": prompt,
                        "generation_length": generation_length,
                        "dataset": dataset_name,
                        "task_type": task_type,
                        "source": "LongBench(local)",
                        "length": token_len,
                        "answers": row.get("answers", []),
                        "all_classes": row.get("all_classes", []),
                    }
                )

        if not dataset_samples:
            print(f"- {dataset_name}: no valid samples, skipped")
            continue

        train_sampled = length_balanced_sample(
            samples=dataset_samples,
            ratio=sample_ratio,
            num_bins=num_length_bins,
            seed=seed,
        )
        train_selected_ids = {id(s) for s in train_sampled}
        remaining_for_eval = [s for s in dataset_samples if id(s) not in train_selected_ids]
        if remaining_for_eval and eval_ratio > 0:
            eval_target_count = int(round(len(dataset_samples) * eval_ratio))
            eval_target_count = min(eval_target_count, len(remaining_for_eval))
            eval_sampled = length_balanced_sample_by_count(
                samples=remaining_for_eval,
                target_count=eval_target_count,
                num_bins=num_length_bins,
                seed=seed + 1,
            )
        else:
            eval_sampled = []

        train_samples_all.extend(train_sampled)
        eval_samples_all.extend(eval_sampled)

        all_lengths = [s["length"] for s in dataset_samples]
        train_lengths = [s["length"] for s in train_sampled]
        eval_lengths = [s["length"] for s in eval_sampled] if eval_sampled else []
        eval_range_str = (
            f"[{min(eval_lengths)}, {max(eval_lengths)}]"
            if eval_lengths
            else "[]"
        )
        print(
            f"- {dataset_name:20s}: total={len(dataset_samples):4d}, "
            f"train={len(train_sampled):4d} ({sample_ratio*100:.1f}%), "
            f"eval={len(eval_sampled):4d} ({eval_ratio*100:.1f}%), "
            f"len_range_all=[{min(all_lengths)}, {max(all_lengths)}], "
            f"len_range_train=[{min(train_lengths)}, {max(train_lengths)}], "
            f"len_range_eval={eval_range_str}"
        )

    return train_samples_all, eval_samples_all


def apply_eval_style_truncation(
    samples: List[Dict[str, Any]],
    tokenizer,
    max_input_length: int,
) -> None:
    """평가와 동일: 초과 시 중간 절단 후 length를 토큰 길이로 갱신."""
    for sample in samples:
        prompt = sample["input_prompt"]
        truncated = truncate_middle_by_max_length(prompt, tokenizer, max_input_length)
        sample["input_prompt"] = truncated
        sample["length"] = int(
            tokenizer(truncated, truncation=False, return_tensors="pt").input_ids.size(1)
        )


def write_jsonl(path: str, rows: List[Dict[str, Any]], split_name: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            payload = dict(row)
            payload["sample_id"] = idx
            payload["split"] = split_name
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def main(args):
    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model)
    max_input_length = int(args.max_input_length or resolve_max_input_length(args.model))

    train_samples, eval_samples = load_longbench_sampled_splits(
        tokenizer=tokenizer,
        longbench_dir=args.longbench_dir,
        sample_ratio=args.sample_ratio,
        eval_ratio=args.eval_ratio,
        num_length_bins=args.num_length_bins,
        seed=args.seed,
    )

    apply_eval_style_truncation(train_samples, tokenizer, max_input_length)
    apply_eval_style_truncation(eval_samples, tokenizer, max_input_length)

    write_jsonl(args.output_file, train_samples, "train")
    write_jsonl(args.eval_output_file, eval_samples, "eval")

    print(f"\nSaved train: {args.output_file} ({len(train_samples)} samples)")
    print(f"Saved eval:  {args.eval_output_file} ({len(eval_samples)} samples)")
    print(f"Truncation max_input_length={max_input_length} (model={args.model})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LongBench 부분 샘플링(10%/2%) + 평가 동일 절단, GT 기반 RL 학습용 jsonl"
    )
    parser.add_argument("--longbench_dir", type=str, default="./datasets/longbench")
    parser.add_argument("--output_file", type=str, default="./datasets/training_data.jsonl")
    parser.add_argument("--eval_output_file", type=str, default="./datasets/eval_data.jsonl")
    parser.add_argument("--sample_ratio", type=float, default=DEFAULT_SAMPLE_RATIO, help="Train 비율 (기본 0.10)")
    parser.add_argument("--eval_ratio", type=float, default=DEFAULT_EVAL_RATIO, help="Eval 비율 (기본 0.02, train 제외 풀에서)")
    parser.add_argument("--num_length_bins", type=int, default=DEFAULT_LENGTH_BINS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--model", type=str, default="llama3", choices=["llama", "llama2", "llama3", "opt"])
    parser.add_argument("--max_input_length", type=int, default=None)
    main(parser.parse_args())
