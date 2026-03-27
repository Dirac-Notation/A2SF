#!/usr/bin/env python3
"""
LongBench 일부만 샘플링해 RL 학습용 jsonl 생성.

이 코드는 기존 `datasets/make_training_dataset.py`의 내용을 그대로
`RL/training/data_generation/` 위치로 이식해, 학습 파이프라인이
datasets 쪽을 호출하지 않도록 구성했습니다.

- 데이터셋별 train: 기본 10% (길이 균형 샘플링)
- 샘플이 확정된 뒤, 평가(longbench_RL 등)와 동일하게 max_input_length 기준 중간 절단 적용
- generation_length: config/dataset2maxlen.json (데이터셋별, 평가와 동일 스케일)
"""

import argparse
import json
import math
import os
import random
import multiprocessing as mp
import sys
import time
from typing import Any, Dict, List

# Silence HF tokenizers fork-parallelism warning in multiprocess workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer

# Ensure repository root is importable when executed as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from utils import CompressionConfig, load_model
from RL.a2sf_model import ModelConfig

DEFAULT_SPLIT_SEED = 42
DEFAULT_SAMPLE_RATIO = 0.10
DEFAULT_LENGTH_BINS = 10

# RL/training/data_generation/make_training_dataset.py 기준으로 repo root로 이동

TASK2DATASET_PATH = os.path.join(REPO_ROOT, "config", "task2dataset.json")
DATASET2MAXLEN_PATH = os.path.join(REPO_ROOT, "config", "dataset2maxlen.json")

with open(TASK2DATASET_PATH, "r", encoding="utf-8") as f:
    TASK_TO_DATASETS = json.load(f)

DATASET_TO_TASK = {dataset_name: task_name for task_name, datasets in TASK_TO_DATASETS.items() for dataset_name in datasets}

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
    return sorted([os.path.join(longbench_dir, name) for name in os.listdir(longbench_dir) if name.endswith(".jsonl")])


def resolve_max_input_length(model_name: str) -> int:
    with open(os.path.join(REPO_ROOT, "config", "model2maxlen.json"), "r", encoding="utf-8") as f:
        model2maxlen = json.load(f)
    model_key = str(model_name).split("_")[0].lower()
    return int(model2maxlen.get(model_key, model2maxlen.get(model_name, 8192)))


def load_tokenizer(model_name: str):
    with open(os.path.join(REPO_ROOT, "config", "model2path.json"), "r", encoding="utf-8") as f:
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


def load_longbench_sampled_splits(
    tokenizer,
    longbench_dir: str,
    sample_ratio: float,
    num_length_bins: int,
    seed: int,
) -> List[Dict[str, Any]]:
    train_samples_all: List[Dict[str, Any]] = []
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
        train_samples_all.extend(train_sampled)

        all_lengths = [s["length"] for s in dataset_samples]
        train_lengths = [s["length"] for s in train_sampled]

        print(
            f"- {dataset_name:20s}: total={len(dataset_samples):4d}, "
            f"train={len(train_sampled):4d} ({sample_ratio*100:.1f}%), "
            f"len_range_all=[{min(all_lengths)}, {max(all_lengths)}], "
            f"len_range_train=[{min(train_lengths)}, {max(train_lengths)}]"
        )

    return train_samples_all


def apply_eval_style_truncation(samples: List[Dict[str, Any]], tokenizer, max_input_length: int) -> None:
    """평가와 동일: 초과 시 중간 절단 후 length를 토큰 길이로 갱신."""
    for sample in samples:
        prompt = sample["input_prompt"]
        truncated = truncate_middle_by_max_length(prompt, tokenizer, max_input_length)
        sample["input_prompt"] = truncated
        sample["length"] = int(tokenizer(truncated, truncation=False, return_tensors="pt").input_ids.size(1))


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


def _format_prompt_like_longbench(prompt: str, dataset_name: str, model_name: str) -> str:
    # Keep prompt wrapping behavior identical to longbench.py
    if str(dataset_name or "").strip().lower() not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        if "llama" in str(model_name).lower():
            return f"[INST]{prompt}[/INST]"
    return prompt


def _build_generation_kwargs(
    tokenizer,
    dataset_name: str,
    generation_length: int,
    context_length: int,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "tokenizer": tokenizer,
        "stop_strings": "[/INST]",
        "max_new_tokens": int(generation_length),
        "num_beams": 1,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "num_logits_to_keep": 1,
    }
    if str(dataset_name or "").strip().lower() == "samsum":
        kwargs["min_length"] = int(context_length) + 1
        kwargs["eos_token_id"] = [
            tokenizer.eos_token_id,
            tokenizer.encode("\n", add_special_tokens=False)[-1],
        ]
    return kwargs


def _build_compression_config(a_vals: torch.Tensor, b_vals: torch.Tensor, token_budget: int) -> CompressionConfig:
    config = CompressionConfig()
    config.compression_method = "sigmoid"
    config.total_budget = int(token_budget)
    config.local_ratios = 0.125
    config.a = a_vals
    config.b = b_vals
    return config


def _offline_generate_for_sample(
    model,
    tokenizer,
    sample: Dict[str, Any],
    model_name: str,
    a_values: torch.Tensor,
    b_values: torch.Tensor,
    token_budget: int,
) -> List[str]:
    dataset_name = str(sample.get("dataset") or "")
    prompt = _format_prompt_like_longbench(
        prompt=str(sample["input_prompt"]),
        dataset_name=dataset_name,
        model_name=model_name,
    )
    encoded = tokenizer(prompt, truncation=False, return_tensors="pt")
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(torch.bfloat16).to(model.device)
    context_length = int(input_ids.shape[-1])
    generation_kwargs = _build_generation_kwargs(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        generation_length=int(sample.get("generation_length", 64)),
        context_length=context_length,
    )
    pred_texts: List[str] = []
    for idx in range(int(a_values.numel())):
        a = a_values[idx : idx + 1]
        b = b_values[idx : idx + 1]
        model.init_cache(_build_compression_config(a, b, token_budget))
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        pred_texts.append(
            tokenizer.decode(output_ids[0, context_length:], skip_special_tokens=True)
        )
    return pred_texts


def _offline_worker(
    worker_id: int,
    gpu_group: List[int],
    model_name: str,
    token_budget: int,
    task_queue: mp.Queue,
    a_values: List[float],
    b_values: List[float],
    result_queue: mp.Queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_group)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_grad_enabled(False)
    print(f"[offline-cache][worker {worker_id}] start with CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    model, tokenizer = load_model(model_name)
    print(f"[offline-cache][worker {worker_id}] model device={model.device}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    a_tensor = torch.tensor(a_values, dtype=torch.float32, device=model.device)
    b_tensor = torch.tensor(b_values, dtype=torch.float32, device=model.device)
    try:
        while True:
            sample = task_queue.get()
            if sample is None:
                break
            pred_texts = _offline_generate_for_sample(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                model_name=model_name,
                a_values=a_tensor,
                b_values=b_tensor,
                token_budget=token_budget,
            )
            result_queue.put((int(sample["sample_id"]), pred_texts))
    except Exception as exc:  # pragma: no cover
        result_queue.put(("__error__", f"worker {worker_id} gpus={gpu_group}: {exc}"))


def build_offline_action_cache(
    samples: List[Dict[str, Any]],
    model_name: str,
    token_budget: int,
    visible_gpu_count: int,
    gpus_per_model: int,
) -> None:
    if visible_gpu_count <= 0:
        raise ValueError("visible_gpu_count must be >= 1")
    if gpus_per_model <= 0:
        raise ValueError("gpus_per_model must be >= 1")
    if gpus_per_model > visible_gpu_count:
        raise ValueError(
            f"gpus_per_model ({gpus_per_model}) cannot exceed visible GPU count ({visible_gpu_count})"
        )
    cfg = ModelConfig(model=model_name)
    a_values = [float(v) for v in cfg.a_values.tolist()]
    b_values = [float(v) for v in cfg.b_values.tolist()]
    action_size = len(a_values) * len(b_values)
    cart_a: List[float] = []
    cart_b: List[float] = []
    for a in a_values:
        for b in b_values:
            cart_a.append(a)
            cart_b.append(b)

    gpu_groups: List[List[int]] = []
    all_gpu_ids = list(range(visible_gpu_count))
    for start in range(0, visible_gpu_count, gpus_per_model):
        gpu_groups.append(all_gpu_ids[start : start + gpus_per_model])

    ctx = mp.get_context("spawn")

    task_queue: mp.Queue = ctx.Queue()
    for sample in samples:
        task_queue.put(sample)
    for _ in range(len(gpu_groups)):
        task_queue.put(None)
    result_queue: mp.Queue = ctx.Queue()
    processes: List[mp.Process] = []
    for worker_id, gpu_group in enumerate(gpu_groups):
        p = ctx.Process(
            target=_offline_worker,
            args=(
                worker_id,
                gpu_group,
                model_name,
                token_budget,
                task_queue,
                cart_a,
                cart_b,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    remaining = len(samples)
    total = len(samples)
    done = 0
    started_at = time.time()
    pred_by_sample: Dict[int, List[str]] = {}
    while remaining > 0:
        item = result_queue.get()
        if item[0] == "__error__":
            raise RuntimeError(item[1])
        sample_id, preds = item
        pred_by_sample[int(sample_id)] = preds
        remaining -= 1
        done += 1

        elapsed = max(1e-6, time.time() - started_at)
        rate = done / elapsed
        eta_sec = int((total - done) / rate) if rate > 0 else 0
        pct = (100.0 * done / total) if total > 0 else 100.0
        print(
            f"[offline-cache] progress: {done}/{total} ({pct:.1f}%) | "
            f"{rate:.2f} samples/s | ETA {eta_sec}s"
        )

    for p in processes:
        p.join()

    for sample in samples:
        sid = int(sample["sample_id"])
        preds = pred_by_sample[sid]
        sample["token_budget"] = int(token_budget)
        sample["action_space"] = {
            "a_values": cart_a,
            "b_values": cart_b,
        }
        sample["action_outputs"] = preds


def main(args):
    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model)
    max_input_length = int(args.max_input_length or resolve_max_input_length(args.model))

    train_samples = load_longbench_sampled_splits(
        tokenizer=tokenizer,
        longbench_dir=args.longbench_dir,
        sample_ratio=args.sample_ratio,
        num_length_bins=args.num_length_bins,
        seed=args.seed,
    )

    apply_eval_style_truncation(train_samples, tokenizer, max_input_length)
    for idx, sample in enumerate(train_samples):
        sample["sample_id"] = idx

    if torch.cuda.is_available():
        visible_gpu_count = int(torch.cuda.device_count())
    else:
        visible_gpu_count = 0
    if visible_gpu_count <= 0:
        raise RuntimeError("No GPU available for offline LLM generation cache.")
    build_offline_action_cache(
        samples=train_samples,
        model_name=args.model,
        token_budget=int(args.token_budget),
        visible_gpu_count=visible_gpu_count,
        gpus_per_model=int(args.gpus_per_model),
    )

    write_jsonl(args.output_file, train_samples, "train")

    print(f"\nSaved train: {args.output_file} ({len(train_samples)} samples)")
    num_workers = (visible_gpu_count + int(args.gpus_per_model) - 1) // int(args.gpus_per_model)
    print(
        f"Offline action cache built with {visible_gpu_count} visible GPU(s), "
        f"gpus_per_model={int(args.gpus_per_model)}, workers={num_workers}, "
        f"token_budget={int(args.token_budget)}"
    )
    print(f"Truncation max_input_length={max_input_length} (model={args.model})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LongBench 부분 샘플링(10%/2%) + 평가 동일 절단, GT 기반 RL 학습용 jsonl"
    )
    parser.add_argument("--longbench_dir", type=str, default=os.path.join(REPO_ROOT, "datasets", "longbench"))
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(REPO_ROOT, "RL", "training", "data", "training_data.jsonl"),
    )
    parser.add_argument("--sample_ratio", type=float, default=DEFAULT_SAMPLE_RATIO, help="Train 비율 (기본 0.10)")
    parser.add_argument("--num_length_bins", type=int, default=DEFAULT_LENGTH_BINS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--model", type=str, default="llama3", choices=["llama", "llama2", "llama3", "opt"])
    parser.add_argument("--max_input_length", type=int, default=None)
    parser.add_argument("--token_budget", type=int, default=1024)
    parser.add_argument("--gpus_per_model", type=int, default=1)
    main(parser.parse_args())

