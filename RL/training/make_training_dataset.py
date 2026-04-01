#!/usr/bin/env python3
"""
LongBench 일부만 샘플링해 RL 학습용 jsonl 생성.

이 코드는 기존 `datasets/make_training_dataset.py`의 내용을 그대로
`RL/training/data_generation/` 위치로 이식해, 학습 파이프라인이
datasets 쪽을 호출하지 않도록 구성했습니다.

- 데이터셋별 train: 기본 10% (길이 균형 샘플링)
- 샘플이 확정된 뒤, 평가(longbench_RL 등)와 동일하게 max_input_length 기준 중간 절단 적용
- generation_length: config/dataset2maxlen.json (데이터셋별, 평가와 동일 스케일)
- 각 라인에 `metric_type`, `action_scores`(액션별 채점 결과)를 함께 기록해 학습 시 재계산 없이 사용
"""

import argparse
import json
import math
import os
import random
import multiprocessing as mp
import sys
import time
import hashlib
from typing import Any, Dict, List, Tuple

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
from longbench_eval import (
    dataset2metric,
    qa_f1_score,
    qa_f1_zh_score,
    rouge_score,
    rouge_zh_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

_METRIC_FN_REGISTRY = {
    "qa_f1_score": qa_f1_score,
    "qa_f1_zh_score": qa_f1_zh_score,
    "rouge_score": rouge_score,
    "rouge_zh_score": rouge_zh_score,
    "classification_score": classification_score,
    "retrieval_score": retrieval_score,
    "retrieval_zh_score": retrieval_zh_score,
    "count_score": count_score,
    "code_sim_score": code_sim_score,
}

DEFAULT_SPLIT_SEED = 42
DEFAULT_SAMPLE_RATIO = 0.10
DEFAULT_LENGTH_BINS = 10
DEFAULT_TOKEN_BUDGETS = [128, 256, 512, 1024, 2048, 4096]

# RL/training/data_generation/make_training_dataset.py 기준으로 repo root로 이동

TASK2DATASET_PATH = os.path.join(REPO_ROOT, "config", "task2dataset.json")
DATASET2MAXLEN_PATH = os.path.join(REPO_ROOT, "config", "dataset2maxlen.json")

with open(TASK2DATASET_PATH, "r", encoding="utf-8") as f:
    TASK_TO_DATASETS = json.load(f)

DATASET_TO_TASK = {dataset_name: task_name for task_name, datasets in TASK_TO_DATASETS.items() for dataset_name in datasets}

with open(DATASET2MAXLEN_PATH, "r", encoding="utf-8") as f:
    DATASET_TO_MAXGEN = json.load(f)


def resolve_metric_type_for_dataset(dataset_name: str) -> str:
    """Same mapping as RL training / LongBench eval (`dataset2metric`)."""
    dataset_key = str(dataset_name or "").strip().lower()
    fn = dataset2metric.get(dataset_key)
    if fn is None:
        return "qa_f1_score"
    return fn.__name__


def compute_action_scores(
    action_outputs: List[str],
    metric_type: str,
    answers: List[str],
    all_classes: List[str],
) -> List[float]:
    """Per-action reward under the dataset metric (max over reference answers)."""
    metric_fn = _METRIC_FN_REGISTRY.get(str(metric_type), qa_f1_score)
    scores: List[float] = []
    for pred_text in action_outputs:
        reward_val = 0.0
        if answers:
            for gt in answers:
                reward_val = max(
                    reward_val,
                    float(metric_fn(str(pred_text), gt, all_classes=all_classes)),
                )
        scores.append(float(reward_val))
    return scores


def _best_action_index_from_scores(scores: List[float]) -> int:
    best_idx = 0
    best_r = float("-inf")
    for i, s in enumerate(scores):
        v = float(s)
        if v > best_r:
            best_r = v
            best_idx = i
    return int(best_idx)


def set_seed(seed: int) -> None:
    random.seed(seed)
def stable_seed_from_name(base_seed: int, name: str) -> int:
    """Create a stable per-dataset seed independent of Python hash randomization."""
    digest = hashlib.sha256(str(name).encode("utf-8")).hexdigest()
    # Use lower 32 bits to keep it in a practical range.
    return int(base_seed) + (int(digest[:8], 16))




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


MIDDLE_TRIM = 128
# 문자 길이가 이 값 이하면 중간 4등분 셔플을 적용하지 않는다(샘플 1개만 유지).
MIN_PROMPT_LEN_FOR_MIDDLE_SHUFFLE = 512


def augment_prompt_middle_quarter_shuffle_variants(
    prompt: str,
    rng: random.Random,
    num_augment: int = 4,
) -> List[str]:
    """
    `prompt[128:-128]` 구간을 4등분한 뒤, 청크 1~4가 모두 한 번씩 쓰이도록 순서를 바꾼 프롬프트들을 만든다.
    - `num_augment == 1`: 원래 순서 (1→2→3→4)만.
    - `num_augment >= 2`: 위 1개 + 랜덤 순열을 `num_augment - 1`개 추가(가능하면 서로 다른 순열).
    앞 128자·뒤 128자는 그대로 둔다.
    `len(prompt) <= MIN_PROMPT_LEN_FOR_MIDDLE_SHUFFLE`이면 셔플을 하지 않고 `[prompt]` 한 개만 반환한다.
    """
    if num_augment < 1:
        raise ValueError(f"num_augment must be >= 1, got {num_augment}")

    if len(prompt) <= MIN_PROMPT_LEN_FOR_MIDDLE_SHUFFLE:
        return [prompt]

    head = prompt[:MIDDLE_TRIM]
    tail = prompt[-MIDDLE_TRIM:]
    middle = prompt[MIDDLE_TRIM:-MIDDLE_TRIM]
    n = len(middle)
    if n < 4:
        return [prompt]

    q, r = divmod(n, 4)
    sizes = [q + (1 if i < r else 0) for i in range(4)]
    parts: List[str] = []
    pos = 0
    for sz in sizes:
        parts.append(middle[pos : pos + sz])
        pos += sz

    def assemble(order: List[int]) -> str:
        return head + "".join(parts[i] for i in order) + tail

    identity = (0, 1, 2, 3)
    orders: List[Tuple[int, ...]] = [identity]
    if num_augment == 1:
        return [assemble(list(identity))]

    seen = {identity}
    for _ in range(num_augment - 1):
        for _attempt in range(500):
            perm = [0, 1, 2, 3]
            rng.shuffle(perm)
            t = tuple(perm)
            if t not in seen:
                seen.add(t)
                orders.append(t)
                break
        else:
            perm = [0, 1, 2, 3]
            rng.shuffle(perm)
            orders.append(tuple(perm))

    return [assemble(list(o)) for o in orders[:num_augment]]


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

        dataset_seed = stable_seed_from_name(seed, dataset_name)
        train_sampled = length_balanced_sample(
            samples=dataset_samples,
            ratio=sample_ratio,
            num_bins=num_length_bins,
            seed=dataset_seed,
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
    action_batch_size: int,
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
    action_batch_size = max(1, int(action_batch_size))
    pred_texts: List[str] = []
    num_actions = int(a_values.numel())
    for start in range(0, num_actions, action_batch_size):
        end = min(start + action_batch_size, num_actions)
        a = a_values[start:end]
        b = b_values[start:end]
        batch_n = int(a.numel())
        input_ids_batch = input_ids.repeat(batch_n, 1)
        attention_mask_batch = attention_mask.repeat(batch_n, 1)
        model.init_cache(_build_compression_config(a, b, token_budget))
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                **generation_kwargs,
            )
        for i in range(batch_n):
            pred_texts.append(
                tokenizer.decode(output_ids[i, context_length:], skip_special_tokens=True)
            )
    return pred_texts


def _offline_worker(
    worker_id: int,
    gpu_group: List[int],
    model_name: str,
    task_queue: mp.Queue,
    a_values: List[float],
    b_values: List[float],
    result_queue: mp.Queue,
    action_batch_size: int,
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
            task = task_queue.get()
            if task is None:
                break
            token_budget, sample = task
            pred_texts = _offline_generate_for_sample(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                model_name=model_name,
                a_values=a_tensor,
                b_values=b_tensor,
                token_budget=int(token_budget),
                action_batch_size=action_batch_size,
            )
            result_queue.put((int(token_budget), int(sample["sample_id"]), pred_texts))
    except Exception as exc:  # pragma: no cover
        result_queue.put(("__error__", f"worker {worker_id} gpus={gpu_group}: {exc}"))


def build_offline_action_cache(
    samples: List[Dict[str, Any]],
    model_name: str,
    output_file: str,
    token_budgets: List[int],
    visible_gpu_count: int,
    gpus_per_model: int,
    action_batch_size: int,
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

    def _budget_cache_path(tb: int) -> str:
        base, ext = os.path.splitext(output_file)
        ext = ext or ".jsonl"
        return f"{base}.budget_{int(tb)}{ext}"

    token_budgets = [int(tb) for tb in token_budgets]
    total_samples = len(samples)
    total_budgets = len(token_budgets)
    total_sample_budget_units = total_samples * total_budgets
    total_action_units = total_sample_budget_units * action_size
    completed_sample_budget_units = 0
    for sample in samples:
        sample["token_budgets"] = token_budgets
        sample["action_space"] = {
            "a_values": cart_a,
            "b_values": cart_b,
        }
        sample["action_outputs_by_budget"] = {}
        sample["action_scores_by_budget"] = {}
        sample["metric_type"] = resolve_metric_type_for_dataset(sample.get("dataset"))

    pending_budgets: List[int] = []
    for token_budget in token_budgets:
        budget_path = _budget_cache_path(token_budget)
        if os.path.exists(budget_path):
            print(f"[offline-cache] skip existing budget={token_budget}: {budget_path}")
            completed_sample_budget_units += total_samples
            continue
        pending_budgets.append(int(token_budget))

    if pending_budgets:
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        for token_budget in pending_budgets:
            for sample in samples:
                task_queue.put((int(token_budget), sample))
        for _ in range(len(gpu_groups)):
            task_queue.put(None)
        result_queue = ctx.Queue()
        processes = []
        for worker_id, gpu_group in enumerate(gpu_groups):
            p = ctx.Process(
                target=_offline_worker,
                args=(
                    worker_id,
                    gpu_group,
                    model_name,
                    task_queue,
                    cart_a,
                    cart_b,
                    result_queue,
                    action_batch_size,
                ),
            )
            p.start()
            processes.append(p)

        total = len(samples) * len(pending_budgets)
        remaining = total
        done = 0
        started_at = time.time()
        done_by_budget: Dict[int, int] = {int(tb): 0 for tb in pending_budgets}
        pred_by_budget_sample: Dict[int, Dict[int, List[str]]] = {
            int(tb): {} for tb in pending_budgets
        }
        while remaining > 0:
            item = result_queue.get()
            if item[0] == "__error__":
                raise RuntimeError(item[1])
            token_budget, sample_id, preds = item
            token_budget = int(token_budget)
            sample_id = int(sample_id)
            pred_by_budget_sample[token_budget][sample_id] = preds
            done_by_budget[token_budget] += 1
            remaining -= 1
            done += 1

            elapsed = max(1e-6, time.time() - started_at)
            rate = done / elapsed
            eta_sec = int((total - done) / rate) if rate > 0 else 0
            pct = (100.0 * done / total) if total > 0 else 100.0
            global_done_sample_budget = completed_sample_budget_units + done
            global_done_actions = global_done_sample_budget * action_size
            global_pct_sample_budget = (
                (100.0 * global_done_sample_budget / total_sample_budget_units)
                if total_sample_budget_units > 0
                else 100.0
            )
            global_pct_actions = (
                (100.0 * global_done_actions / total_action_units)
                if total_action_units > 0
                else 100.0
            )
            print(
                f"[offline-cache] all-budgets-progress: "
                f"{done}/{total} ({pct:.1f}%) | {rate:.2f} sample-budget/s | ETA {eta_sec}s | "
                f"global sample×budget: {global_done_sample_budget}/{total_sample_budget_units} "
                f"({global_pct_sample_budget:.1f}%) | "
                f"global sample×budget×action: {global_done_actions}/{total_action_units} "
                f"({global_pct_actions:.1f}%)"
            )

        for p in processes:
            p.join()
        completed_sample_budget_units += total

        for token_budget in pending_budgets:
            budget_path = _budget_cache_path(token_budget)
            with open(budget_path, "w", encoding="utf-8") as f:
                for sample in samples:
                    sid = int(sample["sample_id"])
                    preds = pred_by_budget_sample[int(token_budget)][sid]
                    answers = sample.get("answers") or []
                    all_classes = sample.get("all_classes") or []
                    mt = str(sample["metric_type"])
                    payload = dict(sample)
                    payload["token_budget"] = int(token_budget)
                    payload["action_space"] = {
                        "a_values": cart_a,
                        "b_values": cart_b,
                    }
                    payload["action_outputs"] = preds
                    payload["action_scores"] = compute_action_scores(preds, mt, answers, all_classes)
                    f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(f"[offline-cache] saved budget file: {budget_path}")

    # Merge per-budget cache files into one unified training jsonl.
    merged_by_sample_id: Dict[int, Dict[str, Any]] = {}
    for token_budget in token_budgets:
        budget_path = _budget_cache_path(token_budget)
        if not os.path.exists(budget_path):
            raise FileNotFoundError(f"Missing budget cache file: {budget_path}")
        with open(budget_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = int(row["sample_id"])
                if sid not in merged_by_sample_id:
                    base = dict(row)
                    base.pop("token_budget", None)
                    base.pop("action_outputs", None)
                    base.pop("action_scores", None)
                    base["token_budgets"] = token_budgets
                    base["action_outputs_by_budget"] = {}
                    base["action_scores_by_budget"] = {}
                    merged_by_sample_id[sid] = base
                merged = merged_by_sample_id[sid]
                bkey = str(int(token_budget))
                merged["action_outputs_by_budget"][bkey] = row["action_outputs"]
                merged["action_scores_by_budget"][bkey] = row["action_scores"]

    merged_rows: List[Dict[str, Any]] = []
    for sid in sorted(merged_by_sample_id.keys()):
        row = merged_by_sample_id[sid]
        ref_budget_key = "1024" if "1024" in row["action_scores_by_budget"] else str(
            int(token_budgets[0])
        )
        row["best_action_index"] = _best_action_index_from_scores(
            [float(v) for v in row["action_scores_by_budget"][ref_budget_key]]
        )
        merged_rows.append(row)

    write_jsonl(output_file, merged_rows, "train")
    print(f"[offline-cache] merged all budget files into: {output_file}")


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

    expanded: List[Dict[str, Any]] = []
    for idx, sample in enumerate(train_samples):
        rng = random.Random(stable_seed_from_name(args.seed, f"middle_shuffle_{idx}"))
        variants = augment_prompt_middle_quarter_shuffle_variants(
            sample["input_prompt"],
            rng,
            num_augment=int(args.num_augment),
        )
        for aug_i, p in enumerate(variants):
            row = dict(sample)
            row["input_prompt"] = p
            row["middle_shuffle_aug"] = int(aug_i)
            tok = tokenizer(p, truncation=False, return_tensors="pt")
            row["length"] = int(tok.input_ids.size(1))
            expanded.append(row)
    train_samples = expanded

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
        output_file=args.output_file,
        token_budgets=DEFAULT_TOKEN_BUDGETS,
        visible_gpu_count=visible_gpu_count,
        gpus_per_model=int(args.gpus_per_model),
        action_batch_size=int(args.action_batch_size),
    )

    write_jsonl(args.output_file, train_samples, "train")

    print(f"\nSaved train: {args.output_file} ({len(train_samples)} samples)")
    num_workers = (visible_gpu_count + int(args.gpus_per_model) - 1) // int(args.gpus_per_model)
    print(
        f"Offline action cache built with {visible_gpu_count} visible GPU(s), "
        f"gpus_per_model={int(args.gpus_per_model)}, workers={num_workers}, "
        f"token_budgets={DEFAULT_TOKEN_BUDGETS}"
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
    parser.add_argument("--model", type=str, default="llama3-1b", choices=["llama3-8b", "llama3-1b", "qwen2"])
    parser.add_argument("--max_input_length", type=int, default=None)
    parser.add_argument("--gpus_per_model", type=int, default=1)
    parser.add_argument("--action_batch_size", type=int, default=1, help="액션 생성 시 batch 크기(기본 1)")
    parser.add_argument("--num_augment", type=int, default=4, help="중간 4등분 셔플 변형 개수. 1이면 순서 1234만, 2 이상이면 그만큼(1+랜덤) 생성.")
    main(parser.parse_args())

