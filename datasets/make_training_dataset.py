#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_SPLIT_SEED = 42
DEFAULT_SAMPLE_RATIO = 0.10
DEFAULT_LENGTH_BINS = 10
DEFAULT_EVAL_RATIO = 0.02

TASK2DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "task2dataset.json",
)
with open(TASK2DATASET_PATH, "r", encoding="utf-8") as f:
    DATA_GROUP = json.load(f)

DATASET_TO_TASK = {
    dataset_name: group_name
    for group_name, datasets in DATA_GROUP.items()
    for dataset_name in datasets
}

# Teacher generation length (max new tokens) per LongBench task group.
# Values follow the same scale as config/dataset2maxlen.json for each category.
TASK_MAX_LEN: Dict[str, int] = {
    "Code Complete": 64,
    "Few Shot": 128,
    "Single-doc QA": 128,
    "Multi-doc QA": 128,
    "Summarization": 512,
    "Passage Retrieval": 32,
    "unknown": 32,
}

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_for_generation(model_name: str):
    with open("./config/model2path.json", "r", encoding="utf-8") as f:
        model2path = json.load(f)

    model_path = model2path[model_name]
    print(f"Loading model: {model_name} from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def resolve_max_input_length(model_name: str, model=None) -> int:
    model_key = model_name.split("_")[0].lower()
    try:
        with open("./config/model2maxlen.json", "r", encoding="utf-8") as f:
            model2maxlen = json.load(f)
        if model_key in model2maxlen:
            return int(model2maxlen[model_key])
        if model_name in model2maxlen:
            return int(model2maxlen[model_name])
    except Exception:
        pass

    if model is not None and hasattr(model, "config"):
        max_pos = getattr(model.config, "max_position_embeddings", None)
        if isinstance(max_pos, int) and max_pos > 0:
            return max_pos
    return 8192


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


def truncate_prompt(prompt: str, tokenizer, max_input_length: int) -> str:
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized_prompt) <= max_input_length:
        return prompt

    half = int(max_input_length / 2)
    return tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
        tokenized_prompt[-half:], skip_special_tokens=True
    )


def evenly_spaced_indices(size: int, count: int) -> List[int]:
    if count <= 0:
        return []
    if count >= size:
        return list(range(size))
    if count == 1:
        return [size // 2]

    # Pick points spread across the whole sorted range.
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

    # Fallback: randomly fill leftover if any.
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

    # Sort by length so each bin corresponds to a segment of sequence length range.
    sorted_samples = sorted(samples, key=lambda x: x["length"])
    actual_bins = max(1, min(num_bins, total))

    # Build equal-count bins (quantile-like).
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

        # Keep each bin's internal span as wide as possible.
        bin_sorted = sorted(bin_samples, key=lambda x: x["length"])
        chosen = evenly_spaced_indices(len(bin_sorted), count)
        selected.extend([bin_sorted[idx] for idx in chosen])

    # Safety net for duplicate indices / rounding edge cases.
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

    # Ensure exact count when ratio/rounding effects appear.
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


def load_longbench_datasets(
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
        generation_length = TASK_MAX_LEN.get(task_type, TASK_MAX_LEN["unknown"])

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
                        "source": "LongBench(local)",
                        "length": token_len,
                        "task_type": task_type,
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


def generate_answer(
    model,
    tokenizer,
    sample: Dict[str, Any],
    model_name: str,
    sample_id: int,
    probs_dir: str,
    teacher_topk: int,
    max_input_length: int,
) -> Dict[str, Any]:
    prompt = sample["input_prompt"]
    generation_length = sample["generation_length"]
    dataset_name = sample.get("dataset", "unknown")
    task_type = sample.get("task_type", "unknown")
    prompt = truncate_prompt(prompt, tokenizer, max_input_length)

    if "llama" in model_name:
        prompt_for_gen = f"[INST]{prompt}[/INST]"
    else:
        prompt_for_gen = prompt

    encoded = tokenizer(prompt_for_gen, truncation=False, return_tensors="pt")
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)

    with torch.no_grad():
        prefill_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=False,
        )
        past_key_values = prefill_outputs.past_key_values

        logits = prefill_outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated_tokens = []
        step_logits = []
        all_decode_attentions = []
        attention_mask_full = attention_mask

        for step in range(generation_length):
            if step == 0:
                cur_input_ids = next_token
                step_logits.append(logits.detach().to(torch.float32).cpu())
            else:
                logits = outputs.logits[:, -1, :]
                cur_input_ids = torch.argmax(logits, dim=-1, keepdim=True)
                step_logits.append(logits.detach().to(torch.float32).cpu())

            generated_tokens.append(cur_input_ids)

            attention_mask_full = torch.cat(
                [
                    attention_mask_full,
                    attention_mask_full.new_ones((attention_mask_full.size(0), 1)),
                ],
                dim=-1,
            )

            outputs = model(
                input_ids=cur_input_ids,
                attention_mask=attention_mask_full,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )
            past_key_values = outputs.past_key_values
            all_decode_attentions.append(outputs.attentions)

            if tokenizer.eos_token_id is not None and (cur_input_ids == tokenizer.eos_token_id).all():
                break

            decoded_text_step = tokenizer.decode(torch.cat(generated_tokens, dim=-1)[0], skip_special_tokens=False)
            if "[/INST]" in decoded_text_step:
                break

        if generated_tokens:
            generated_ids = torch.cat(generated_tokens, dim=-1)
        else:
            generated_ids = input_ids.new_zeros((1, 0))

        # Safety clamp: never exceed requested generation length.
        if generated_ids.size(1) > generation_length:
            generated_ids = generated_ids[:, :generation_length]

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        output_ids = torch.cat([input_ids, generated_ids], dim=-1)

        decode_attentions = [
            torch.cat(step_attn, dim=0) for step_attn in all_decode_attentions
        ]

        output = {
            "sequences": output_ids,
            "decode_attentions": decode_attentions,
        }

    if output["decode_attentions"]:
        first_decode_attention = output["decode_attentions"][0]
        scores = torch.zeros(
            *first_decode_attention.shape[:3],
            input_ids.size(1),
            device=first_decode_attention.device,
            dtype=first_decode_attention.dtype,
        )
        for decode_attention in output["decode_attentions"]:
            scores += decode_attention[:, :, :, : input_ids.size(1)]

        num_heads = model.config.num_attention_heads
        num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
        group_size = num_heads // num_kv_heads
        num_layers = scores.size(0)
        seq_len = scores.size(3)
        scores = scores.view(num_layers, num_kv_heads, group_size, 1, seq_len)
        grouped_scores = scores.sum(dim=2)
        answer_indices = grouped_scores.topk(128, dim=3).indices.tolist()
    else:
        answer_indices = []

    if step_logits:
        step_logits_tensor = torch.stack(step_logits, dim=1).squeeze(0)
        topk = min(int(teacher_topk), step_logits_tensor.size(-1))
        teacher_probs = torch.softmax(step_logits_tensor, dim=-1)
        teacher_topk_probs, teacher_topk_indices = torch.topk(teacher_probs, k=topk, dim=-1)
    else:
        teacher_topk_indices = torch.empty((0, 0), dtype=torch.long)
        teacher_topk_probs = torch.empty((0, 0), dtype=torch.float32)

    probs_path = os.path.join(probs_dir, f"{sample_id}.pt")
    torch.save(
        {
            "sample_id": sample_id,
            "answer_token_ids": generated_ids[0].detach().to(torch.long).cpu(),
            "teacher_topk_indices": teacher_topk_indices.to(torch.long).cpu(),
            "teacher_topk_probs": teacher_topk_probs.to(torch.float32).cpu(),
        },
        probs_path,
    )

    return {
        "sample_id": sample_id,
        "dataset": dataset_name,
        "task_type": task_type,
        "input_prompt": prompt_for_gen,
        "generation_length": len(output["decode_attentions"]),
        "generated_text": generated_text,
        "answer_indices": answer_indices,
        "target_prob_file": probs_path,
    }


def split_samples_fixed(
    all_samples: List[Dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not (0.0 < eval_ratio < 1.0):
        raise ValueError(f"eval_ratio must be in (0, 1), got {eval_ratio}")
    if len(all_samples) < 2:
        raise ValueError("Need at least 2 samples to split into train/eval")

    rng = random.Random(seed)
    task_groups = defaultdict(list)
    for sample in all_samples:
        task_groups[sample.get("task_type", "unknown")].append(sample)

    train_samples: List[Dict[str, Any]] = []
    eval_samples: List[Dict[str, Any]] = []

    for _, samples in task_groups.items():
        rng.shuffle(samples)
        eval_count = int(round(len(samples) * eval_ratio))
        if len(samples) == 1:
            eval_count = 0
        elif eval_count == 0:
            eval_count = 1
        elif eval_count >= len(samples):
            eval_count = len(samples) - 1

        eval_samples.extend(samples[:eval_count])
        train_samples.extend(samples[eval_count:])

    rng.shuffle(train_samples)
    rng.shuffle(eval_samples)
    return train_samples, eval_samples


def check_training_data_stats(file_path: str) -> None:
    dataset_counts = defaultdict(int)
    task_counts = defaultdict(int)
    total_lines = 0

    print(f"Reading {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: line {line_num} is not valid JSON: {exc}")
                continue

            dataset = data.get("dataset", "unknown")
            task_type = data.get("task_type", "unknown")
            dataset_counts[dataset] += 1
            task_counts[task_type] += 1
            total_lines += 1

    print(f"\nTotal samples: {total_lines}")
    print("Dataset counts:")
    for dataset in sorted(dataset_counts.keys()):
        count = dataset_counts[dataset]
        pct = (count / total_lines * 100) if total_lines else 0
        print(f"  {dataset:24s}: {count:5d} ({pct:5.2f}%)")

    print("Task type counts:")
    for task in sorted(task_counts.keys()):
        count = task_counts[task]
        pct = (count / total_lines * 100) if total_lines else 0
        print(f"  {task:24s}: {count:5d} ({pct:5.2f}%)")


def main(args):
    set_seed(42)
    model_name = args.model
    model, tokenizer = load_model_for_generation(model_name)
    max_input_length = resolve_max_input_length(model_name, model=model)
    print(f"Max input length (longbench style truncation): {max_input_length}")

    print(f"\n{'=' * 60}")
    print("Loading local LongBench data with per-dataset disjoint train/eval sampling")
    print(f"{'=' * 60}")
    train_samples, eval_samples = load_longbench_datasets(
        tokenizer=tokenizer,
        longbench_dir=args.longbench_dir,
        sample_ratio=DEFAULT_SAMPLE_RATIO,
        eval_ratio=DEFAULT_EVAL_RATIO,
        num_length_bins=DEFAULT_LENGTH_BINS,
        seed=DEFAULT_SPLIT_SEED,
    )
    print(
        f"\nDisjoint sampling complete "
        f"(seed={DEFAULT_SPLIT_SEED}, train_ratio={DEFAULT_SAMPLE_RATIO:.3f}, eval_ratio={DEFAULT_EVAL_RATIO:.3f})"
    )
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Eval samples:  {len(eval_samples)}")

    output_parent = os.path.dirname(args.output_file)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    os.makedirs(args.probs_dir, exist_ok=True)

    split_to_samples = {"train": train_samples, "eval": eval_samples}
    split_to_output = {"train": args.output_file, "eval": args.eval_output_file}
    for out_path in split_to_output.values():
        out_parent = os.path.dirname(out_path)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)

    split_counts = defaultdict(int)
    split_dataset_counts = {"train": defaultdict(int), "eval": defaultdict(int)}
    next_sample_id = 0

    with open(split_to_output["train"], "w", encoding="utf-8") as f_train, open(
        split_to_output["eval"], "w", encoding="utf-8"
    ) as f_eval:
        split_to_file = {"train": f_train, "eval": f_eval}
        for split_name in ("train", "eval"):
            samples = split_to_samples[split_name]
            for sample in tqdm(samples, desc=f"Generating & saving ({split_name})"):
                try:
                    result = generate_answer(
                        model=model,
                        tokenizer=tokenizer,
                        sample=sample,
                        model_name=model_name,
                        sample_id=next_sample_id,
                        probs_dir=args.probs_dir,
                        teacher_topk=args.teacher_topk,
                        max_input_length=max_input_length,
                    )
                    next_sample_id += 1

                    training_sample = {
                        "sample_id": result["sample_id"],
                        "split": split_name,
                        "dataset": result["dataset"],
                        "task_type": result["task_type"],
                        "input_prompt": result["input_prompt"],
                        "generation_length": result["generation_length"],
                        "generated_text": result["generated_text"],
                        "answer_indices": result["answer_indices"],
                        "target_prob_file": result["target_prob_file"],
                    }

                    fout = split_to_file[split_name]
                    fout.write(json.dumps(training_sample, ensure_ascii=False, separators=(",", ":")) + "\n")
                    fout.flush()

                    split_counts[split_name] += 1
                    split_dataset_counts[split_name][result["dataset"]] += 1
                except Exception as exc:
                    print(f"\nError processing {split_name} sample: {exc}")
                    continue

    print(f"\n{'=' * 60}")
    print("Training data generation complete")
    print(f"Train file: {args.output_file}")
    print(f"Eval file:  {args.eval_output_file}")
    print(f"{'=' * 60}")

    print(f"\nSamples per split:")
    print(f"  train: {split_counts['train']}")
    print(f"  eval:  {split_counts['eval']}")

    for split_name in ("train", "eval"):
        print(f"\nSamples per dataset ({split_name}):")
        for dataset, count in sorted(split_dataset_counts[split_name].items()):
            print(f"  {dataset}: {count}")

    print(f"\n{'=' * 60}")
    print("Detailed statistics")
    print(f"{'=' * 60}")
    check_training_data_stats(args.output_file)
    check_training_data_stats(args.eval_output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training data from local LongBench jsonl files with per-dataset length-balanced sampling"
    )
    parser.add_argument(
        "--longbench_dir",
        type=str,
        default="./datasets/longbench",
        help="Directory containing local LongBench jsonl files",
    )
    parser.add_argument("--output_file", type=str, default="./datasets/training_data.jsonl")
    parser.add_argument("--eval_output_file", type=str, default="./datasets/eval_data.jsonl")
    parser.add_argument("--probs_dir", type=str, default="./datasets/training_probs")
    parser.add_argument("--teacher_topk", type=int, default=128)
    parser.add_argument("--model", type=str, default="llama3", choices=["llama", "llama2", "llama3", "opt"])
    main(parser.parse_args())
