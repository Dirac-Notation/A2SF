#!/usr/bin/env python3
"""학습 데이터의 각 샘플에 대해 full cache 추론을 돌리고,
action_scores를 full cache output 기준으로 재계산하여 새 학습 데이터를 생성.

- Code dataset(lcc, repobench-p)은 aug=0만 유지, aug>0 제거.
- Multi-GPU로 full cache 추론을 병렬 처리.
- 출력 파일에 reward_reference="full_cache" 표시.

사용법:
    python script/build_fullcache_training_data.py \
        --training_data RL/training/1b_fix_exp_aug4/training_data_backup.jsonl \
        --model llama3-1b \
        --output RL/training/1b_fix_exp_aug4/training_data.jsonl \
        --gpus_per_model 1
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from utils import load_model
from longbench_eval import dataset2metric

CODE_DATASETS = {"lcc", "repobench-p"}


def _format_prompt(prompt: str, dataset: str, model_name: str) -> str:
    if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        if "llama" in model_name:
            return f"[INST]{prompt}[/INST]"
    return prompt


def _build_gen_kwargs(tokenizer, dataset: str, gen_length: int, ctx_len: int) -> dict:
    kwargs = {
        "max_new_tokens": int(gen_length),
        "num_beams": 1,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "num_logits_to_keep": 1,
    }
    if dataset == "samsum":
        kwargs["min_length"] = ctx_len + 1
        kwargs["eos_token_id"] = [
            tokenizer.eos_token_id,
            tokenizer.encode("\n", add_special_tokens=False)[-1],
        ]
    return kwargs


def _fullcache_worker(
    worker_id: int,
    gpu_group: List[int],
    model_name: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_group)
    torch.set_grad_enabled(False)
    print(f"[fullcache][worker {worker_id}] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    model, tokenizer = load_model(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[fullcache][worker {worker_id}] model loaded, device={model.device}")

    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            sample_id = int(task["sample_id"])
            dataset = str(task["dataset"])
            prompt = _format_prompt(
                str(task["input_prompt"]), dataset, model_name
            )
            encoded = tokenizer(prompt, truncation=False, return_tensors="pt")
            input_ids = encoded.input_ids.to(model.device)
            attention_mask = encoded.attention_mask.to(torch.bfloat16).to(model.device)
            ctx_len = int(input_ids.shape[-1])
            gen_length = int(task.get("generation_length", 64))

            gen_kwargs = _build_gen_kwargs(tokenizer, dataset, gen_length, ctx_len)

            # Full cache (no compression)
            model.init_cache(None)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            pred_text = tokenizer.decode(output_ids[0, ctx_len:], skip_special_tokens=True)
            result_queue.put(("ok", sample_id, pred_text))
    except Exception as exc:
        result_queue.put(("__error__", f"worker {worker_id}: {exc}"))


def main():
    parser = argparse.ArgumentParser(
        description="Full cache 추론 → action_scores 재계산 → 새 학습 데이터 생성"
    )
    parser.add_argument("--training_data", type=str, required=True, help="원본 학습 데이터 (backup)")
    parser.add_argument("--model", type=str, default="llama3-1b")
    parser.add_argument("--output", type=str, required=True, help="출력 학습 데이터 경로")
    parser.add_argument("--gpus_per_model", type=int, default=1)
    args = parser.parse_args()

    # 1. 원본 학습 데이터 로드
    print(f"Loading training data: {args.training_data}")
    with open(args.training_data, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]
    print(f"  Total: {len(all_samples)}")

    # Code dataset aug>0 제거
    before = len(all_samples)
    all_samples = [
        s for s in all_samples
        if s.get("dataset") not in CODE_DATASETS or int(s.get("middle_shuffle_aug", 0)) == 0
    ]
    removed = before - len(all_samples)
    if removed > 0:
        print(f"  Removed {removed} code augmented samples (kept aug=0 only)")
    print(f"  After filter: {len(all_samples)}")

    # 2. 모든 샘플에 대해 full cache 추론 (augmented 포함 — 각자의 prompt로 추론)
    print(f"  Full cache inference targets: {len(all_samples)}")

    # 3. Multi-GPU full cache 추론
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    visible_gpus = torch.cuda.device_count()
    gpus_per_model = int(args.gpus_per_model)
    all_gpu_ids = list(range(visible_gpus))
    gpu_groups = [all_gpu_ids[i:i + gpus_per_model] for i in range(0, visible_gpus, gpus_per_model)]
    print(f"\n  GPUs: {visible_gpus}, groups: {gpu_groups}")

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for s in all_samples:
        task_queue.put(s)
    for _ in gpu_groups:
        task_queue.put(None)

    processes = []
    model_key = args.model.split("_")[0].lower()
    for wid, gg in enumerate(gpu_groups):
        p = ctx.Process(target=_fullcache_worker, args=(wid, gg, model_key, task_queue, result_queue))
        p.start()
        processes.append(p)

    # 결과 수집
    fullcache_preds: Dict[int, str] = {}  # sample_id → pred_text
    total = len(all_samples)
    done = 0
    started_at = time.time()
    print(f"\n  Running full cache inference on {total} samples...")

    while done < total:
        item = result_queue.get()
        if item[0] == "__error__":
            raise RuntimeError(item[1])
        _, sample_id, pred_text = item
        fullcache_preds[sample_id] = pred_text
        done += 1
        elapsed = max(1e-6, time.time() - started_at)
        rate = done / elapsed
        print(
            f"  [{done}/{total}] {rate:.2f} samples/s",
            flush=True,
        )

    for p in processes:
        p.join()
    print(f"  Full cache inference done. {len(fullcache_preds)} predictions collected.")

    # 4. action_scores 재계산 — 각 샘플별 full cache pred 직접 사용
    print("\nRecomputing action scores with full cache reference...")
    output_rows: List[Dict[str, Any]] = []
    recomputed = 0
    skipped = 0

    for sample in all_samples:
        sid = int(sample.get("sample_id", 0))
        if sid not in fullcache_preds:
            skipped += 1
            output_rows.append(sample)
            continue

        full_pred = fullcache_preds[sid]
        metric_type = str(sample.get("metric_type", "qa_f1_score"))
        all_classes = sample.get("all_classes", [])

        # dataset2metric에서 함수 가져오기
        metric_fn = None
        for ds_name, fn in dataset2metric.items():
            if fn.__name__ == metric_type:
                metric_fn = fn
                break
        if metric_fn is None:
            metric_fn = dataset2metric.get(ds, list(dataset2metric.values())[0])

        # budget별 재계산
        outputs_by_budget = sample.get("action_outputs_by_budget", {})
        new_scores_by_budget: Dict[str, List[float]] = {}
        for bkey, outputs in outputs_by_budget.items():
            new_scores = [
                float(metric_fn(str(pred), full_pred, all_classes=all_classes))
                for pred in outputs
            ]
            new_scores_by_budget[bkey] = new_scores
        sample["action_scores_by_budget"] = new_scores_by_budget

        # 단일 action_scores도 교체
        action_outputs = sample.get("action_outputs", [])
        if action_outputs:
            sample["action_scores"] = [
                float(metric_fn(str(pred), full_pred, all_classes=all_classes))
                for pred in action_outputs
            ]

        sample["reward_reference"] = "full_cache"
        sample["full_cache_pred"] = full_pred
        output_rows.append(sample)
        recomputed += 1

    # 6. 저장
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"\nDone!")
    print(f"  Recomputed: {recomputed}")
    print(f"  Skipped: {skipped}")
    print(f"  Output: {args.output} ({len(output_rows)} samples)")


if __name__ == "__main__":
    main()
