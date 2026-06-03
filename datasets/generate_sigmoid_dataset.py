#!/usr/bin/env python3
"""Generate full_cache + 16-sigmoid-action outputs for a common_{server}.jsonl.

Output structure:
  <outdir>/common.jsonl     — input rows augmented with full_cache_pred, full_cache_score
  <outdir>/budget_128.jsonl — per-sample rows with action_outputs, action_scores_gt, action_scores_fc

Resumable: skips sample_ids already present in output.

Multi-GPU via torch.multiprocessing spawn. Assign 1 GPU per worker.
"""
import argparse
import json
import os
import sys
import time
import multiprocessing as mp
from typing import Any, Dict, List, Set

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import CompressionConfig, load_model
from longbench_eval import (
    qa_f1_score, qa_f1_zh_score, rouge_score, rouge_zh_score,
    classification_score, retrieval_score, retrieval_zh_score,
    count_score, code_sim_score,
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

# 13-action paired space (post-collapse of redundant a=0 duplicates).
# See ModelConfig.sigmoid() in RL/a2sf_model.py for reference.
# 34-action grid: a=0 collapses to single (0,1); other a values use full b grid.
# a × b = {0.0001, 0.01, 100} × {1, 4, 8, 12, 20, 24, 32, 48, 64, 96, 128} + (0, 1)
# Champion-era 13 paired actions: (0,1) + cartesian({0.01, 0.1, 10} × {1, 16, 32, 128}).
# a=0 collapses (b irrelevant since σ(0)=0.5 uniform), so only one (0,1) entry kept.
_A_BASE = [0.01, 0.1, 10.0]
_B_BASE = [1.0, 16.0, 32.0, 128.0]
SIGMOID_A_VALUES = [0.0]
SIGMOID_B_VALUES = [1.0]
for _a in _A_BASE:
    for _b in _B_BASE:
        SIGMOID_A_VALUES.append(_a)
        SIGMOID_B_VALUES.append(_b)
assert len(SIGMOID_A_VALUES) == 13, len(SIGMOID_A_VALUES)


def score_vs_ref(pred: str, refs: List[str], metric_type: str, all_classes: List[str]) -> float:
    fn = _METRIC_FN_REGISTRY.get(metric_type, qa_f1_score)
    best = 0.0
    for r in refs:
        if not r:
            continue
        try:
            v = float(fn(str(pred), str(r), all_classes=all_classes or []))
        except Exception:
            v = 0.0
        if v > best:
            best = v
    return best


def score_vs_single(pred: str, ref: str, metric_type: str, all_classes: List[str]) -> float:
    if not ref:
        return 0.0
    fn = _METRIC_FN_REGISTRY.get(metric_type, qa_f1_score)
    try:
        return float(fn(str(pred), str(ref), all_classes=all_classes or []))
    except Exception:
        return 0.0


def _format_prompt(prompt: str, dataset_name: str, model_name: str) -> str:
    if str(dataset_name or "").strip().lower() not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        if "llama" in str(model_name).lower():
            return f"[INST]{prompt}[/INST]"
    return prompt


def _build_gen_kwargs(tokenizer, dataset_name: str, gen_len: int, ctx_len: int) -> Dict[str, Any]:
    kwargs = {
        "tokenizer": tokenizer,
        "stop_strings": "[/INST]",
        "max_new_tokens": int(gen_len),
        "num_beams": 1,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "num_logits_to_keep": 1,
    }
    if str(dataset_name or "").strip().lower() == "samsum":
        kwargs["min_length"] = int(ctx_len) + 1
        kwargs["eos_token_id"] = [
            tokenizer.eos_token_id,
            tokenizer.encode("\n", add_special_tokens=False)[-1],
        ]
    return kwargs


def _build_compression_config(a_vals: torch.Tensor, b_vals: torch.Tensor, budget: int) -> CompressionConfig:
    cfg = CompressionConfig()
    cfg.compression_method = "sigmoid"
    cfg.total_budget = int(budget)
    cfg.local_ratios = 0.125
    cfg.a = a_vals
    cfg.b = b_vals
    return cfg


def _worker(
    worker_id: int,
    gpu_ids: List[int],
    model_name: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    budget: int,
    action_batch_size: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_grad_enabled(False)
    print(f"[worker {worker_id}] gpus={gpu_ids} starting", flush=True)
    model, tokenizer = load_model(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    a_tensor = torch.tensor(SIGMOID_A_VALUES, dtype=torch.float32, device=model.device)
    b_tensor = torch.tensor(SIGMOID_B_VALUES, dtype=torch.float32, device=model.device)
    num_actions = len(SIGMOID_A_VALUES)
    action_batch_size = max(1, int(action_batch_size))

    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            phase, sample = task
            sid = int(sample["sample_id"])
            dataset_name = str(sample.get("dataset") or "")
            prompt_raw = str(sample["input_prompt"])
            prompt = _format_prompt(prompt_raw, dataset_name, model_name)
            encoded = tokenizer(prompt, truncation=False, return_tensors="pt")
            input_ids = encoded.input_ids.to(model.device)
            attention_mask = encoded.attention_mask.to(torch.bfloat16).to(model.device)
            ctx_len = int(input_ids.shape[-1])
            gen_kwargs = _build_gen_kwargs(
                tokenizer, dataset_name,
                gen_len=int(sample.get("generation_length", 64)),
                ctx_len=ctx_len,
            )

            if phase == "fullcache":
                model.init_cache(None)
                t0 = time.time()
                with torch.inference_mode():
                    out = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **gen_kwargs,
                    )
                pred = tokenizer.decode(out[0, ctx_len:], skip_special_tokens=True)
                result_queue.put(("fullcache", sid, pred, time.time() - t0, ctx_len))

            elif phase == "actions":
                outputs: List[str] = [""] * num_actions
                t0 = time.time()
                for start in range(0, num_actions, action_batch_size):
                    end = min(start + action_batch_size, num_actions)
                    a = a_tensor[start:end]
                    b = b_tensor[start:end]
                    batch_n = end - start
                    in_ids = input_ids.repeat(batch_n, 1)
                    attn = attention_mask.repeat(batch_n, 1)
                    model.init_cache(_build_compression_config(a, b, budget))
                    with torch.inference_mode():
                        out = model.generate(
                            input_ids=in_ids,
                            attention_mask=attn,
                            **gen_kwargs,
                        )
                    for i in range(batch_n):
                        outputs[start + i] = tokenizer.decode(out[i, ctx_len:], skip_special_tokens=True)
                result_queue.put(("actions", sid, outputs, time.time() - t0, ctx_len))
    except Exception as exc:
        import traceback
        result_queue.put(("__error__", f"worker {worker_id} gpu={gpu_ids}: {exc}\n{traceback.format_exc()}"))


def run_phase(
    phase: str,
    samples: List[Dict],
    model_name: str,
    gpu_ids: List[int],
    budget: int,
    action_batch_size: int,
) -> Dict[int, Any]:
    ctx = mp.get_context("spawn")
    tq = ctx.Queue()
    rq = ctx.Queue()
    # longest first for better load balance
    ordered = sorted(samples, key=lambda s: -int(s.get("length", 0)))
    for s in ordered:
        tq.put((phase, s))
    for _ in gpu_ids:
        tq.put(None)

    procs = []
    for wid, g in enumerate(gpu_ids):
        p = ctx.Process(target=_worker, args=(wid, [g], model_name, tq, rq, budget, action_batch_size))
        p.start()
        procs.append(p)

    results: Dict[int, Any] = {}
    total = len(samples)
    t_start = time.time()
    for i in range(total):
        item = rq.get()
        if item[0] == "__error__":
            for p in procs:
                p.terminate()
            raise RuntimeError(item[1])
        tag, sid, payload, dur, ctx_len = item
        results[sid] = payload
        done = i + 1
        elapsed = max(1e-6, time.time() - t_start)
        rate = done / elapsed
        eta = (total - done) / max(rate, 1e-6)
        print(f"[{phase}] {done}/{total} sid={sid} ctx={ctx_len} dur={dur:.1f}s rate={rate:.2f}/s eta={eta/60:.1f}m", flush=True)

    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    return results


def load_existing_fc(path: str) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "full_cache_pred" in r:
                    out[int(r["sample_id"])] = r
    return out


def load_existing_budget(path: str) -> Set[int]:
    done: Set[int] = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done.add(int(r["sample_id"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="common_{server}.jsonl")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--model", default="llama3-1b")
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--gpus", required=True, help="comma-separated logical GPU indices visible to this process")
    ap.add_argument("--action_batch_size", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    common_out = os.path.join(args.outdir, "common.jsonl")
    budget_out = os.path.join(args.outdir, f"budget_{args.budget}.jsonl")

    rows = [json.loads(l) for l in open(args.input)]
    print(f"Loaded {len(rows)} samples from {args.input}")

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    print(f"Using GPUs: {gpu_ids}")

    # PHASE 1: fullcache. Resume if common_out exists.
    existing_fc = load_existing_fc(common_out)
    print(f"Existing fullcache rows: {len(existing_fc)}")
    missing_fc = [r for r in rows if int(r["sample_id"]) not in existing_fc]
    if missing_fc:
        print(f"Running fullcache phase for {len(missing_fc)} samples")
        fc_results = run_phase("fullcache", missing_fc, args.model, gpu_ids, args.budget,
                                args.action_batch_size)
        # append augmented rows to common_out
        mode = "a"
        with open(common_out, mode) as f:
            for r in missing_fc:
                sid = int(r["sample_id"])
                pred = fc_results.get(sid, "")
                fc_score = score_vs_ref(pred, r.get("answers", []) or [], r.get("metric_type", "qa_f1_score"), r.get("all_classes", []))
                new_row = dict(r)
                new_row["full_cache_pred"] = pred
                new_row["full_cache_score"] = fc_score
                f.write(json.dumps(new_row, ensure_ascii=False) + "\n")
        print(f"Appended {len(missing_fc)} fullcache rows to {common_out}")
    else:
        print("Fullcache phase: nothing to do (all cached)")

    # Reload common_out to have fc_preds in memory
    fc_rows = load_existing_fc(common_out)

    # PHASE 2: actions at budget
    done_ids = load_existing_budget(budget_out)
    print(f"Existing budget_{args.budget} rows: {len(done_ids)}")
    missing_act = [r for r in rows if int(r["sample_id"]) not in done_ids]
    if not missing_act:
        print("Action phase: nothing to do")
        return

    print(f"Running action phase for {len(missing_act)} samples (batch_size={args.action_batch_size})")
    act_results = run_phase("actions", missing_act, args.model, gpu_ids, args.budget,
                            args.action_batch_size)

    # compute scores per action, write to budget_out
    num_actions = len(SIGMOID_A_VALUES)
    with open(budget_out, "a") as f:
        for r in missing_act:
            sid = int(r["sample_id"])
            outputs = act_results.get(sid, [""] * num_actions)
            metric = r.get("metric_type", "qa_f1_score")
            all_cls = r.get("all_classes", []) or []
            answers = r.get("answers", []) or []
            fc_pred = fc_rows.get(sid, {}).get("full_cache_pred", "")
            scores_gt = [score_vs_ref(o, answers, metric, all_cls) for o in outputs]
            scores_fc = [score_vs_single(o, fc_pred, metric, all_cls) for o in outputs]
            row = {
                "sample_id": sid,
                "action_outputs": outputs,
                "action_scores_gt": scores_gt,
                "action_scores_fc": scores_fc,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Appended {len(missing_act)} action rows to {budget_out}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
