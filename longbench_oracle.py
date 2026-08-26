"""Oracle KV-cache evaluation on LongBench.

Two-pass per example:
  Pass 1 – full-cache generation. Decode-step attention is captured (via an
            SDPA monkey-patch) and accumulated to derive the per-layer
            "post-generation reference set" K_post (top-(budget - local) keys
            per kv-head).
  Pass 2 – fresh prefill+decode through CompressedCache + OracleSelector
            with K_post as precomputed indices. Framework handles
            position_ids/RoPE/cache layout natively.

Multi-GPU / cross-server pattern matches longbench_RL.py:
  --shard_count S --shard_id i  : round-robin task partition across S shards
                                   (use across servers; deterministic shuffle)
  --gpus_per_model G            : GPUs per model instance (1 for 1B)
  CUDA_VISIBLE_DEVICES sets      : which GPUs this shard process can use; the
                                   shard launches one subprocess per GPU group.

Standard 5-shard split (eslab17 + eslab19, weighted 8:12 ≈ 40:60):
  eslab17  shard 0  CUDA_VISIBLE_DEVICES=0,1,2,3
  eslab17  shard 1  CUDA_VISIBLE_DEVICES=4,5,6,7
  eslab19  shard 2  CUDA_VISIBLE_DEVICES=0,1,2
  eslab19  shard 3  CUDA_VISIBLE_DEVICES=3,4,5
  eslab19  shard 4  CUDA_VISIBLE_DEVICES=6,7

Outputs:
  result_txt/pred/<budget>/oracle_<model>/<dataset>.jsonl   (single, append-merge)
"""
import os, json, time, argparse
import multiprocessing as mp
import random as _random
from collections import defaultdict

import torch
import torch.nn.functional as F


LOCAL_BUDGET = 16
MAX_SEQ_LEN  = 32768
SEED = 42

DATASETS = [
    "lcc", "repobench-p",
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "samsum", "trec", "triviaqa",
    "passage_count", "passage_retrieval_en",
]
CHAT_DATASETS = {"lcc", "repobench-p", "trec", "triviaqa", "samsum"}


# ── Worker subprocess ─────────────────────────────────────────────────────────

def _oracle_worker(
    worker_id, gpu_group, model_name, budget, max_length, d2l,
    task_queue, result_queue, total_tasks,
):
    """Subprocess pinned to gpu_group; pulls tasks from queue, generates oracle pred."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_group)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_grad_enabled(False)

    print(f"[w{worker_id}] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}",
          flush=True)

    import json as _json
    from transformers import AutoTokenizer
    from utils import CompressionConfig, load_compressed_lm

    with open("config/model2path.json") as f:
        model_path = _json.load(f)[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # v5 plugin: Pass 1 monkey-patches F.scaled_dot_product_attention (the "waits"
    # attention fn calls it, so capture still works); Pass 2 uses CompressedCache +
    # OracleSelector via init_cache(oracle cfg). (Completed: B=128/256/512 = 30.22/31.01/31.40.)
    model = load_compressed_lm(model_path, dtype=torch.bfloat16, device_map="auto")
    device = next(model.parameters()).device
    eos_ids = {tokenizer.eos_token_id}

    # ── SDPA monkey-patch (Pass 1 only) for decode attention capture ──
    _ORIG_SDPA = F.scaled_dot_product_attention
    state = {"active": False, "accum": None, "call_idx": 0,
             "n_layers": 0, "prefill_len": 0}

    def capturing_sdpa(q, k, v, attn_mask=None, dropout_p=0.0,
                       is_causal=False, scale=None, **kwargs):
        if state["active"] and q.shape[2] == 1 and state["call_idx"] < state["n_layers"]:
            D = q.shape[3]
            s = scale if scale is not None else (D ** -0.5)
            logit = (q @ k.transpose(-2, -1)).squeeze(2).float() * s
            probs = logit.softmax(dim=-1)
            acc = state["accum"][state["call_idx"]]
            H_kv = acc.shape[0]
            H_q = q.shape[1]
            group = H_q // H_kv
            L_use = min(state["prefill_len"], k.shape[2], acc.shape[1])
            prob_kv = probs[0, :, :L_use].reshape(H_kv, group, L_use).sum(1)
            acc[:, :L_use].add_(prob_kv.detach())
        state["call_idx"] += 1
        return _ORIG_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale, **kwargs)

    def gather_oracle_indices(input_ids, max_gen):
        cfg = model.config
        n_layers = cfg.num_hidden_layers
        H_kv = cfg.num_key_value_heads
        seq_len = input_ids.shape[1]
        sel_budget = max(1, budget - LOCAL_BUDGET)
        head_len = seq_len - LOCAL_BUDGET

        state["active"] = True
        state["n_layers"] = n_layers
        state["prefill_len"] = head_len
        state["accum"] = [
            torch.zeros(H_kv, head_len, dtype=torch.float32, device=device)
            for _ in range(n_layers)
        ]
        F.scaled_dot_product_attention = capturing_sdpa
        try:
            model.init_cache(None)
            with torch.inference_mode():
                state["call_idx"] = 0
                out = model(input_ids, use_cache=True)
                past_kv = out.past_key_values
                next_tok = out.logits[:, -1:].argmax(-1)
                del out
                for _ in range(max_gen):
                    state["call_idx"] = 0
                    if next_tok.item() in eos_ids:
                        break
                    out = model(next_tok, past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
                    next_tok = out.logits[:, -1:].argmax(-1)
                    del out
                del past_kv
        finally:
            F.scaled_dot_product_attention = _ORIG_SDPA
            state["active"] = False

        indices = []
        for layer_idx in range(n_layers):
            scores = state["accum"][layer_idx]
            k = min(sel_budget, head_len)
            _, idx = scores.topk(k, dim=1)
            idx, _ = idx.sort(dim=1)
            indices.append(idx.unsqueeze(0).to(torch.int64).cpu())
        state["accum"] = None
        return indices

    def oracle_generate(input_ids, oracle_indices, max_gen, dataset):
        cfg_obj = CompressionConfig()
        cfg_obj["compression_method"] = "oracle"
        cfg_obj["total_budget"] = int(budget)
        cfg_obj["recent_budget"] = LOCAL_BUDGET
        cfg_obj["oracle_indices"] = oracle_indices
        model.init_cache(cfg_obj)

        attention_mask = torch.ones_like(input_ids).to(device)
        ctx = int(input_ids.shape[-1])

        with torch.inference_mode():
            if dataset == "samsum":
                output = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=max_gen, num_beams=1, do_sample=False,
                    min_length=ctx + 1,
                    eos_token_id=[tokenizer.eos_token_id,
                                  tokenizer.encode("\n", add_special_tokens=False)[-1]],
                    pad_token_id=tokenizer.eos_token_id,
                    tokenizer=tokenizer, stop_strings="[/INST]",
                    num_logits_to_keep=1,
                )[0]
            else:
                output = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=max_gen, num_beams=1, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    tokenizer=tokenizer, stop_strings="[/INST]",
                    num_logits_to_keep=1,
                )[0]
        return tokenizer.decode(output[ctx:], skip_special_tokens=True)

    # ── Main loop: pull tasks from queue ──
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            dataset = task["dataset"]
            sample_idx = int(task["sample_idx"])
            ex = task["json_obj"]

            prompt = ex["input_prompt"]
            raw = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(raw) > max_length:
                h = max_length // 2
                prompt = (tokenizer.decode(raw[:h], skip_special_tokens=True) +
                          tokenizer.decode(raw[-h:], skip_special_tokens=True))
            if dataset not in CHAT_DATASETS:
                prompt = f"[INST]{prompt}[/INST]"

            input_ids = tokenizer(prompt, truncation=False, return_tensors="pt"
                                  ).input_ids.to(device)
            max_gen = int(d2l.get(dataset, 64))

            t0 = time.time()
            try:
                oracle_idx = gather_oracle_indices(input_ids, max_gen)
                pred = oracle_generate(input_ids, oracle_idx, max_gen, dataset)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                pred = ""
                print(f"[w{worker_id}] OOM on {dataset}#{sample_idx}", flush=True)

            torch.cuda.empty_cache()
            result_queue.put((
                dataset, sample_idx,
                {
                    "pred": pred,
                    "answers":     ex.get("answers", []),
                    "all_classes": ex.get("all_classes", []),
                    "length":      ex.get("length"),
                },
                time.time() - t0,
                input_ids.shape[1],
            ))
    except Exception as exc:  # noqa
        result_queue.put(("__error__", f"worker {worker_id} gpus={gpu_group}: {exc}"))


# ── Task list builder ────────────────────────────────────────────────────────

def _build_tasks(longbench_dir, datasets, d2l):
    tasks = []
    for dataset in datasets:
        if dataset not in d2l:
            continue
        path = os.path.join(longbench_dir, f"{dataset}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                tasks.append({
                    "dataset": dataset,
                    "sample_idx": idx,
                    "json_obj": json.loads(line),
                })
    return tasks


# ── Main: shard + per-GPU worker spawn ────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          type=str, default="llama3-1b")
    p.add_argument("--budget",         type=int, default=128)
    p.add_argument("--datasets",       type=str, default="")
    p.add_argument("--shard_id",       type=int, default=0)
    p.add_argument("--shard_count",    type=int, default=1)
    p.add_argument("--gpus_per_model", type=int, default=1)
    p.add_argument("--output_dir",     type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)

    with open("config/model2maxlen.json") as f:
        max_length = int(json.load(f)[args.model.split("_")[0].lower()])
    with open("config/dataset2maxlen.json") as f:
        d2l = json.load(f)

    target = ([d.strip() for d in args.datasets.split(",") if d.strip()]
              if args.datasets else DATASETS)
    target = [d for d in target if d in d2l]

    output_dir = args.output_dir or f"result_txt/pred/{args.budget}/oracle_{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    longbench_dir = os.path.join("datasets", "longbench")
    tasks = _build_tasks(longbench_dir, target, d2l)
    if not tasks:
        print("No tasks found."); return

    # Length-balanced shuffle (deterministic across shards)
    _random.Random(SEED).shuffle(tasks)

    n_total = len(tasks)
    if args.shard_count > 1:
        tasks = [t for i, t in enumerate(tasks)
                 if i % args.shard_count == args.shard_id]
    n_shard = len(tasks)
    print(f"shard {args.shard_id+1}/{args.shard_count}: {n_shard}/{n_total} tasks",
          flush=True)

    # ── Resume: skip tasks whose pred already in the merged JSONL ──
    done_keys = set()
    for ds in target:
        f = os.path.join(output_dir, f"{ds}.jsonl")
        if os.path.exists(f):
            with open(f) as fp:
                for line in fp:
                    try:
                        r = json.loads(line)
                        done_keys.add((ds, int(r.get("sample_idx", -1))))
                    except Exception:
                        pass
    if done_keys:
        n_before = len(tasks)
        tasks = [t for t in tasks
                 if (t["dataset"], t["sample_idx"]) not in done_keys]
        print(f"resume: {n_before - len(tasks)} already done, {len(tasks)} to go",
              flush=True)
    n_remaining = len(tasks)

    # ── Set up GPU groups ──
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        all_gpu_ids = [int(g) for g in visible.split(",") if g.strip()]
    else:
        all_gpu_ids = list(range(torch.cuda.device_count()))
    gpus_per = max(1, int(args.gpus_per_model))
    gpu_groups = [all_gpu_ids[i:i + gpus_per]
                  for i in range(0, len(all_gpu_ids), gpus_per)]
    gpu_groups = [g for g in gpu_groups if len(g) == gpus_per]
    print(f"gpu_groups={gpu_groups}", flush=True)

    if n_remaining == 0:
        print("nothing to do.", flush=True)
        return

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for t in tasks: task_queue.put(t)
    for _ in gpu_groups: task_queue.put(None)

    procs = []
    for wid, gg in enumerate(gpu_groups):
        p = ctx.Process(
            target=_oracle_worker,
            args=(wid, gg, args.model, int(args.budget), max_length, d2l,
                  task_queue, result_queue, n_remaining),
        )
        p.start()
        procs.append(p)

    # ── Drain result queue, write per-dataset JSONL with progress n/m ──
    started = time.time()
    done = 0
    while done < n_remaining:
        item = result_queue.get()
        if item[0] == "__error__":
            for p in procs:
                p.terminate()
            raise RuntimeError(item[1])
        dataset, sample_idx, payload, dt, L = item
        # write to per-dataset jsonl
        out_jsonl = os.path.join(output_dir, f"{dataset}.jsonl")
        with open(out_jsonl, "a", encoding="utf-8") as f:
            json.dump({"sample_idx": sample_idx, "dataset": dataset, **payload},
                      f, ensure_ascii=False)
            f.write("\n")
        done += 1
        elapsed = max(1e-6, time.time() - started)
        rate = done / elapsed
        eta = int((n_remaining - done) / rate) if rate > 0 else 0
        print(f"[shard {args.shard_id+1}/{args.shard_count}] "
              f"{done}/{n_remaining} ({100.0*done/n_remaining:.1f}%) "
              f"| {dataset}#{sample_idx} L={L} t={dt:.1f}s "
              f"| {rate:.2f}/s ETA {eta}s "
              f"| pred={payload['pred'][:30]!r}",
              flush=True)

    for p in procs:
        p.join(timeout=60)

    print(f"[shard {args.shard_id+1}/{args.shard_count}] done.", flush=True)


if __name__ == "__main__":
    main()
