"""Build (or rebuild) the LongBench fast-eval index with full text predictions.

For every LB sample × every 13 sigmoid action, runs model inference and stores:
  {ds}/preds:       list[N][13]  — actual generated text (string)
  {ds}/scores:      Tensor[N,13] — float32 metric scores (0-100)
  {ds}/answers:     list[N]
  {ds}/all_classes: list[N]
  {ds}/lengths:     list[N]
  datasets:         list[str]

Multi-GPU: one worker process per GPU (spawn). Resumable.

Usage:
    # single server, all GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 python script/build_lb_index.py \\
        --model llama3-1b --out runs/fast_lb_eval/index.pt

    # cross-server shard (run on each server independently, merge after)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python script/build_lb_index.py \\
        --model llama3-1b --shard_id 0 --shard_count 2 \\
        --out runs/fast_lb_eval/index_shard0.pt
"""
import argparse, json, os, sys, time
import multiprocessing as mp
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from RL.a2sf_model import SIGMOID_A_VALUES, SIGMOID_B_VALUES
from longbench_eval import scorer as lb_scorer

DATASETS = [
    "lcc", "repobench-p",
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "samsum", "trec", "triviaqa",
    "passage_count", "passage_retrieval_en",
]
CHAT_SKIP = {"lcc", "repobench-p", "trec", "triviaqa", "samsum"}


# ── worker ────────────────────────────────────────────────────────────────────

def _worker(worker_id, gpu_id, model_name, budget,
            max_length, d2l, task_queue, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_grad_enabled(False)
    print(f"[w{worker_id}] GPU={gpu_id} starting", flush=True)

    from utils import load_model
    from utils_real_drop import KVLlamaForCausalLM

    with open("config/model2path.json") as f:
        model_path = json.load(f)[model_name]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = KVLlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()
    device = next(model.parameters()).device

    from utils import CompressionConfig
    def make_cfg(a, b):
        cfg = CompressionConfig()
        cfg.compression_method = "sigmoid"
        cfg.total_budget = int(budget)
        cfg.local_ratios = 0.125
        cfg.a = torch.tensor([a], dtype=torch.float32, device=device)
        cfg.b = torch.tensor([b], dtype=torch.float32, device=device)
        return cfg

    try:
        while True:
            task = task_queue.get()
            if task is None:
                break

            dataset   = task["dataset"]
            sample_idx= int(task["sample_idx"])
            ex        = task["json_obj"]

            prompt_raw = ex["input_prompt"]
            raw_ids    = tokenizer(prompt_raw, truncation=False,
                                   return_tensors="pt").input_ids[0]
            if len(raw_ids) > max_length:
                h = max_length // 2
                prompt_raw = (tokenizer.decode(raw_ids[:h], skip_special_tokens=True)
                            + tokenizer.decode(raw_ids[-h:], skip_special_tokens=True))

            if dataset not in CHAT_SKIP and "llama" in model_name:
                prompt = f"[INST]{prompt_raw}[/INST]"
            else:
                prompt = prompt_raw

            encoded    = tokenizer(prompt, truncation=False, return_tensors="pt")
            input_ids  = encoded.input_ids.to(device)
            attn_mask  = encoded.attention_mask.to(device)
            ctx_len    = int(input_ids.shape[-1])
            max_gen    = int(d2l.get(dataset, 64))

            gen_kwargs = dict(
                input_ids=input_ids, attention_mask=attn_mask,
                max_new_tokens=max_gen, num_beams=1, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                tokenizer=tokenizer, stop_strings="[/INST]",
                num_logits_to_keep=1,
            )
            if dataset == "samsum":
                gen_kwargs["min_length"]   = ctx_len + 1
                gen_kwargs["eos_token_id"] = [
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ]

            t0 = time.time()
            preds = []
            for a, b in zip(SIGMOID_A_VALUES, SIGMOID_B_VALUES):
                model.init_cache(make_cfg(a, b))
                with torch.inference_mode():
                    out = model.generate(**gen_kwargs)
                preds.append(tokenizer.decode(out[0, ctx_len:], skip_special_tokens=True))
            torch.cuda.empty_cache()

            result_queue.put((dataset, sample_idx, preds, time.time() - t0))

    except Exception as exc:
        import traceback
        result_queue.put(("__error__",
                          f"worker {worker_id} gpu={gpu_id}: {exc}\n{traceback.format_exc()}"))


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="llama3-1b")
    p.add_argument("--budget",      type=int, default=128)
    p.add_argument("--out",         default="runs/fast_lb_eval/index.pt")
    p.add_argument("--shard_id",    type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--datasets",    type=str, default="",
                   help="Comma-separated dataset names to process (default: all).")
    p.add_argument("--max_per_ds",  type=int, default=0,
                   help="Limit samples per dataset for testing (0=all).")
    return p.parse_args()


def main():
    args = parse_args()

    with open("config/model2maxlen.json") as f:
        max_length = int(json.load(f)[args.model.split("_")[0]])
    with open("config/dataset2maxlen.json") as f:
        d2l = json.load(f)

    # Build task list
    target_ds = ([d.strip() for d in args.datasets.split(",") if d.strip()]
                 if args.datasets else DATASETS)
    lb_dir = os.path.join("datasets", "longbench")
    tasks  = []
    for ds in target_ds:
        path = os.path.join(lb_dir, f"{ds}.jsonl")
        if not os.path.exists(path):
            print(f"[skip] {ds}")
            continue
        with open(path) as f:
            for idx, line in enumerate(f):
                if line.strip():
                    if args.max_per_ds > 0 and idx >= args.max_per_ds:
                        break
                    tasks.append({"dataset": ds, "sample_idx": idx,
                                  "json_obj": json.loads(line)})
    print(f"Total tasks: {len(tasks)}")

    # Shard
    if args.shard_count > 1:
        import random as _r
        rng = _r.Random(42); rng.shuffle(tasks)
        tasks = [t for i, t in enumerate(tasks) if i % args.shard_count == args.shard_id]
        print(f"Shard {args.shard_id+1}/{args.shard_count}: {len(tasks)} tasks")

    # Load partial results if out exists
    partial: dict = {}          # (ds, sample_idx) -> list[13] preds
    if os.path.exists(args.out):
        saved = torch.load(args.out, map_location="cpu", weights_only=False)
        for ds in saved.get("datasets", []):
            if f"{ds}/preds" in saved:
                for i, preds in enumerate(saved[f"{ds}/preds"]):
                    if preds is not None:
                        partial[(ds, i)] = preds
        print(f"Resumed: {len(partial)} already done")

    tasks = [t for t in tasks
             if (t["dataset"], t["sample_idx"]) not in partial]
    print(f"Remaining: {len(tasks)} tasks")

    if not tasks:
        print("Nothing to do.")
        return

    # GPU groups
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    gpu_ids = ([int(g) for g in visible.split(",") if g.strip()]
               if visible else list(range(torch.cuda.device_count())))
    print(f"GPUs: {gpu_ids}")

    ctx = mp.get_context("spawn")
    tq  = ctx.Queue()
    rq  = ctx.Queue()
    for t in tasks: tq.put(t)
    for _ in gpu_ids: tq.put(None)

    procs = []
    for wid, gid in enumerate(gpu_ids):
        p = ctx.Process(target=_worker,
                        args=(wid, gid, args.model, args.budget,
                              max_length, d2l, tq, rq))
        p.start(); procs.append(p)

    # Collect results into ds -> {idx -> preds}
    ds_preds: dict = defaultdict(dict)
    for (ds, idx), preds in partial.items():
        ds_preds[ds][idx] = preds

    total = len(tasks); t_start = time.time(); done = 0
    while done < total:
        item = rq.get()
        if item[0] == "__error__":
            for p in procs: p.terminate()
            raise RuntimeError(item[1])
        ds, idx, preds, dur = item
        ds_preds[ds][idx] = preds
        done += 1
        rate = done / max(1e-6, time.time() - t_start)
        eta  = int((total - done) / max(rate, 1e-6))
        print(f"  {done}/{total} {ds}#{idx} {dur:.1f}s  ETA {eta}s", flush=True)

        # Save checkpoint every 50 completed tasks
        if done % 50 == 0:
            _save(args, ds_preds, d2l)

    for p in procs: p.join(timeout=60)

    _save(args, ds_preds, d2l)
    print(f"\nDone → {args.out}")


def _save(args, ds_preds: dict, d2l: dict):
    """Build and save index.pt with preds + scores."""
    lb_dir = os.path.join("datasets", "longbench")
    out    = {}
    ds_list = []

    for ds in DATASETS:
        path = os.path.join(lb_dir, f"{ds}.jsonl")
        if not os.path.exists(path): continue
        with open(path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        N = len(rows)
        if ds not in ds_preds: continue

        preds_by_idx = ds_preds[ds]
        all_preds    = [preds_by_idx.get(i) for i in range(N)]  # None if not done

        # Compute scores only for completed entries
        scores_mat = torch.zeros(N, len(SIGMOID_A_VALUES), dtype=torch.float32)
        for i, preds in enumerate(all_preds):
            if preds is None: continue
            row = rows[i]
            answers    = row.get("answers", []) or []
            all_cls    = row.get("all_classes", []) or []
            for j, pred in enumerate(preds):
                try:
                    s = lb_scorer(ds, [pred], [answers], all_cls)
                except Exception:
                    s = 0.0
                scores_mat[i, j] = float(s)

        answers    = [r.get("answers",    []) for r in rows]
        all_cls_l  = [r.get("all_classes",[]) for r in rows]
        lengths    = [r.get("length")        for r in rows]

        out[f"{ds}/preds"]      = all_preds
        out[f"{ds}/scores"]     = scores_mat
        out[f"{ds}/answers"]    = answers
        out[f"{ds}/all_classes"]= all_cls_l
        out[f"{ds}/lengths"]    = lengths
        ds_list.append(ds)

    out["datasets"] = ds_list
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"  [checkpoint] saved {sum(1 for v in ds_preds.values() for vv in v.values() if vv)} preds → {args.out}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
