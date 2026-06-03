"""Multi-GPU input-parallel computation of obs2 figure data.

Builds, per dataset and per budget, the per-prompt arrays needed by:
  - obs1_jaccard_recovery.{pdf,png}  (uses metrics.npz["br"] and coord_descent.npz["j_weight"])
  - obs1_sigmoid_band.{pdf,png}      (uses coord_descent.npz["w_cd"], "j_weight"])

Compared to the original optimal.py + coord_descent.py:
  - One model per GPU (no model-parallel device_map). 16 model instances
    parallel across local + eslab19 (or just local).
  - Tasks (one prompt at a time) round-robined across worker subprocesses
    via mp.Queue. Cross-server: --shard_id / --shard_count round-robin too
    (matches longbench_oracle.py / longbench_RL.py pattern).
  - Computes ONLY br and forward-greedy w_cd / j_cd / j_weight (cheap subset
    of optimal.py).
  - MAX_WINDOW configurable via CLI (default 256, was 128).

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m experiments.temporal_bias.parallel_obs2 \
      --budget 128 --max_window 256 --n_items 40 \
      --shard_id 0 --shard_count 1 \
      --datasets samsum,qasper,hotpotqa,gov_report

Sequential 3-budget run is the caller's job: invoke this script three times.
Outputs land in:
  experiments/temporal_bias/plots[/b<budget>]/<task>/<dataset>/coord_descent.npz
  experiments/temporal_bias/plots[/b<budget>]/<task>/<dataset>/metrics.npz
"""
import os, sys, json, time, argparse, random
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

WORKPATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(WORKPATH))
sys.path.append(ROOT_PATH)

# Import existing helpers (these don't trigger heavy CUDA work on import)
from experiments.temporal_bias.optimal import (
    AttentionCollector, gqa_topk, jaccard,
    analyze_block_hit_rate,
    LOCAL_RATIO, MAX_SEQ_LEN, LENGTH_MIN, LENGTH_MAX, MODEL_NAME,
    TASK_GROUP, SEED,
)


# Forward-greedy search constants (mirror coord_descent.py)
GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CHUNK = 4
MIN_REL_GAIN = 0.01


def _jaccard_gpu(a_idx, b_idx, seq_len):
    L, H, _ = a_idx.shape
    dev = a_idx.device
    a = torch.zeros(L, H, seq_len, dtype=torch.bool, device=dev)
    b = torch.zeros(L, H, seq_len, dtype=torch.bool, device=dev)
    a.scatter_(2, a_idx, True)
    b.scatter_(2, b_idx, True)
    inter = (a & b).sum(2).float()
    union = (a | b).sum(2).float().clamp(min=1)
    return float((inter / union).mean().item())


def coord_descent(prefill, answer_idx, budget, kv_h, gs, seq_len, chunk=CHUNK):
    """Forward-greedy block-wise coefficient search.
    Returns (w (G,), final_jaccard, j_weight (G,) running J after each chunk).
    """
    L, H, W, S = prefill.shape
    assert W % chunk == 0, f"W={W} not divisible by chunk={chunk}"
    G = W // chunk
    local_b = max(1, int(budget * LOCAL_RATIO))
    sel_b = budget - local_b

    prefill_w = prefill.permute(2, 0, 1, 3).contiguous()
    prefill_d = prefill_w.flip(0)
    prefill_chunked = prefill_d.view(G, chunk, L, H, S).sum(dim=1).contiguous()

    w = np.zeros(G, dtype=np.float32)
    j_weight = np.zeros(G, dtype=np.float32)
    accumulated = torch.zeros_like(prefill_chunked[0])
    temp = accumulated.clone()
    saved_local = torch.empty_like(temp[:, :, -local_b:])

    def eval_temp():
        saved_local.copy_(temp[:, :, -local_b:])
        temp[:, :, -local_b:] = temp.max()
        idx = gqa_topk(temp, sel_b, kv_h, gs)
        j = _jaccard_gpu(answer_idx, idx, seq_len)
        temp[:, :, -local_b:].copy_(saved_local)
        return j

    base_j = -1.0
    for g in range(G):
        block = prefill_chunked[g]
        best_c, best_j = GRID[0], -1.0
        j_zero = -1.0
        prev_c = 0.0
        temp.copy_(accumulated)
        for i, c in enumerate(GRID):
            delta = c - prev_c
            if i == 0:
                if abs(delta) > 1e-9:
                    temp.add_(block, alpha=delta)
            else:
                temp.add_(block, alpha=delta)
            j = eval_temp()
            if i == 0:
                j_zero = j
            if j > best_j + 1e-9:
                best_j = j
                best_c = c
            prev_c = c

        ref = max(j_zero, 1e-9)
        if best_c > 0.0 and (best_j - j_zero) < MIN_REL_GAIN * ref:
            best_c = 0.0
            best_j = j_zero

        if best_c > 1e-9:
            accumulated.add_(block, alpha=best_c)
        w[g] = best_c
        base_j = best_j
        j_weight[g] = base_j

    return w, base_j, j_weight


# ── Worker subprocess ─────────────────────────────────────────────────────────

def _worker(worker_id, gpu_id, model_path, budget, max_window, max_new_default,
            task_queue, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_grad_enabled(False)

    print(f"[w{worker_id}] CUDA_VISIBLE_DEVICES={gpu_id}", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map={"": 0},      # entire model on the single visible GPU
        attn_implementation="sdpa",
    ).eval()
    device = next(model.parameters()).device

    collector = AttentionCollector(model, max_window)

    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            dataset_name, sample_idx, prompt, max_new = task
            t0 = time.time()
            try:
                enc = tokenizer(f"[INST]{prompt}[/INST]", return_tensors="pt")
                input_ids = enc.input_ids.to(device)
                if input_ids.size(1) > MAX_SEQ_LEN:
                    half = MAX_SEQ_LEN // 2
                    input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)
                seq_len = int(input_ids.size(1))

                collector.reset(seq_len)
                with torch.no_grad():
                    out = model(input_ids, use_cache=True, num_logits_to_keep=1)
                    past_kv = out.past_key_values
                    next_tok = out.logits[:, -1:].argmax(dim=-1)
                    del out
                    data = collector.compute_window_data(past_kv)
                    collector.set_decode()
                    for _ in range(max_new):
                        out = model(next_tok, past_key_values=past_kv,
                                    use_cache=True, output_attentions=True)
                        past_kv = out.past_key_values
                        next_tok = out.logits[:, -1:].argmax(dim=-1)
                        if next_tok.item() == tokenizer.eos_token_id:
                            del out; break
                        del out
                del past_kv

                prefill_attn = data["prefill_attn"]
                answer_score = collector.answer_score
                kv_h, gs = collector.num_kv_heads, collector.group_size
                answer_idx = gqa_topk(answer_score, budget, kv_h, gs)

                br = analyze_block_hit_rate(
                    prefill_attn, answer_idx, budget, kv_h, gs, seq_len)
                w_cd, j_cd, j_weight = coord_descent(
                    prefill_attn, answer_idx, budget, kv_h, gs, seq_len, chunk=CHUNK)

                result_queue.put((dataset_name, sample_idx, {
                    "br": br.astype(np.float32),
                    "w_cd": w_cd.astype(np.float32),
                    "j_cd": float(j_cd),
                    "j_weight": j_weight.astype(np.float32),
                    "seq_len": seq_len,
                }, time.time() - t0))

                del data, prefill_attn, answer_score
                torch.cuda.empty_cache()
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                result_queue.put(("__skip__", sample_idx, "OOM", 0.0))
            except Exception as e:
                result_queue.put(("__error__",
                                  f"w{worker_id} dataset={dataset_name} idx={sample_idx}: {e}"))
    except Exception as e:
        result_queue.put(("__error__", f"w{worker_id} fatal: {e}"))


# ── Master ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--budget",      type=int, default=128)
    p.add_argument("--max_window",  type=int, default=256)
    p.add_argument("--n_items",     type=int, default=40)
    p.add_argument("--datasets",    type=str, default="",
                   help="comma-separated subset; default = all in TASK_GROUP")
    p.add_argument("--shard_id",    type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    with open(os.path.join(ROOT_PATH, "config", "model2path.json")) as f:
        model_path = json.load(f)[MODEL_NAME]
    with open(os.path.join(ROOT_PATH, "config", "dataset2maxlen.json")) as f:
        d2l = json.load(f)

    longbench_dir = os.path.join(ROOT_PATH, "datasets", "longbench")
    dataset_prompts = defaultdict(list)
    for fname in os.listdir(longbench_dir):
        with open(os.path.join(longbench_dir, fname)) as f:
            for line in f:
                item = json.loads(line)
                if LENGTH_MIN <= item.get("length", 0) <= LENGTH_MAX:
                    dataset_prompts[item["dataset"]].append(item["input_prompt"])

    dataset_filter = (
        set([d.strip() for d in args.datasets.split(",") if d.strip()])
        if args.datasets else None
    )
    dataset2task, selected = {}, {}
    for task_name, datasets in TASK_GROUP.items():
        for d in datasets:
            dataset2task[d] = task_name
            if dataset_filter is not None and d not in dataset_filter:
                continue
            if d in dataset_prompts:
                pool = dataset_prompts[d]
                selected[d] = random.sample(pool, min(args.n_items, len(pool)))

    plot_dir = os.path.join(WORKPATH, "plots")
    if args.budget != 128:
        plot_dir = os.path.join(plot_dir, f"b{args.budget}")

    print(f"budget={args.budget}  max_window={args.max_window}  "
          f"n_items={args.n_items}  shard {args.shard_id+1}/{args.shard_count}",
          flush=True)
    print(f"datasets={list(selected.keys())}", flush=True)
    print(f"output dir: {plot_dir}", flush=True)

    # ── GPU group setup (one model per GPU) ──
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        all_gpu_ids = [int(g) for g in visible.split(",") if g.strip()]
    else:
        all_gpu_ids = list(range(torch.cuda.device_count()))
    if not all_gpu_ids:
        raise RuntimeError("no visible GPUs")
    print(f"GPUs={all_gpu_ids}", flush=True)

    # Build full task list (across all selected datasets), then shard, then dispatch
    all_tasks = []
    for ds, prompts in selected.items():
        max_new = int(d2l.get(ds, 512))
        for idx, p in enumerate(prompts):
            all_tasks.append((ds, idx, p, max_new))

    # Length-balanced shuffle (deterministic)
    random.Random(SEED).shuffle(all_tasks)

    n_total = len(all_tasks)
    if args.shard_count > 1:
        all_tasks = [t for i, t in enumerate(all_tasks)
                     if i % args.shard_count == args.shard_id]
    n_shard = len(all_tasks)
    print(f"shard tasks: {n_shard}/{n_total}", flush=True)

    # ── multiprocessing pool: one worker per GPU ──
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for t in all_tasks: task_queue.put(t)
    for _ in all_gpu_ids: task_queue.put(None)

    procs = []
    for wid, gpu_id in enumerate(all_gpu_ids):
        p = ctx.Process(
            target=_worker,
            args=(wid, gpu_id, model_path, int(args.budget), int(args.max_window),
                  512, task_queue, result_queue),
        )
        p.start()
        procs.append(p)

    # ── Drain results, accumulate per-dataset ──
    agg = defaultdict(lambda: defaultdict(list))   # agg[dataset][key] -> list of arrays/values
    started = time.time()
    done = 0
    while done < n_shard:
        item = result_queue.get()
        if item[0] == "__error__":
            for p in procs: p.terminate()
            raise RuntimeError(item[1])
        if item[0] == "__skip__":
            done += 1
            continue
        ds, idx, payload, dt = item
        agg[ds]["br"].append(payload["br"])
        agg[ds]["w_cd"].append(payload["w_cd"])
        agg[ds]["j_cd"].append(payload["j_cd"])
        agg[ds]["j_weight"].append(payload["j_weight"])
        agg[ds]["sample_idx"].append(idx)
        agg[ds]["seq_len"].append(payload["seq_len"])
        done += 1
        elapsed = max(1e-6, time.time() - started)
        rate = done / elapsed
        eta = int((n_shard - done) / rate) if rate > 0 else 0
        print(f"[shard {args.shard_id+1}/{args.shard_count}] "
              f"{done}/{n_shard} ({100.0*done/n_shard:.1f}%) "
              f"| {ds}#{idx} L={payload['seq_len']} t={dt:.1f}s "
              f"| {rate:.2f}/s ETA {eta}s",
              flush=True)

    for p in procs:
        p.join(timeout=60)

    # ── Save per-shard partial npz ──
    for ds, A in agg.items():
        task_name = dataset2task[ds]
        ds_dir = os.path.join(plot_dir, task_name.replace(" ", "_"), ds)
        os.makedirs(ds_dir, exist_ok=True)

        partial_path = os.path.join(ds_dir,
                                    f"shard_{args.shard_id}_of_{args.shard_count}.npz")
        np.savez_compressed(
            partial_path,
            br=np.stack(A["br"], axis=0),
            w_cd=np.stack(A["w_cd"], axis=0),
            j_cd=np.array(A["j_cd"], dtype=np.float32),
            j_weight=np.stack(A["j_weight"], axis=0),
            sample_idx=np.array(A["sample_idx"], dtype=np.int32),
            seq_len=np.array(A["seq_len"], dtype=np.int32),
            chunk=np.int32(CHUNK),
            window=np.int32(args.max_window),
        )
        print(f"saved partial: {partial_path}  N={len(A['br'])}", flush=True)

    # ── Master shard merges all partials into final npz ──
    if args.shard_id == 0 and args.shard_count > 1:
        # wait for siblings (presence of partial files); skip -- caller will trigger merge
        pass

    print(f"[shard {args.shard_id+1}/{args.shard_count}] done.", flush=True)


if __name__ == "__main__":
    main()
