"""N1: gold-NLL dense reward — teacher-forced NLL of the gold answer under a
compressed cache, per (prompt, action). Forward-only (no generation): ~10x cheaper
than the generation+metric recipe reward, continuous (no 62%-tie problem).

reward = -mean_token_NLL(gold | compressed prefix)   (higher = better preservation)
Also records the full-cache gold-NLL as a per-prompt reference (for normalization).

Output: <outdir>/nll_<budget>.jsonl  {sample_id, nll_full, nll_actions: [..]}
Resumable. Multi-GPU (spawn, 1 GPU per worker).

  python RL/nll_reward.py --input datasets/training/raw/recipe_v3_1b/common.jsonl \
      --outdir datasets/training/raw/recipe_v3_1b_nll --model llama3-1b \
      --gpus 0,1,2,3 --actions "0:1,0.01:128,1:16,10:1,10:16"
"""
import argparse
import json
import os
import sys
import time
import multiprocessing as mp
from typing import List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import CompressionConfig, load_model
from utils_real_drop.compress import CompressedCache


def _format_prompt(prompt, dataset_name, model_name):
    from RL.dataset import _format_prompt as fp
    return fp(prompt, dataset_name, model_name)


def gold_nll(model, tok, input_ids, gold_text, cfg):
    """NLL of gold under (optionally compressed) prefix. cfg None = full cache.
    Raw forward needs the CompressedCache passed explicitly (init_cache only wires
    the generate() wrapper; the attention pre-hook captures past_key_values)."""
    gold_ids = tok(str(gold_text), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    if gold_ids.shape[1] < 1:
        return None
    model.init_cache(cfg)
    with torch.inference_mode():
        past = CompressedCache(model.config, cfg) if cfg is not None else None
        out = (model(input_ids, past_key_values=past, use_cache=True) if past is not None
               else model(input_ids, use_cache=True))
        past = out.past_key_values
        out2 = model(gold_ids, past_key_values=past, use_cache=True)
        logits = torch.cat([out.logits[:, -1:, :], out2.logits[:, :-1, :]], dim=1).float()
        nll = F.cross_entropy(logits[0], gold_ids[0], reduction="mean")
    return float(nll)


def _worker(wid, gpu_ids, model_name, tq, rq, budget, actions):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    torch.set_grad_enabled(False)
    model, tok = load_model(model_name)
    print(f"[w{wid}] gpus={gpu_ids} start", flush=True)
    while True:
        task = tq.get()
        if task is None:
            break
        r = task
        sid = int(r["sample_id"])
        try:
            prompt = _format_prompt(str(r["input_prompt"]), str(r.get("dataset") or ""), model_name)
            ids = tok(prompt, truncation=False, return_tensors="pt").input_ids
            _maxin = int(os.environ.get("MAX_INPUT_TOKENS", "0") or "0")
            if _maxin and ids.shape[-1] > _maxin:
                ids = ids[:, -_maxin:]
            ids = ids.to(model.device)
            gold = (r.get("answers") or [""])[0]
            nll_full = gold_nll(model, tok, ids, gold, None)
            nlls = []
            for (a, b) in actions:
                cfg = CompressionConfig()
                cfg["compression_method"] = "waits"; cfg["total_budget"] = int(budget)
                cfg["recent_budget"] = 16; cfg["a"] = float(a); cfg["b"] = int(b)
                cfg["observation_window"] = int(b)
                nlls.append(gold_nll(model, tok, ids, gold, cfg))
            rq.put((sid, nll_full, nlls, ids.shape[1]))
        except Exception as e:
            rq.put((sid, None, None, str(e)))
    rq.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="llama3-1b")
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--actions", required=True, help="'a:b,a:b,...'")
    args = ap.parse_args()
    actions = [(float(p.split(":")[0]), float(p.split(":")[1])) for p in args.actions.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    out_p = os.path.join(args.outdir, f"nll_{args.budget}.jsonl")
    done = set()
    if os.path.exists(out_p):
        for l in open(out_p):
            done.add(int(json.loads(l)["sample_id"]))
    rows = [json.loads(l) for l in open(args.input)]
    todo = [r for r in rows if int(r["sample_id"]) not in done]
    print(f"{len(todo)}/{len(rows)} to score, actions={actions}")
    gpu_ids = [int(x) for x in args.gpus.split(",")]
    ctx = mp.get_context("spawn")
    tq, rq = ctx.Queue(), ctx.Queue()
    for r in sorted(todo, key=lambda x: -int(x.get("length", 0))):
        tq.put(r)
    for _ in gpu_ids:
        tq.put(None)
    procs = [ctx.Process(target=_worker, args=(i, [g], args.model, tq, rq, args.budget, actions))
             for i, g in enumerate(gpu_ids)]
    for p in procs: p.start()
    n_done, n_alive, t0 = 0, len(procs), time.time()
    with open(out_p, "a") as f:
        while n_alive > 0:
            item = rq.get()
            if item is None:
                n_alive -= 1; continue
            sid, nf, na, meta = item
            if nf is None:
                print(f"[err] sid={sid}: {meta}", flush=True); continue
            f.write(json.dumps({"sample_id": sid, "nll_full": nf, "nll_actions": na}) + "\n")
            f.flush()
            n_done += 1
            if n_done % 25 == 0:
                r = n_done / (time.time() - t0)
                print(f"[nll] {n_done}/{len(todo)} rate={r:.2f}/s eta={(len(todo)-n_done)/max(r,1e-9)/60:.0f}m", flush=True)
    for p in procs: p.join()
    print("DONE", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
