"""Prefill cost of Q-Trust (Table 3 and the tau table, Llama-3.1-8B, one RTX 3090; 32K on two GPUs).

    python experiments/cost/prefill_cost.py --mode table --docs <docs> --lens 4096,8192,16384 --out cost_1gpu.json
    CUDA_VISIBLE_DEVICES=0,1 python experiments/cost/prefill_cost.py --mode table --docs <docs> --lens 32768 --out cost_32k.json
    python experiments/cost/prefill_cost.py --mode tau --docs <docs> --lens 16384 --out cost_tau.json

--mode table: plain prefill vs Q-Trust (tau = 1) on the PyTorch path and with the Triton kernels.
--mode tau:   plain prefill vs Q-Trust (kernel) at tau in {0, 0.1, 0.25, 0.5, 1, 2, 4}.
The same documents are cut to every length, so only the length changes. Each call is prefill + 1 generated token,
CUDA-synchronized over all visible GPUs, 1 warm-up + `reps` timed runs, median per document. Overhead = median over
documents of (time / plain prefill - 1). --docs: a JSON list of [name, token_ids] (Llama-3.1 tokenizer), or a jsonl
file whose rows carry a "context" text (the first 5 rows with at least max(lens) tokens are used).
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import chdir_root


def sync():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def load_docs(path, tok, need):
    if path.endswith(".json"):
        return json.load(open(path))
    docs = []
    for i, l in enumerate(open(path)):
        ids = tok(json.loads(l)["context"], add_special_tokens=False).input_ids
        if len(ids) >= need:
            docs.append([f"doc{i}", ids[:need]])
        if len(docs) == 5:
            break
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["table", "tau"], default="table")
    ap.add_argument("--docs", required=True); ap.add_argument("--model", default="llama3-8b")
    ap.add_argument("--lens", default="4096,8192,16384"); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    docs_path, out_path = os.path.abspath(a.docs), os.path.abspath(a.out)
    chdir_root()
    import utils as U
    from kvcompress.compress import CompressionConfig
    model, tok = U.load_model(a.model)
    lens = list(map(int, a.lens.split(",")))
    docs = load_docs(docs_path, tok, max(lens))
    base = dict(compression_method="qtrust", observation_window=16, total_budget=128, recent_budget=16)
    if a.mode == "table":
        settings = [("prefill", None), ("Q-Trust PyTorch", dict(base, tau=1.0)), ("Q-Trust kernel", dict(base, tau=1.0, kernel=True))]
    else:
        settings = [("prefill", None)] + [(f"tau={t:g}", dict(base, tau=t, kernel=True)) for t in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)]
    out = dict(gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], docs=[d for d, _ in docs], res={})
    for L in lens:
        res = {n: [] for n, _ in settings}
        for d, ids in tqdm(docs, desc=f"L={L}"):
            x = torch.tensor([ids[:L]], device=model.device)
            for name, kw in settings:
                cfg = None if kw is None else CompressionConfig(**kw)
                ts = []
                for _ in range(a.reps + 1):
                    model.init_cache(cfg)
                    sync(); t0 = time.perf_counter()
                    with torch.inference_mode():
                        model.generate(x, max_new_tokens=1, do_sample=False, pad_token_id=tok.eos_token_id)
                    sync(); ts.append(time.perf_counter() - t0)
                res[name].append(float(np.median(ts[1:])))
            torch.cuda.empty_cache()
        b = np.array(res["prefill"])
        out["res"][L] = dict(per_prompt=res, prefill_s=float(b.mean()),
                             overhead_pct={n: float(100 * (np.median(np.array(v) / b) - 1)) for n, v in res.items()})
        print(f"L={L}: prefill {b.mean():.2f}s, " + ", ".join(f"{n} {v:+.1f}%" for n, v in out["res"][L]["overhead_pct"].items()), flush=True)
        json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
