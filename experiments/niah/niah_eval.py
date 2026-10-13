"""NIAH with prefill-time KV compression (Section 4.3, Figure 5).

    python experiments/niah/niah_eval.py --data datasets/niah_single.jsonl --model llama3-8b --budget 128 \
        --methods full,qtrust,snapkv,snapkv_pool,tova,h2o --out runs/niah_llama3-8b [--shard 0/4]
Settings match longbench.py (16 recent tokens, no sinks, observation window 16). The paper's NIAH Q-Trust run uses
the request-queries-weighted-1 form of Eq. (6) (window_one), tau = 1 and the Triton kernel. Greedy, 24 new tokens.
`correct` is exact match of the 7-digit answer; Figure 5 uses the soft digit score of fig_niah.py.
"""
import argparse, json, os, sys
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import chdir_root

METHODS = {"full": None,
           "qtrust": dict(compression_method="qtrust", observation_window=16, tau=1.0, kernel=True, window_one=True),
           "snapkv": dict(compression_method="snap", observation_window=16),
           "tova": dict(compression_method="snap", observation_window=1),
           "h2o": dict(compression_method="snap", observation_window=32768)}
METHODS["snapkv_pool"] = dict(METHODS["snapkv"], score_pool=7)          # SnapKV with its max pooling (kernel 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="llama3-8b"); ap.add_argument("--methods", default="full,qtrust,snapkv,snapkv_pool,tova,h2o")
    ap.add_argument("--budget", type=int, default=128); ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    data, out = os.path.abspath(a.data), os.path.abspath(a.out)
    chdir_root()
    import utils as U
    from kvcompress.compress import CompressionConfig
    torch.set_grad_enabled(False)
    rows = [json.loads(l) for l in open(data)]
    si, sn = map(int, a.shard.split("/")); rows = rows[si::sn]
    model, tok = U.load_model(a.model)
    os.makedirs(out, exist_ok=True)
    for m in a.methods.split(","):
        fn = f"{out}/{m}_{a.budget}_shard{si}.jsonl"
        done = {json.loads(l)["id"] for l in open(fn)} if os.path.exists(fn) else set()
        cfg = None if METHODS[m] is None else CompressionConfig(total_budget=a.budget, recent_budget=16, n_sink=0, **METHODS[m])
        with open(fn, "a") as f:
            for r in tqdm(rows, desc=f"{m} shard {si}"):
                if r["id"] in done:
                    continue
                x = tok(U.build_chat_prompt(r["prompt"], a.model, tok, "niah"), return_tensors="pt").input_ids.to(model.device)
                model.init_cache(cfg)
                with torch.inference_mode():
                    g = model.generate(input_ids=x, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id,
                                       tokenizer=tok, stop_strings=U.chat_stop_strings(a.model))
                pred = tok.decode(g[0, x.shape[1]:], skip_special_tokens=True)
                f.write(json.dumps(dict(id=r["id"], length=r["length"], depth=r["depth"], n_tok=int(x.shape[1]),
                                        pred=pred, correct=r["answer"] in pred)) + "\n"); f.flush()
                torch.cuda.empty_cache()
    print("[niah] SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
