"""Share of query rows the second hop computes at each tau (Section 3.2 and the tau table, Llama-3.1-8B, 8K prompts).

    python experiments/cost/rows_computed.py --data datasets/longbench --per 3 --out rows.json

Per prompt and setting, one generate call of 1 token (= prefill + compression) after a warm-up. Records the wall
time, the mean over layers of the share of query rows computed, and the Jaccard overlap of the kept keys with
the dense (tau = 0) kept set, averaged over layers and KV heads. The paper's 48 prompts were long source documents
(first 3 of each of the 16 task types), truncated in the middle to 8,128 tokens.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import DATASETS, chdir_root

TAUS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory of <dataset>.jsonl with an input_prompt field")
    ap.add_argument("--per", type=int, default=3); ap.add_argument("--model", default="llama3-8b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data, out_path = os.path.abspath(a.data), os.path.abspath(a.out)
    chdir_root()
    import utils as U
    from kvcompress.compress import CompressionConfig, CompressedCache
    from kvcompress.scorers.twohop import QTrustScorer
    KEPT, ROWS = {}, []
    oc, oh = CompressedCache.compress, QTrustScorer.set_first_hop

    def rc(self, li, idx):
        if idx is not None:
            KEPT[li] = idx[0].detach().cpu()
        return oc(self, li, idx)

    def rh(self, scores):
        oh(self, scores)
        if self.kept_query_frac:
            ROWS.append(self.kept_query_frac[-1])
    CompressedCache.compress = rc; QTrustScorer.set_first_hop = rh

    model, tok = U.load_model(a.model)
    settings = [("SnapKV", dict(compression_method="snap"))] + \
               [(f"tau={t:g}", dict(compression_method="qtrust", tau=t)) for t in TAUS]
    out = []
    for d in DATASETS:
        for j, line in enumerate(open(f"{data}/{d}.jsonl")):
            if j >= a.per:
                break
            p = json.loads(line)["input_prompt"]; ids = tok(p, truncation=False).input_ids
            if len(ids) > 8192 - 64:
                h = (8192 - 64) // 2
                p = tok.decode(ids[:h], skip_special_tokens=True) + tok.decode(ids[-h:], skip_special_tokens=True)
            x = tok(U.build_chat_prompt(p, a.model, tok, d), return_tensors="pt").input_ids.to(model.device)
            rec = dict(dataset=d, idx=j, N=int(x.shape[1]))
            kept_all = {}
            for name, kw in settings:
                cfg = CompressionConfig(observation_window=16, total_budget=128, recent_budget=16, **kw)
                times = []
                for _ in range(2):                                             # the first call is a warm-up
                    KEPT.clear(); ROWS.clear()
                    model.init_cache(cfg); torch.cuda.synchronize(); t0 = time.perf_counter()
                    with torch.inference_mode():
                        model.generate(x, max_new_tokens=1, do_sample=False, pad_token_id=tok.eos_token_id)
                    torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
                kept_all[name] = {l: {(h, int(i)) for h in range(v.shape[0]) for i in v[h].tolist()} for l, v in KEPT.items()}
                rec[name] = dict(time=times[-1], rows=float(np.mean(ROWS)) if ROWS else None)
            ref = kept_all["tau=0"]
            for name, _ in settings:
                k = kept_all[name]
                rec[name]["jaccard"] = float(np.mean([len(k[l] & ref[l]) / max(1, len(k[l] | ref[l])) for l in ref]))
            out.append(rec)
            print(d, j, rec["N"], {n: rec[n]["rows"] for n, _ in settings}, flush=True)
    json.dump(out, open(out_path, "w"))
    print("mean rows computed: " + ", ".join(f"{n} {np.mean([r[n]['rows'] for r in out]):.3f}"
                                            for n, _ in settings if out[0][n]["rows"] is not None))


if __name__ == "__main__":
    main()
