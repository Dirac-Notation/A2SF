"""Break down router preparation time against TTFT and prefill latency.

Preparation = probe prompt construction (CPU string work) + probe tokenization + probe
forward (~300 tokens) + softmax and feature assembly + router forward + compression config
construction. The main path is the real prefill followed by the first token. Compression acts
during prefill only, so decoding is out of scope here.

Every GPU segment is wrapped in torch.cuda.synchronize(); the reported number is the median
over `reps` repetitions after warmup.

  python iclr/overhead_bench.py --model llama3-8b --gpu 3
"""
import argparse
import json
import os
import statistics as st
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from iclr.probe_v3 import TMPL_V1, surface          # noqa: E402

BUCK = ["S", "M", "L"]


class Timer:
    def __init__(self): self.t = {}
    def __call__(self, name, fn, sync=True, reps=1):
        if sync: torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps): out = fn()
        if sync: torch.cuda.synchronize()
        self.t[name] = (time.perf_counter() - t0) * 1000 / reps
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--lengths", default="2000,4000,8000,16000,32000")
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.chdir(ROOT)
    import utils as U
    from utils_real_drop import CompressionConfig

    model, tok = U.load_model(args.model)
    dev = model.device
    # the deployed MLP shape: input [bucket 3 | option probs 5] -> 64 -> 64 -> k (=9)
    mlp = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 9))
    mlp.eval()
    opt_ids = [tok.encode(f" {c}", add_special_tokens=False)[-1] for c in "ABCDE"]

    rows = []
    for L in [int(x) for x in args.lengths.split(",")]:
        ids_full = torch.randint(1000, 20000, (1, L), device=dev)
        prompt_txt = "word " * (L * 3)          # stand-in prompt for the string-side work
        res = {}

        def probe_all():
            q = TMPL_V1.format(head=prompt_txt[:400], tail=prompt_txt[-400:])
            e = tok(q, return_tensors="pt").input_ids.to(dev)
            lg = model(e).logits[0, -1]
            v = torch.tensor([float(lg[i]) for i in opt_ids])
            p = torch.softmax(v, 0)
            x = torch.tensor([[0., 1., 0.] + p.tolist()])
            return int(mlp(x).argmax())

        # ---- preparation breakdown ----
        T = Timer()
        T("1_prompt_build", lambda: TMPL_V1.format(head=prompt_txt[:400], tail=prompt_txt[-400:]),
          sync=False, reps=50)
        q = TMPL_V1.format(head=prompt_txt[:400], tail=prompt_txt[-400:])
        T("2_probe_tokenize", lambda: tok(q, return_tensors="pt").input_ids, sync=False, reps=20)
        pe = tok(q, return_tensors="pt").input_ids.to(dev)
        model.init_cache(None)
        with torch.no_grad():
            model(pe)                                            # warmup
            T("3_probe_forward", lambda: model(pe), reps=args.reps)
            lg = model(pe).logits[0, -1]
        T("4_softmax_feats",
          lambda: torch.softmax(torch.tensor([float(lg[i]) for i in opt_ids]), 0),
          sync=False, reps=50)
        xin = torch.rand(1, 8)
        with torch.no_grad():
            T("5_mlp_forward", lambda: mlp(xin), sync=False, reps=200)

        def build_cfg():
            c = CompressionConfig()
            c["compression_method"] = "waits"; c["total_budget"] = args.budget
            c["recent_budget"] = 16; c["observation_window"] = 16; c["a"] = 10.0; c["b"] = 16
            return c
        T("6_cfg_build", build_cfg, sync=False, reps=200)
        res["probe_tokens"] = int(pe.shape[1])
        prep = sum(T.t[k] for k in T.t)
        res.update(T.t); res["prep_total"] = prep

        # ---- main path ----
        cfg = build_cfg()
        T2 = Timer()
        with torch.no_grad():
            model.init_cache(None)
            from utils_real_drop.compress import make_cache
            def prefill_plain():
                model.init_cache(None)
                return model(ids_full, past_key_values=make_cache(model, None),
                             use_cache=True, logits_to_keep=1)
            prefill_plain()
            T2("prefill_nocomp", prefill_plain, reps=max(2, args.reps // 2))

            def prefill_comp():
                model.init_cache(cfg)
                return model(ids_full, past_key_values=make_cache(model, cfg),
                             use_cache=True, logits_to_keep=1)
            prefill_comp()
            T2("prefill_comp", prefill_comp, reps=max(2, args.reps // 2))

            def ttft():
                model.init_cache(cfg)
                o = model(ids_full, past_key_values=make_cache(model, cfg),
                          use_cache=True, logits_to_keep=1)
                return o.logits[:, -1].argmax(-1)
            T2("ttft_comp", ttft, reps=max(2, args.reps // 2))

        model.init_cache(None)
        res.update(T2.t); res["length"] = L
        rows.append(res)
        pn, pc, tt = T2.t['prefill_nocomp'], T2.t['prefill_comp'], T2.t['ttft_comp']
        print(f"[oh] L={L:6d} prep={prep:6.2f}ms | prefill uncompressed {pn:8.1f} / compressed {pc:8.1f}ms "
              f"(compression cost {100*(pc-pn)/pn:+5.1f}%) | TTFT {tt:8.1f}ms "
              f"| prep/prefill={100*prep/pc:5.2f}%  prep/TTFT={100*prep/tt:5.2f}%", flush=True)
        os.makedirs("result_txt/analysis/overhead", exist_ok=True)
        json.dump(rows, open(f"result_txt/analysis/overhead/{args.model}.json","w"), indent=1)
    os.makedirs("result_txt/analysis/overhead", exist_ok=True)
    json.dump(rows, open(f"result_txt/analysis/overhead/{args.model}.json", "w"), indent=1)
    print(f"saved -> result_txt/analysis/overhead/{args.model}.json")


if __name__ == "__main__":
    main()
