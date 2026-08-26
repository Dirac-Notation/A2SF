"""Pipeline smoke / determinism check for the v5 KV-compression path.

Drives the integrated pipeline (utils.load_model -> model.init_cache ->
model.generate) exactly like longbench.py's worker (num_logits_to_keep=1,
stop_strings, tokenizer) for full / snap / sigmoid configs, and dumps the greedy
generations. `compare` two dumps to confirm determinism (e.g. across machines or
GPUs). The 4.46.2<->v5 equivalence this established (all configs bit-identical) is
recorded in logs/history #41-#42.

Usage:
  python script/verify_pipeline.py --mode dump --out /tmp/pipe_a.pt
  python script/verify_pipeline.py --mode dump --out /tmp/pipe_b.pt
  python script/verify_pipeline.py --mode compare --a /tmp/pipe_a.pt --b /tmp/pipe_b.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_model, set_seed, CompressionConfig

MODEL_NAME = "llama3-1b"
PROMPT = (("The quick brown fox jumps over the lazy dog. " * 20) +
          "Continue the pattern and list ten more sentences about animals:")
N_GEN = 40
CONFIGS = {
    "full":        dict(compression_method="full", observation_window=16, total_budget=128, a=10, b=16),
    "snap16":      dict(compression_method="snap", observation_window=16, total_budget=64, a=10, b=16, recent_budget=16),
    "sigmoid_a01": dict(compression_method="waits", observation_window=16, total_budget=64, a=0.1, b=16, recent_budget=16),
    "sigmoid_a10": dict(compression_method="waits", observation_window=16, total_budget=64, a=10.0, b=1, recent_budget=16),
}


def _make_config(d):
    cfg = CompressionConfig()
    cfg["compression_method"] = d["compression_method"]
    cfg["observation_window"] = d.get("observation_window", 16)
    cfg["total_budget"] = d["total_budget"]
    cfg["a"] = d["a"]
    cfg["b"] = d["b"]
    cfg["recent_budget"] = d.get("recent_budget", 16)
    cfg["chunk_size"] = 0
    cfg["chunk_group_size"] = 1
    cfg["pyramid_kv"] = False
    cfg["pyramid_ratio"] = 4.0
    return cfg


def dump(out):
    set_seed(42)
    model, tok = load_model(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(PROMPT, truncation=False, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    attn = enc.attention_mask.to(model.device)
    ctx = int(input_ids.shape[-1])

    rec = {"ids": input_ids.cpu(), "gen": {}}
    for name, d in CONFIGS.items():
        model.init_cache(_make_config(d))
        with torch.inference_mode():
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=N_GEN,
                num_beams=1,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                tokenizer=tok,
                stop_strings="[/INST]",
                num_logits_to_keep=1,
            )[0]
        rec["gen"][name] = out_ids[ctx:].cpu()
    torch.save(rec, out)
    print(f"saved {out}: prompt_len={ctx}, transformers via load_model OK")
    for name in CONFIGS:
        print(f"  [{name}] gen_len={len(rec['gen'][name])}")


def compare(a, b):
    A = torch.load(a); B = torch.load(b)
    print(f"prompt ids identical: {torch.equal(A['ids'], B['ids'])}")
    ok = True
    for name in A["gen"]:
        ga, gb = A["gen"][name], B["gen"][name]
        n = min(len(ga), len(gb))
        match = int((ga[:n] == gb[:n]).sum())
        first = next((i for i in range(n) if ga[i] != gb[i]), None)
        ok = ok and (match == n and len(ga) == len(gb))
        print(f"[{name}] {match}/{n} identical, lens={len(ga)}/{len(gb)}"
              + ("" if first is None else f"  (first diff @ tok {first})"))
    print("ALL IDENTICAL" if ok else "MISMATCH FOUND")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["dump", "compare"])
    p.add_argument("--out"); p.add_argument("--a"); p.add_argument("--b")
    args = p.parse_args()
    if args.mode == "dump":
        dump(args.out)
    else:
        compare(args.a, args.b)
