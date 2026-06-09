"""Verify the transformers-v5 KV-compression plugin == the 4.46.2 kv_llama path.

Run the same fixed prompt + config through each implementation and compare:
  - no-compression last-token logits (base-model parity across versions),
  - snap-compression greedy generation (end-to-end compression equivalence).

Usage:
  # under the 4.46.2 env (A2SF):
  python script/verify_v5_equiv.py --mode dump446 --out /tmp/eq_446.pt
  # under the v5 env (A2SF_v5):
  python script/verify_v5_equiv.py --mode dumpv5  --out /tmp/eq_v5.pt
  # either env:
  python script/verify_v5_equiv.py --mode compare --a /tmp/eq_446.pt --b /tmp/eq_v5.pt
"""
import argparse, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT = (("The quick brown fox jumps over the lazy dog. " * 20) +
          "Continue the pattern and list ten more sentences about animals:")
CONFIGS = {
    "snap16":     dict(compression_method="snap", observation_window=16, total_budget=64, recent_budget=16),
    "sigmoid_a01": dict(compression_method="sigmoid", a=0.1, b=16, total_budget=64, recent_budget=16),
    "sigmoid_a10": dict(compression_method="sigmoid", a=10.0, b=1, total_budget=64, recent_budget=16),
}
N_GEN = 40


def dump446(out):
    from transformers import AutoTokenizer
    from utils_real_drop import KVLlamaForCausalLM
    from utils import CompressionConfig
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = KVLlamaForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, device_map={"": "cuda"}).eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    model.init_cache(None)
    with torch.no_grad():
        logits = model(ids).logits[:, -1].float().cpu()
    rec = {"logits": logits, "ids": ids.cpu(), "gen": {}}
    for name, c in CONFIGS.items():
        cfg = CompressionConfig(); cfg.update(c)
        model.init_cache(cfg)
        gen = model.generate(ids, max_new_tokens=N_GEN, do_sample=False, use_cache=True)
        rec["gen"][name] = gen[0, ids.shape[1]:].cpu()
    torch.save(rec, out)
    print(f"saved {out}: prompt_len={ids.shape[1]}")


def dumpv5(out):
    from utils_real_drop.v5_compress import (load_compressed_model, init_cache,
                                             make_cache, CompressionConfig)
    model, tok = load_compressed_model(MODEL, dtype=torch.float32, device_map={"": "cuda"})
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    init_cache(model, None)
    with torch.no_grad():
        logits = model(ids, past_key_values=make_cache(model, None)).logits[:, -1].float().cpu()
    rec = {"logits": logits, "ids": ids.cpu(), "gen": {}}
    for name, c in CONFIGS.items():
        cfg = CompressionConfig(**c)
        init_cache(model, cfg)
        gen = model.generate(ids, max_new_tokens=N_GEN, do_sample=False, use_cache=True,
                             past_key_values=make_cache(model, cfg))
        rec["gen"][name] = gen[0, ids.shape[1]:].cpu()
    torch.save(rec, out)
    print(f"saved {out}: prompt_len={ids.shape[1]}")


def compare(a, b):
    A = torch.load(a); B = torch.load(b)
    print(f"prompt ids identical: {torch.equal(A['ids'], B['ids'])}")
    md = (A["logits"] - B["logits"]).abs().max().item()
    am = bool((A["logits"].argmax(-1) == B["logits"].argmax(-1)).item())
    print(f"[no-comp logits] maxdiff={md:.3e}  argmax_match={am}")
    for name in A["gen"]:
        ga, gb = A["gen"][name], B["gen"][name]
        n = min(len(ga), len(gb))
        match = int((ga[:n] == gb[:n]).sum())
        first_div = next((i for i in range(n) if ga[i] != gb[i]), None)
        print(f"[{name}] {match}/{n} identical"
              + ("" if first_div is None else f"  (first diff @ tok {first_div})"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["dump446", "dumpv5", "compare"])
    p.add_argument("--out"); p.add_argument("--a"); p.add_argument("--b")
    args = p.parse_args()
    if args.mode == "dump446":
        dump446(args.out)
    elif args.mode == "dumpv5":
        dumpv5(args.out)
    else:
        compare(args.a, args.b)
