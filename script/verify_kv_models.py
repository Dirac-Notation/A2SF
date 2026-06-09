"""Verify a KV-compression model class against stock HF for a given checkpoint.

Two checks:
  1. Correctness: KV model with NO compression must match stock HF logits.
  2. Compression: with a SnapKV config, generation runs and the cache is capped
     at the budget after prefill.

Usage: python script/verify_kv_models.py <hf_model_path_or_id>
"""
import sys, os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from utils_real_drop import get_kv_class


def main(model_path):
    dtype = torch.float32  # fp32 for an exact-ish correctness comparison
    cfg = AutoConfig.from_pretrained(model_path)
    print(f"\n=== {model_path}  (model_type={cfg.model_type}) ===")
    tok = AutoTokenizer.from_pretrained(model_path)
    kv_class = get_kv_class(cfg.model_type)

    prompt = "The capital of France is Paris. The capital of Japan is"
    ids = tok(prompt, return_tensors="pt").input_ids

    # ---- 1) correctness: KV (no compression) vs stock HF ----
    stock = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).eval()
    with torch.no_grad():
        ref = stock(ids).logits[:, -1, :]
    del stock

    kv = kv_class.from_pretrained(model_path, torch_dtype=dtype).eval()
    kv.init_cache(None)  # no compression
    with torch.no_grad():
        got = kv(ids, use_cache=False).logits[:, -1, :]
    maxdiff = (ref - got).abs().max().item()
    same_argmax = int(ref.argmax(-1) == got.argmax(-1))
    print(f"[correctness] last-token logits maxdiff={maxdiff:.3e}  argmax_match={bool(same_argmax)}  "
          f"next_tok={tok.decode(got.argmax(-1))!r}")

    # ---- 2) compression runs + cache capped at budget ----
    BUDGET = 32
    comp = type("C", (dict,), {"__getattr__": dict.get, "__setattr__": dict.__setitem__})()
    comp["compression_method"] = "snap"
    comp["observation_window"] = 16
    comp["total_budget"] = BUDGET
    comp["recent_budget"] = 16
    kv.init_cache(comp)

    long_prompt = (prompt + " ") * 12   # ~150 tokens >> budget
    lids = tok(long_prompt, return_tensors="pt").input_ids
    out = kv.generate(lids, max_new_tokens=20, do_sample=False, use_cache=True)
    gen = tok.decode(out[0, lids.shape[1]:], skip_special_tokens=True)
    print(f"[compression] prompt_len={lids.shape[1]} budget={BUDGET}  generated 20 toks OK")
    print(f"              gen={gen!r}")

    ok = bool(same_argmax) and maxdiff < 1e-2
    print(f"[RESULT] {model_path}: {'PASS' if ok else 'CHECK'} "
          f"(correctness {'ok' if ok else 'maxdiff='+format(maxdiff,'.2e')})")
    return ok


if __name__ == "__main__":
    main(sys.argv[1])
