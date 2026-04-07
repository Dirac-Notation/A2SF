"""
Sanity check: KVLlamaForCausalLM (no compression) vs HF AutoModelForCausalLM
should produce identical next-token logits / greedy continuations on a small
set of hard-coded prompts.

Run:
    python test.py
"""

import glob
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils_real_drop.kv_llama import KVLlamaForCausalLM


SEED = 42
NUM_PROMPTS = 10
TARGET_TOKENS = 2000
TOKEN_TOLERANCE = 500  # accept prompts in [TARGET-TOL, TARGET+TOL]
MAX_NEW_TOKENS = 128
LONGBENCH_DIR = "datasets/longbench"


def sample_long_prompts(tokenizer, n: int, target: int, tol: int):
    """Randomly sample `n` LongBench prompts whose token length is near `target`."""
    rng = random.Random(SEED)
    files = sorted(glob.glob(os.path.join(LONGBENCH_DIR, "*.jsonl")))
    if not files:
        raise RuntimeError(f"No jsonl files in {LONGBENCH_DIR}")

    pool = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("input_prompt") or obj.get("input") or obj.get("context")
                if isinstance(text, str) and text:
                    pool.append((os.path.basename(fp), text))

    rng.shuffle(pool)
    chosen = []
    lo, hi = target - tol, target + tol
    for src, text in pool:
        n_tok = len(tokenizer(text, add_special_tokens=False).input_ids)
        if lo <= n_tok <= hi:
            chosen.append((src, text, n_tok))
            if len(chosen) == n:
                break
    if len(chosen) < n:
        raise RuntimeError(
            f"Only found {len(chosen)}/{n} prompts within {lo}-{hi} tokens. "
            f"Loosen TOKEN_TOLERANCE or change TARGET_TOKENS."
        )
    return chosen


def load_model_path(name: str = "llama3-1b") -> str:
    with open("config/model2path.json") as f:
        return json.load(f)[name]


@torch.no_grad()
def main():
    model_path = load_model_path("llama3-1b")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading HF AutoModelForCausalLM...")
    # Use sdpa to match KVLlama's attention backend (both end up in
    # F.scaled_dot_product_attention with is_causal=True for prefill).
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()

    print("Loading KVLlamaForCausalLM...")
    kv_model = KVLlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype
    ).to(device).eval()
    kv_model.init_cache(compression_config=None)  # full cache, no compression

    print(
        f"Sampling {NUM_PROMPTS} LongBench prompts near {TARGET_TOKENS} tokens "
        f"(±{TOKEN_TOLERANCE}, seed={SEED})..."
    )
    samples = sample_long_prompts(tokenizer, NUM_PROMPTS, TARGET_TOKENS, TOKEN_TOLERANCE)
    for i, (src, _, n_tok) in enumerate(samples):
        print(f"  [{i+1:2d}] {src} ({n_tok} tokens)")

    n_logit_pass = 0
    n_gen_pass = 0
    max_logit_diff_overall = 0.0

    for i, (src, prompt, n_tok) in enumerate(samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # ---- 1) single forward: compare last-token logits ----
        hf_out = hf_model(**inputs, use_cache=False)
        kv_out = kv_model(**inputs, use_cache=False)

        hf_last = hf_out.logits[0, -1].float()
        kv_last = kv_out.logits[0, -1].float()
        max_diff = (hf_last - kv_last).abs().max().item()
        argmax_match = hf_last.argmax().item() == kv_last.argmax().item()
        max_logit_diff_overall = max(max_logit_diff_overall, max_diff)

        # ---- 2) greedy generation: compare token sequences ----
        gen_kwargs = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        hf_ids = hf_model.generate(**inputs, **gen_kwargs)[0].tolist()
        kv_ids = kv_model.generate(**inputs, **gen_kwargs)[0].tolist()
        gen_match = hf_ids == kv_ids

        n_logit_pass += int(argmax_match)
        n_gen_pass += int(gen_match)

        status_logit = "OK" if argmax_match else "MISMATCH"
        status_gen = "OK" if gen_match else "MISMATCH"
        print(
            f"[{i+1:2d}/{len(samples)}] {src} ({n_tok} tok)  "
            f"logits: {status_logit} (max|Δ|={max_diff:.2e})  greedy: {status_gen}"
        )
        if not gen_match:
            print(f"     HF : {tokenizer.decode(hf_ids[inputs.input_ids.shape[1]:])!r}")
            print(f"     KV : {tokenizer.decode(kv_ids[inputs.input_ids.shape[1]:])!r}")

    print()
    print(f"logit-argmax match : {n_logit_pass}/{len(samples)}")
    print(f"greedy-gen   match : {n_gen_pass}/{len(samples)}")
    print(f"max |logit diff|   : {max_logit_diff_overall:.3e}")


if __name__ == "__main__":
    main()
