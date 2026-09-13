"""Generate full-cache and 13-action outputs for an arbitrary prompt jsonl.

Input jsonl requires sample_id, input_prompt and generation_length (dataset / task_type are
carried through when present). Output adds full_cache_pred and action_outputs (13), i.e. the
recipe schema consumed by iclr/reward.py.

Sharding is by sample id, so shards can run on different GPUs or hosts and be concatenated.

  python iclr/gen_actions.py --model llama3-8b \
      --src datasets/training/raw/docqa_div_llama3-8b.jsonl \
      --out docqa_gen_s0.jsonl --shard 0/6 --gpu 1
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GRID13 = [(0.0, 1.0)] + [(a, b) for a in [0.01, 0.1, 10.0] for b in [1.0, 16.0, 32.0, 128.0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--budget", type=int, default=128)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.chdir(ROOT)
    import utils as U
    from utils_real_drop import CompressionConfig

    rows = [json.loads(l) for l in open(args.src)]
    si, sn = map(int, args.shard.split("/"))
    rows = rows[si::sn]
    model, tok = U.load_model(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["sample_id"] for l in open(args.out)}

    def gen(ids, am, cfg, n):
        model.init_cache(cfg)
        with torch.inference_mode():
            o = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=n,
                               num_beams=1, do_sample=False, pad_token_id=tok.eos_token_id,
                               tokenizer=tok, stop_strings=U.chat_stop_strings(args.model),
                               num_logits_to_keep=1)[0]
        model.init_cache(None)
        return tok.decode(o[ids.shape[-1]:], skip_special_tokens=True)

    for r in rows:
        if r["sample_id"] in done:
            continue
        enc = tok(r["input_prompt"], truncation=False, return_tensors="pt")
        ids = enc.input_ids.to(model.device)
        am = enc.attention_mask.to(torch.bfloat16).to(model.device)
        n = int(r["generation_length"])
        full = gen(ids, am, None, n)
        outs = []
        for (a, b) in GRID13:
            cfg = CompressionConfig()
            cfg["compression_method"] = "waits"; cfg["total_budget"] = args.budget
            cfg["recent_budget"] = 16; cfg["observation_window"] = int(b)
            cfg["a"] = float(a); cfg["b"] = int(b)
            outs.append(gen(ids, am, cfg, n))
        rec = dict(r); rec["full_cache_pred"] = full; rec["action_outputs"] = outs
        with open(args.out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[gen] {r['sample_id']} done", flush=True)
    print("[gen] SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
