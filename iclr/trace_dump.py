"""Trace dump v3 — agent training data: per-head (K, V) + future-attention target.

Adds to v2: stores each head's post-RoPE prefill K and V (fp16) so the (k, v, pos)
-> (a, b) agent can be trained offline. Output goes to /data2 (large).

Same measurement as w0_dump.py (per-head future-attention target u + candidate
sigmoid-curve scores), but attention is never materialized as full [L][Hq,N,N]:
a custom attention function ("w0_capture", registered like the production waits
plugin) computes the reductions layer-by-layer in query chunks and discards the
probabilities. This fits 8B @ 4k on a 24GB card.

Extensions over v1: 17-candidate grid ((0,1) + {0.01,0.1,1,10} x {1,16,32,128}),
multi-budget retained fractions {64,128,256}, --doc_offset for multi-GPU sharding.

Usage (one shard):
  python iclr/w0_dump_v2.py --model llama3-8b --doc_offset 0 --n_docs 25 --gpu 0
Shards write doc_<offset+i>.npz into the same dir; doc list is seed-deterministic
so disjoint offsets never collide.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

A_VALUES = [0.01, 0.1, 1.0, 10.0]
B_VALUES = [1.0, 16.0, 32.0, 128.0]
CANDIDATES = [(0.0, 1.0)] + [(a, b) for a in A_VALUES for b in B_VALUES]  # 17
BUDGETS = [64, 128, 256]
SINK_N = 4
Q_CHUNK = 256


class Capture:
    """Per-doc capture state shared with the registered attention function."""

    def __init__(self, n_layers, n_kv, n_prefill, curve_w, device):
        self.phase = "prefill"          # "prefill" | "decode"
        self.n_prefill = n_prefill
        self.curve_w = curve_w          # [C, N] fp32
        # CPU-resident accumulators: chunk contributions are computed on GPU and
        # moved off immediately, so 8B @ 32k capture fits in 24GB.
        self.cand = torch.zeros(len(CANDIDATES), n_layers, n_kv, n_prefill)
        self.u = torch.zeros(n_layers, n_kv, n_prefill)
        self.k_store = [None] * n_layers   # post-RoPE prefill K per layer [Hkv,N,D]
        self.v_store = [None] * n_layers


_CAP = {"obj": None}


def w0_capture_attention(module, query, key, value, attention_mask,
                         scaling=None, dropout=0.0, **kwargs):
    from utils_real_drop.compress import repeat_kv
    cap = _CAP["obj"]
    num_kv = key.shape[1]
    num_heads = query.shape[1]
    group = num_heads // num_kv
    Sq, Sk = query.shape[2], key.shape[2]
    head_dim = query.shape[-1]
    if scaling is None:
        scaling = head_dim ** -0.5
    lidx = module.layer_idx

    key_rep = repeat_kv(key, group)
    value_rep = repeat_kv(value, group)

    outs = []
    for q0 in range(0, Sq, Q_CHUNK):
        q1 = min(q0 + Q_CHUNK, Sq)
        qc = query[:, :, q0:q1]                                   # [1,Hq,qc,D]
        logits = torch.matmul(qc, key_rep.transpose(2, 3)) * scaling
        if attention_mask is not None:
            logits = logits + attention_mask[:, :, q0:q1, :Sk]
        elif Sq > 1:
            # transformers v5 passes attention_mask=None to custom attention fns and
            # leaves causality to the implementation (same contract as compress.py).
            # Without this, prefill attends to FUTURE keys - silently catastrophic.
            kpos = torch.arange(Sk, device=logits.device)
            qpos = torch.arange(q0, q1, device=logits.device) + (Sk - Sq)
            causal = kpos.view(1, Sk) > qpos.view(-1, 1)
            logits.masked_fill_(causal.view(1, 1, q1 - q0, Sk), float("-inf"))
        probs = F.softmax(logits.float(), dim=-1)                 # [1,Hq,qc,Sk]
        outs.append(torch.matmul(probs.to(value_rep.dtype), value_rep))

        if cap is not None:
            N = cap.n_prefill
            p = probs[0]                                          # [Hq,qc,Sk]
            if cap.phase == "combined":
                # single teacher-forced forward of [prompt | continuation]:
                # rows with global pos < n_prefill feed cand, rows >= feed u.
                N = cap.n_prefill
                off = Sk - Sq  # 0 for a single full forward
                g0, g1 = q0 + off, q1 + off
                if g0 < N:  # prompt-row part of this chunk
                    hi = min(g1, N)
                    pk = p[:, :hi - g0, :N].view(num_kv, group, hi - g0, N).sum(1)
                    w = cap.curve_w[:, g0:hi]
                    cap.cand[:, lidx] += torch.einsum("cq,hqk->chk", w, pk).cpu()
                if g1 > N:  # continuation-row part
                    lo = max(g0, N) - g0
                    pk = p[:, lo:, :N].view(num_kv, group, -1, N).sum(1).sum(1)
                    cap.u[lidx] += pk.cpu()
            elif cap.phase == "prefill":
                if q0 == 0 and cap.k_store[lidx] is None:
                    cap.k_store[lidx] = key[0].detach().to(torch.float16).cpu()
                    cap.v_store[lidx] = value[0].detach().to(torch.float16).cpu()
                # candidate scores over key positions [:N]
                pk = p[..., :N].view(num_kv, group, q1 - q0, N).sum(1)  # [KV,qc,N]
                w = cap.curve_w[:, q0:q1]                         # [C,qc]
                cap.cand[:, lidx] += torch.einsum("cq,hqk->chk", w, pk).cpu()
            else:
                pk = p[..., :N].view(num_kv, group, Sq, N).sum(1).sum(1)  # [KV,N]
                cap.u[lidx] += pk.cpu()
        del probs, logits
    attn_output = torch.cat(outs, dim=2)
    return attn_output.transpose(1, 2).contiguous(), None


def build_docs(tokenizer, n_total, seq_len, seed=0):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.RandomState(seed)
    docs, n_lines = [], len(ds)
    while len(docs) < n_total:
        start = int(rng.randint(0, n_lines - 2000))
        buf, i = [], start
        while i < n_lines and sum(len(t) for t in buf) < seq_len * 8:
            t = ds[i]["text"]
            if t.strip():
                buf.append(t)
            i += 1
        ids = tokenizer("".join(buf), return_tensors="pt",
                        truncation=True, max_length=seq_len)
        if ids["input_ids"].shape[1] >= seq_len:
            docs.append(ids["input_ids"][:, :seq_len])
    return docs


def curve_weights(n_q, device):
    q = torch.arange(n_q, dtype=torch.float32, device=device)
    ws = []
    for a, b in CANDIDATES:
        if a == 0.0:
            ws.append(torch.ones(n_q, device=device))
        else:
            ws.append(torch.sigmoid(a * (q - (n_q - b - 0.5))))
    return torch.stack(ws)


def retained_fraction(u, cand, budgets, sink_n):
    """-> frac [2(variant), n_budgets, C, L, H]"""
    C, L, H, N = cand.shape
    out = np.zeros((2, len(budgets), C, L, H), dtype=np.float32)
    for bi, budget in enumerate(budgets):
        for c in range(C):
            for l in range(L):
                for h in range(H):
                    idx = np.argpartition(-cand[c, l, h], budget)[:budget]
                    tot = u[l, h].sum() + 1e-9
                    out[0, bi, c, l, h] = u[l, h][idx].sum() / tot
                    mask = idx >= sink_n
                    tot_ns = u[l, h][sink_n:].sum() + 1e-9
                    out[1, bi, c, l, h] = u[l, h][idx[mask]].sum() / tot_ns
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--doc_offset", type=int, default=0)
    ap.add_argument("--n_docs", type=int, default=25)
    ap.add_argument("--total_docs", type=int, default=100)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--f_tokens", type=int, default=128)
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = "cuda"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    os.chdir(root)
    with open("config/model2path.json") as f:
        model_path = json.load(f)[args.model]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register("w0_capture", w0_capture_attention)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="w0_capture",
    ).to(device).eval()
    cfg = model.config
    L, Hkv = cfg.num_hidden_layers, cfg.num_key_value_heads

    out_dir = args.out_dir or f"/data2/smp9898/iclr_traces/{args.model}"
    os.makedirs(out_dir, exist_ok=True)
    if args.doc_offset == 0:
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"model": args.model, "candidates": CANDIDATES,
                       "budgets": BUDGETS, "sink_n": SINK_N,
                       "seq_len": args.seq_len, "f_tokens": args.f_tokens,
                       "group_reduce": "sum", "total_docs": args.total_docs}, f, indent=2)

    docs = build_docs(tokenizer, args.total_docs, args.seq_len)
    shard = docs[args.doc_offset:args.doc_offset + args.n_docs]
    print(f"[trace] {args.model} L={L} Hkv={Hkv} shard [{args.doc_offset},"
          f"{args.doc_offset + len(shard)})", flush=True)

    w = curve_weights(args.seq_len, device)
    for k, ids in enumerate(shard):
        d = args.doc_offset + k
        path = os.path.join(out_dir, f"doc_{d:04d}.npz")
        if os.path.exists(path):
            print(f"[trace] doc {d} exists, skip", flush=True)
            continue
        cap = Capture(L, Hkv, args.seq_len, w, device)
        _CAP["obj"] = cap
        with torch.no_grad():
            cap.phase = "prefill"
            out = model(ids.to(device), use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1:].argmax(-1)
            cap.phase = "decode"
            for _ in range(args.f_tokens):
                step = model(next_id, past_key_values=past, use_cache=True)
                past = step.past_key_values
                next_id = step.logits[:, -1:].argmax(-1)
        _CAP["obj"] = None
        u_np = cap.u.cpu().numpy()
        cand_np = cap.cand.cpu().numpy()
        frac = retained_fraction(u_np, cand_np, BUDGETS, SINK_N)
        K = torch.stack(cap.k_store).numpy()   # [L, Hkv, N, D] fp16
        V = torch.stack(cap.v_store).numpy()
        np.savez_compressed(path, u=u_np.astype(np.float16),
                            cand=cand_np.astype(np.float16), frac=frac,
                            K=K, V=V)
        del past, cap
        torch.cuda.empty_cache()
        print(f"[trace] doc {d} done", flush=True)
    print("[trace] SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
