"""Recompute TriAttention calibration stats (q_mean / q_abs_mean / omega / rep_position).

The original calibration script was lost; this is a faithful reconstruction that is
directive-compliant: it calibrates on SELF-CONTAINED SYNTHETIC long prompts (NOT
LongBench), gathering pre-RoPE query statistics per (layer, kv-head, freq).

Stats produced (consumed by utils_real_drop/scorers/triattention.py):
    q_mean_real  [L, H_kv, F]   mean of pre-RoPE query, real part  (half-RoPE layout)
    q_mean_imag  [L, H_kv, F]
    q_abs_mean   [L, H_kv, F]   mean |q_complex|
    omega        [F]            RoPE inverse frequencies (theta^(-2i/d))
    rep_position scalar         mean calibration sequence length
    freq_scale_sq: omitted (optional per-freq weight; scorer treats absent as uniform)

Usage: python script/calibrate_triattention.py --model llama3-8b --out runs/triattention_stats/llama3-8b_stats.pt
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


WORDS = ("system network module buffer kernel matrix gradient policy vector tensor "
         "protocol cluster session payload anchor lattice horizon cascade meridian "
         "quantum harbor velocity texture compiler register manifold spectrum cipher").split()


def synth_prompts(n, approx_tokens, seedbase=0):
    """Deterministic synthetic long documents + a question. No external/LongBench data."""
    prompts = []
    for i in range(n):
        # vary content per prompt with a simple LCG over the word list (no Math.random needed)
        s = (seedbase + i * 2654435761) & 0xFFFFFFFF
        toks = []
        for _ in range(approx_tokens):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF
            toks.append(WORDS[s % len(WORDS)])
        body = " ".join(toks)
        prompts.append(f"Read the following log and answer.\n{body}\nQuestion: summarize the key module. Answer:")
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_prompts", type=int, default=48)
    ap.add_argument("--approx_tokens", type=int, default=4500)
    args = ap.parse_args()

    model, tok = utils.load_model(args.model)
    model.eval()
    cfg = model.config
    L = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_heads)
    group = n_heads // n_kv
    F = head_dim // 2
    theta = float(getattr(cfg, "rope_theta", 10000.0))
    omega = theta ** (-(2.0 * torch.arange(F, dtype=torch.float32)) / head_dim)  # [F]

    # accumulators (fp64 for stability)
    sum_real = torch.zeros(L, n_kv, F, dtype=torch.float64)
    sum_imag = torch.zeros(L, n_kv, F, dtype=torch.float64)
    sum_abs  = torch.zeros(L, n_kv, F, dtype=torch.float64)
    count = 0
    seqlens = []

    captured = {}
    hooks = []
    def mk_hook(layer_idx):
        def hook(mod, inp, out):
            captured[layer_idx] = out.detach()  # [B, S, n_heads*head_dim]
        return hook
    layers = model.model.layers
    for li in range(L):
        hooks.append(layers[li].self_attn.q_proj.register_forward_hook(mk_hook(li)))

    prompts = synth_prompts(args.n_prompts, args.approx_tokens)
    with torch.inference_mode():
        for pi, p in enumerate(prompts):
            captured.clear()
            ids = tok(p, return_tensors="pt").input_ids.to(model.device)
            S = ids.shape[1]
            seqlens.append(S)
            model.init_cache(None)
            model(input_ids=ids)  # prefill only; hooks capture pre-RoPE q_proj per layer
            for li in range(L):
                q = captured[li].float()                     # [B, S, n_heads*hd]
                B = q.shape[0]
                q = q.view(B, S, n_heads, head_dim).permute(0, 2, 1, 3)   # [B, nh, S, hd]
                q = q.view(B, n_kv, group, S, head_dim).mean(dim=2)        # GQA: avg within group -> [B, n_kv, S, hd]
                qr = q[..., :F]                              # [B, n_kv, S, F]
                qi = q[..., F:]
                qabs = torch.sqrt(qr * qr + qi * qi)
                sum_real[li] += qr.sum(dim=(0, 2)).double().cpu()
                sum_imag[li] += qi.sum(dim=(0, 2)).double().cpu()
                sum_abs[li]  += qabs.sum(dim=(0, 2)).double().cpu()
            count += B * S
            if (pi + 1) % 8 == 0:
                print(f"[calib] {pi+1}/{len(prompts)} prompts, tokens so far={count}", flush=True)

    for h in hooks:
        h.remove()

    stats = {
        "q_mean_real": (sum_real / count).float(),
        "q_mean_imag": (sum_imag / count).float(),
        "q_abs_mean":  (sum_abs / count).float(),
        "omega": omega,
        "rep_position": float(sum(seqlens) / len(seqlens)),
        "model": args.model,
        "num_samples": len(prompts),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(stats, args.out)
    print(f"[calib] saved {args.out}  rep_position={stats['rep_position']:.0f}  "
          f"shapes q_mean_real={tuple(stats['q_mean_real'].shape)}")


if __name__ == "__main__":
    main()
