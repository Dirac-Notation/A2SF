"""Improved TriAttention calibration (v2): REAL text (recipe, non-LongBench) + freq_scale_sq.

Over the lost original script, this restores the per-(layer, kv-head, freq) weight freq_scale_sq
in a principled way: the VARIANCE across keys of each frequency's pre-RoPE magnitude
|K[f]| = sqrt(K[:F][f]^2 + K[F:][f]^2) (RoPE-invariant). Frequencies whose key-magnitude varies
more are more discriminative -> weighted more. q_mean/q_abs_mean as before but on real text.

Usage: python script/calibrate_triattention_v2.py --model llama3-8b \
    --recipe datasets/training/raw/recipe_v3_8b/train.jsonl --out runs/triattention_stats/llama3-8b_v2_stats.pt
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_prompts", type=int, default=48)
    ap.add_argument("--max_tokens", type=int, default=6000)
    args = ap.parse_args()

    model, tok = utils.load_model(args.model); model.eval()
    cfg = model.config
    L = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_heads)
    group = n_heads // n_kv
    F = head_dim // 2
    theta = float(getattr(cfg, "rope_theta", 10000.0))
    omega = theta ** (-(2.0 * torch.arange(F, dtype=torch.float32)) / head_dim)

    s_qr = torch.zeros(L, n_kv, F, dtype=torch.float64)
    s_qi = torch.zeros(L, n_kv, F, dtype=torch.float64)
    s_qa = torch.zeros(L, n_kv, F, dtype=torch.float64)
    s_kmag = torch.zeros(L, n_kv, F, dtype=torch.float64)      # sum |K[f]|
    s_kmag2 = torch.zeros(L, n_kv, F, dtype=torch.float64)     # sum |K[f]|^2
    cnt = 0; seqlens = []

    cap = {}; hooks = []
    def mk(li, which):
        def h(mod, inp, out): cap[(li, which)] = out.detach()
        return h
    for li in range(L):
        hooks.append(model.model.layers[li].self_attn.q_proj.register_forward_hook(mk(li, "q")))
        hooks.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(mk(li, "k")))

    # real, non-LongBench prompts from the recipe
    prompts = []
    for line in open(args.recipe):
        p = json.loads(line).get("input_prompt") or ""
        if len(p) > 2000:
            prompts.append(p)
        if len(prompts) >= args.n_prompts:
            break

    with torch.inference_mode():
        for pi, p in enumerate(prompts):
            cap.clear()
            ids = tok(p, return_tensors="pt", truncation=True, max_length=args.max_tokens).input_ids.to(model.device)
            S = ids.shape[1]; seqlens.append(S)
            model.init_cache(None); model(input_ids=ids)
            for li in range(L):
                q = cap[(li, "q")].float().view(1, S, n_heads, head_dim).permute(0, 2, 1, 3)
                q = q.view(1, n_kv, group, S, head_dim).mean(dim=2)            # GQA avg -> [1,nkv,S,hd]
                qr, qi = q[..., :F], q[..., F:]
                s_qr[li] += qr.sum(dim=(0, 2)).double().cpu()
                s_qi[li] += qi.sum(dim=(0, 2)).double().cpu()
                s_qa[li] += torch.sqrt(qr * qr + qi * qi).sum(dim=(0, 2)).double().cpu()
                k = cap[(li, "k")].float().view(1, S, n_kv, head_dim).permute(0, 2, 1, 3)  # [1,nkv,S,hd]
                kmag = torch.sqrt(k[..., :F] ** 2 + k[..., F:] ** 2)           # RoPE-invariant |K[f]|
                s_kmag[li] += kmag.sum(dim=(0, 2)).double().cpu()
                s_kmag2[li] += (kmag * kmag).sum(dim=(0, 2)).double().cpu()
            cnt += S
            if (pi + 1) % 8 == 0:
                print(f"[calib-v2] {pi+1}/{len(prompts)} tokens={cnt}", flush=True)
    for h in hooks:
        h.remove()

    kmean = s_kmag / cnt
    kvar = (s_kmag2 / cnt) - kmean * kmean                    # variance of |K[f]| across keys
    stats = {
        "q_mean_real": (s_qr / cnt).float(),
        "q_mean_imag": (s_qi / cnt).float(),
        "q_abs_mean": (s_qa / cnt).float(),
        "freq_scale_sq": kvar.clamp_min(0).float(),           # per-freq discriminativeness
        "omega": omega,
        "rep_position": float(sum(seqlens) / len(seqlens)),
        "model": args.model, "num_samples": len(prompts),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(stats, args.out)
    print(f"[calib-v2] saved {args.out} rep_pos={stats['rep_position']:.0f} "
          f"freq_scale_sq mean={stats['freq_scale_sq'].mean():.4f}")


if __name__ == "__main__":
    main()
