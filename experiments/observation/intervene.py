"""Section 2.2 / 2.4 interventions: do informative queries help, and which query weighting is best?

Per prompt: (1) full-cache generation with the task's own length; one forward over prompt + output measures
iota(q) for every layer and KV head (same definition as capture_iota.py); (2) one compressed generation per arm
(budget 128, 16 recent tokens). Three designs share this protocol:

  --design quartile    (Figure 1b) method-free: every arm is an H2O-style uniform attention sum over sinks + the
                       last 16 queries + a selected 25% of the document queries.
                       arms q1..q4 = iota quartiles (q1 = most informative), random, all (= H2O)
  --design percentile  (Section 2.4) the selected 10% of document queries inside the Q-Trust sum weighted by s
                       (request queries weighted by s, tau = 0): top10 / q10_25 / q25_50 / q50_75 / bottom10
                       iota bands, snap10 (top 10% by s), random10, dense (no restriction)
  --design weights     (Section 2.4) the same Q-Trust sum with different document-query weights:
                       s (w ∝ s), iota (w ∝ max(iota, 0), an oracle), plus SnapKV (W=16) and H2O
All arms of a design use the same number of document queries k and give them the same total mass, so they
differ only in WHICH queries (or weights) enter the sum. Score with analyze_intervene.py.
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F_
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ATTN_IMPL, chdir_root, encode_prompt, load_rows

NSINK, WIN = 4, 16
DESIGNS = {
    "quartile": dict(frac=0.25, bands={"q1": (0.0, 0.25), "q2": (0.25, 0.50), "q3": (0.50, 0.75), "q4": (0.75, 1.00)},
                     extra=["random", "all"], uniform=True),
    "percentile": dict(frac=0.10, bands={"top10": (0.0, 0.10), "q10_25": (0.10, 0.25), "q25_50": (0.25, 0.50),
                                         "q50_75": (0.50, 0.75), "bottom10": (0.90, 1.00)},
                       extra=["snap10", "random10", "dense"], uniform=False),
    "weights": dict(frac=0.25, bands={}, extra=["s", "iota", "snapkv", "h2o"], uniform=False),
}
ST = {}


def cos(x, y):
    if y.dim() == 1:
        num, yn = x @ y, y.norm()
    else:
        num, yn = (x * y).sum(-1), y.norm(dim=-1)
    return num / (x.norm(dim=-1) * yn).clamp_min(1e-30)


def iota_hook(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Phase 1: iota(q) for this layer, every KV head."""
    from transformers.integrations.sdpa_attention import repeat_kv
    num_kv, H, S, hd = key.shape[1], query.shape[1], query.shape[2], query.shape[-1]
    g = H // num_kv
    scaling = scaling if scaling is not None else hd ** -0.5
    out = F_.scaled_dot_product_attention(query, repeat_kv(key, g), repeat_kv(value, g), is_causal=S > 1, scale=scaling)
    if S <= 1:
        return out.transpose(1, 2).contiguous(), None
    N, dev = ST["N"], query.device
    C0, C1 = NSINK, N - WIN
    res = torch.zeros(num_kv, N, device=dev)
    snap = torch.zeros(num_kv, N, device=dev)
    kpos = torch.arange(S, device=dev)
    qpos = torch.arange(N, device=dev)
    bins = torch.floor(torch.log2((N - 1 - qpos).clamp_min(1).float())).long()
    for h in range(num_kv):
        qh = torch.arange(h * g, (h + 1) * g, device=dev)
        q = query[0, qh].float(); k = key[0, h].float()

        def rows(bs, be):
            lg = (q[:, bs:be] @ k.transpose(0, 1).unsqueeze(0)) * scaling
            lg.masked_fill_(kpos.view(1, 1, S) > torch.arange(bs, be, device=dev).view(1, -1, 1), float("-inf"))
            return F_.softmax(lg, -1).sum(0)
        A = torch.zeros(N, C1 - C0, device=dev); U = torch.zeros(C1 - C0, device=dev)
        for bs in range(0, S, 256):
            be = min(bs + 256, S); r = rows(bs, be)[:, :N]
            if bs < N:
                A[bs:min(be, N)] = r[:min(be, N) - bs, C0:C1]
            if be > N:
                U += r[max(bs, N) - bs:, C0:C1].sum(0)
        snap[h] = torch.cat([torch.zeros(C0, device=dev), A[C1:].sum(0), torch.zeros(N - C1, device=dev)])
        A = A / A.sum(-1, keepdim=True).clamp_min(1e-30)
        u = U / U.sum().clamp_min(1e-30)
        valid = (qpos >= C0) & (qpos < C1)
        band = torch.zeros(int(bins.max()) + 1, C1 - C0, device=dev)
        for b in range(band.shape[0]):
            m = valid & (bins == b)
            if m.any():
                band[b] = A[m].mean(0)
        r = band[bins.clamp_min(0)]
        r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        Ap = A - (A * r).sum(-1, keepdim=True) * r
        up = u - (u * r).sum(-1, keepdim=True) * r
        res[h] = cos(Ap, up)
        res[h][~valid] = -1e9
    ST["iota"][module.layer_idx] = res.cpu()
    ST["snap"][module.layer_idx] = snap.cpu()
    return out.transpose(1, 2).contiguous(), None


def masks_for(arm, bands, iota, snap, N, k, rng):
    """[num_kv, N] bool mask of the document queries this arm lets contribute."""
    num_kv = iota.shape[0]
    m = torch.zeros(num_kv, N, dtype=torch.bool)
    for h in range(num_kv):
        v = iota[h]
        order = torch.argsort(v, descending=True)
        order = order[v[order] > -1e8]
        n = order.numel()
        if n == 0:
            continue
        if arm in bands:
            lo, hi = bands[arm]
            pool = order[int(lo * n):max(int(hi * n), int(lo * n) + 1)]
            sel = pool[torch.linspace(0, pool.numel() - 1, min(k, pool.numel())).long()]   # spread over the band
        elif arm == "all":
            sel = order
        elif arm == "snap10":
            s = snap[h].clone(); s[v <= -1e8] = -1e9
            sel = torch.argsort(s, descending=True)[:k]
        elif arm in ("random", "random10"):
            sel = order[torch.randperm(n, generator=rng)[:k]]
        else:
            raise ValueError(arm)
        m[h, sel] = True
    return m


def arm_config(design, arm, N, iota):
    """(CompressionConfig, masks?, doc_w?) of one arm."""
    from kvcompress.compress import CompressionConfig
    base = dict(total_budget=128, recent_budget=16)
    if arm == "snapkv":
        return CompressionConfig(compression_method="snap", observation_window=16, **base), False, None
    if arm == "h2o":
        return CompressionConfig(compression_method="snap", observation_window=32768, **base), False, None
    cfg = CompressionConfig(compression_method="qtrust", observation_window=16, **base)
    if design == "weights":
        doc_w = {L: iota[L][:, NSINK:N - WIN].clamp_min(0) for L in iota} if arm == "iota" else None
        return cfg, False, doc_w
    return cfg, arm != "dense", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=list(DESIGNS), required=True)
    ap.add_argument("--model", default="llama3-8b")
    ap.add_argument("--data", default="datasets/longbench", help="LongBench jsonl directory")
    ap.add_argument("--per", type=int, default=20, help="prompts per task (20 -> the paper's 315 prompts)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--reverse", action="store_true", help="walk the shard from the end (a second worker meets the first in the middle)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); out_dir = os.path.abspath(a.out); data = os.path.abspath(a.data)
    D = DESIGNS[a.design]
    chdir_root()
    import utils as U_
    from kvcompress.compress import make_cache
    from transformers import AttentionInterface
    AttentionInterface.register("iota_capture", iota_hook)
    torch.set_grad_enabled(False)
    maxgen = json.load(open("config/dataset2maxlen.json"))
    rows = load_rows(data, a.per, a.shard, a.reverse)
    model, tok = U_.load_model(a.model)
    os.makedirs(out_dir, exist_ok=True)
    arms = list(D["bands"]) + D["extra"]

    def gen(ds, x, cfg, masks=None, doc_w=None):
        cache = make_cache(model, cfg) if cfg is not None else None
        if masks is not None:
            for L, sc in enumerate(cache.scorers):
                sc.ext_sel = masks[L]
                sc.ext_sel_uniform = D["uniform"]
        if doc_w is not None:
            for L, sc in enumerate(cache.scorers):
                sc.ext_doc_w = doc_w[L]
        model.init_cache(cfg)
        kw = dict(input_ids=x, max_new_tokens=int(maxgen[ds]), do_sample=False, num_beams=1,
                  pad_token_id=tok.eos_token_id, tokenizer=tok, stop_strings=U_.chat_stop_strings(a.model))
        if ds == "samsum":
            kw.update(min_length=x.shape[1] + 1, eos_token_id=[tok.eos_token_id, tok.encode("\n", add_special_tokens=False)[-1]])
        if cache is not None:
            kw["past_key_values"] = cache
        with torch.inference_mode():
            o = model.generate(**kw)[0]
        return tok.decode(o[x.shape[1]:], skip_special_tokens=True)

    for d, j, r in tqdm(rows, desc=f"{a.design} shard {a.shard}"):
        fn = f"{out_dir}/{d}_{j}.json"
        if os.path.exists(fn):
            continue
        x = encode_prompt(tok, a.model, d, r, maxgen, model.device)
        N = x.shape[1]
        if N < 1024:
            continue
        fc = gen(d, x, None)
        y = tok(fc, add_special_tokens=False, return_tensors="pt").input_ids
        if y.numel() < 2:
            continue
        ST.update(N=N, iota={}, snap={})
        model.config._attn_implementation = "iota_capture"
        try:
            with torch.inference_mode():
                model(input_ids=torch.cat([x, y.to(x.device)], 1), use_cache=False, logits_to_keep=1)
        finally:
            model.config._attn_implementation = ATTN_IMPL
        n_doc = N - NSINK - WIN
        k = max(1, int(n_doc * D["frac"]))
        rec = dict(dataset=d, idx=j, N=N, k=k, answers=r.get("answers", []), all_classes=r.get("all_classes"),
                   length=r.get("length"), fc=fc, preds={})
        rng = torch.Generator().manual_seed(1234 + j)
        for arm in arms:
            cfg, use_masks, doc_w = arm_config(a.design, arm, N, ST["iota"])
            masks = ({L: masks_for(arm, D["bands"], ST["iota"][L], ST["snap"][L], N, k, rng) for L in ST["iota"]}
                     if use_masks else None)
            rec["preds"][arm] = gen(d, x, cfg, masks, doc_w)
        iot = torch.stack([ST["iota"][L] for L in sorted(ST["iota"])])
        v = iot[iot > -1e8]
        rec["iota_stats"] = dict(median=float(v.median()), top10=float(v.topk(max(1, v.numel() // 10)).values.mean()),
                                 top1=float(v.topk(max(1, v.numel() // 100)).values.mean()),
                                 bottom10=float((-v).topk(max(1, v.numel() // 10)).values.mean() * -1))
        json.dump(rec, open(fn, "w"))
        torch.cuda.empty_cache()
    print(f"[{a.design}] SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
