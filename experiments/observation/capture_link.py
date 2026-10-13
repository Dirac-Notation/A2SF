"""Section 2.2/2.4 (Table 1): how well does the request score s point at informative queries?\n\nRun once per model; Table 1 reads the three outputs with analyze_link.py.

  rho       cos(A(q,.) - band part, s_profile - band part): does the query of a token the request attends to
            look at the same keys as the request? (s_profile = the window attention over the document keys)
  s in-band s divided by the mean s of its log-distance band, so it cannot use distance at all
  iota_raw / iota_glob   iota without the band removal / with only the document-wide mean removed
  keyrec    the end target: recall of the top-112 keys of u by the key score each weighting produces
            (sum_q w(q) A(q, .)), the u mass it keeps, recall of the far (>2k from the end) top keys, and the
            effective number of queries 1/sum w^2. 'SnapKV' is s itself used as the key score.
  nonlocal  iota recomputed with every key within +-32 tokens of the query masked out (locality control)

Original notes follow.

How well does the window score s point at informative queries? Same prompts and the same iota definition
as capture_iota.py, plus what that capture could not answer:

  ceiling   iota is measured from a finite generation, so no predictor can match it perfectly. The generated
            tokens are split into even / odd steps, iota_a and iota_b are measured from each half, and iota_a is
            scored as a predictor of iota_b. That is the best any query weighting could do against a noisy target.
  steps     iota from the first 8 generated tokens vs the rest (does s point at what the answer starts from?)
  rivals    other query weightings that are available at prefill time, scored exactly like s:
            H2O-transfer (the key's accumulated attention from all queries, moved onto its query), recency 1/d,
            random, s from the last 1 / 64 queries, s squared, s averaged over the KV heads of the layer
  curve     mean iota and the informative share in 20 bins of the s percentile

Every weighting is restricted to the document queries [4, N-16) and normalized there. Informative = top 25%.
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F_
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ATTN_IMPL, chdir_root, encode_prompt, load_rows

NSINK, WIN, EARLY, NBIN, KEEP, RLOC = 4, 16, 8, 20, 112, 32
ST = {}


def cos(x, y):
    if y.dim() == 1:
        num, yn = x @ y, y.norm()
    else:
        num, yn = (x * y).sum(-1), y.norm(dim=-1)
    return num / (x.norm(dim=-1) * yn).clamp_min(1e-30)


def spear(x, y):
    rx = x.argsort().argsort().float(); ry = y.argsort().argsort().float()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx @ ry) / (rx.norm() * ry.norm()).clamp_min(1e-30))


def score(w, iota):
    """w, iota over document queries. Mass on the top 1/10/25% of iota (as a lift over uniform), E_w[iota], rank corr."""
    w = w.clamp_min(0); w = w / w.sum().clamp_min(1e-30)
    order = iota.argsort(descending=True); n = iota.numel()
    out = dict(Ew=float((w * iota).sum()), spearman=spear(w, iota))
    for p in (1, 10, 25):
        out[f"lift{p}"] = float(w[order[:max(1, n * p // 100)]].sum() / (p / 100))
    return out


def iota_of(An, u, r):
    return cos(An - (An * r).sum(-1, keepdim=True) * r, u - (u * r).sum(-1, keepdim=True) * r)


def hook(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    from transformers.integrations.sdpa_attention import repeat_kv
    num_kv, H, S, hd = key.shape[1], query.shape[1], query.shape[2], query.shape[-1]
    g = H // num_kv
    scaling = scaling if scaling is not None else hd ** -0.5
    out = F_.scaled_dot_product_attention(query, repeat_kv(key, g), repeat_kv(value, g), is_causal=S > 1, scale=scaling)
    if S <= 1:
        return out.transpose(1, 2).contiguous(), None
    N, dev = ST["N"], query.device
    C0, C1 = NSINK, N - WIN
    qpos = torch.arange(N, device=dev); kpos = torch.arange(S, device=dev)
    dist = (N - 1 - qpos).float()
    bins = torch.floor(torch.log2(dist.clamp_min(1))).long()
    doc = (qpos >= C0) & (qpos < C1)
    tgen = torch.arange(S - N, device=dev)                                # generation step of each row >= N
    parts = {"all": tgen >= 0, "even": tgen % 2 == 0, "odd": tgen % 2 == 1,
             "early": tgen < EARLY, "late": tgen >= EARLY}
    per_head = []
    for h in range(num_kv):
        qh = torch.arange(h * g, (h + 1) * g, device=dev)
        q = query[0, qh].float(); k = key[0, h].float()
        A = torch.zeros(N, C1 - C0, device=dev)
        U = {nm: torch.zeros(C1 - C0, device=dev) for nm in parts}
        s_full = torch.zeros(N, device=dev)                             # window attention to every prompt key
        for bs in range(0, S, 256):
            be = min(bs + 256, S)
            lg = (q[:, bs:be] @ k.transpose(0, 1).unsqueeze(0)) * scaling
            lg.masked_fill_(kpos.view(1, 1, S) > torch.arange(bs, be, device=dev).view(1, -1, 1), float("-inf"))
            rr = F_.softmax(lg, -1).sum(0)[:, :N]
            if bs < N:
                A[bs:min(be, N)] = rr[:min(be, N) - bs, C0:C1]
                lo = max(bs, C1)
                if lo < min(be, N):
                    s_full += rr[lo - bs:min(be, N) - bs, :N].sum(0)
            if be > N:
                gr = rr[max(bs, N) - bs:, C0:C1]; ts = tgen[max(bs, N) - N: be - N]
                for nm, m in parts.items():
                    U[nm] += gr[m[ts]].sum(0)
        per_head.append((A, U, s_full))
    # s of every KV head, so the layer average is available to each head
    s_all = torch.stack([A[C1:].sum(0) for A, _, _ in per_head])             # [kv, doc keys]
    s_layer = (s_all / s_all.sum(-1, keepdim=True).clamp_min(1e-30)).mean(0)
    gen_rand = torch.Generator(device="cpu").manual_seed(0)
    for h, (A, U, s_full) in enumerate(per_head):
        An = A / A.sum(-1, keepdim=True).clamp_min(1e-30)
        band = torch.zeros(int(bins.max()) + 1, C1 - C0, device=dev)
        for b in range(band.shape[0]):
            m = doc & (bins == b)
            if m.any():
                band[b] = An[m].mean(0)
        r = band[bins.clamp_min(0)]
        r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        io = {}
        for nm, v in U.items():
            if v.sum() > 0:
                io[nm] = iota_of(An, v / v.sum(), r)[doc]
        iota = io["all"]
        s16 = s_all[h]                                                   # doc keys == doc query positions
        uall = U["all"] / U["all"].sum()
        mg = An[doc].mean(0); mg = mg / mg.norm().clamp_min(1e-30)
        iota_raw = cos(An, uall)[doc]
        iota_glob = cos(An - (An @ mg).unsqueeze(-1) * mg, uall - (uall @ mg) * mg)[doc]
        sp = s16 / s16.sum().clamp_min(1e-30)                            # window attention profile over doc keys
        rho = iota_of(An, sp, r)[doc]
        bd = bins[doc]; sb = torch.zeros_like(s16)
        for b_ in bd.unique():
            mm = bd == b_; sb[mm] = s16[mm] / s16[mm].mean().clamp_min(1e-30)
        W = {"s (W=16)": s16, "s (W=1)": A[N - 1], "s (W=64)": A[N - 64:].sum(0), "s squared": s16 ** 2,
             "s layer mean": s_layer, "H2O transfer": A[:N].sum(0),
             "recency 1/d": 1.0 / (dist[doc] + 1.0), "random": torch.rand(C1 - C0, generator=gen_rand).to(dev), "s in-band": sb, "rho": rho,
             "s^(1/2) (T=2)": s16 ** 0.5, "s^(1/4) (T=4)": s16 ** 0.25, "s^(1/8) (T=8)": s16 ** 0.125,
             "s (W=4)": A[N - 4:].sum(0), "s (W=8)": A[N - 8:].sum(0), "s (W=32)": A[N - 32:].sum(0), "s (W=128)": A[N - 128:].sum(0),
             "uniform": torch.ones(C1 - C0, device=dev)}
        rec = dict(layer=module.layer_idx, head=h, N=N, gen=S - N, rivals={m: score(w, iota) for m, w in W.items()})
        # end target: which keys each weighting's score keeps
        kd = (N - 1 - (C0 + torch.arange(C1 - C0, device=dev))).float()          # key distance from the end
        topu = torch.zeros(C1 - C0, dtype=torch.bool, device=dev); topu[uall.topk(KEEP).indices] = True
        far = topu & (kd > 2048)
        def keyrec(sc, w=None):
            kept = torch.zeros_like(topu); kept[sc.topk(KEEP).indices] = True
            out = dict(recall=float((kept & topu).sum() / KEEP), umass=float(uall[kept].sum()),
                       far_recall=float((kept & far).sum() / far.sum()) if far.any() else None)
            if w is not None:
                w = w.clamp_min(0); w = w / w.sum().clamp_min(1e-30); out["ess"] = float(1.0 / (w ** 2).sum())
            return out
        Ad = A[C0:C1]
        rec["keyrec"] = {m: keyrec(w.clamp_min(0) / w.clamp_min(0).sum().clamp_min(1e-30) @ Ad, w) for m, w in W.items()
                         if m not in ("rho",)}
        rec["keyrec"]["SnapKV"] = keyrec(s16)
        wf = s_full / s_full.sum().clamp_min(1e-30)                     # the deployed Q-Trust weights, all positions
        rec["keyrec"]["Q-Trust (actual)"] = keyrec(wf @ A, wf)
        rec["keyrec"]["H2O (actual)"] = keyrec(A.sum(0))
        rec["keyrec"]["iota oracle"] = keyrec(iota.clamp_min(0) / iota.clamp_min(0).sum().clamp_min(1e-30) @ Ad, iota.clamp_min(0))
        # locality control: keys within +-RLOC of the query masked out
        qd = qpos[doc].view(-1, 1); kp = (C0 + torch.arange(C1 - C0, device=dev)).view(1, -1)
        M = ((qd - kp).abs() > RLOC).float()
        X = An[doc] * M; X = X / X.sum(-1, keepdim=True).clamp_min(1e-30)
        Y = uall.view(1, -1) * M
        rd = r[doc]
        iota_nl = cos(X - (X * rd).sum(-1, keepdim=True) * rd, Y - (Y * rd).sum(-1, keepdim=True) * rd)
        del X, Y, M
        rec["nonlocal"] = {m: score(W[m], iota_nl) for m in ("s (W=16)", "recency 1/d", "H2O transfer", "uniform")}
        rec["nonlocal"]["sp_iota_nl"] = spear(iota, iota_nl)
        rec["mech"] = dict(sp_s_rho=spear(s16, rho), sp_rho_iota=spear(rho, iota), sp_s_iota=spear(s16, iota))
        # section 3.2: raw cosine with u of the content-free mean row vs the best individual rows (top 1%)
        rawc = cos(An[doc], uall)
        rec["rawcos"] = dict(meanrow=float(cos(An[doc].mean(0), uall)),
                             top1=float(rawc.topk(max(1, rawc.numel() // 100)).values.mean()))
        rec["variants"] = {"iota_raw": score(s16, iota_raw), "iota_glob": score(s16, iota_glob),
                           "iota_raw_uniform": score(torch.ones_like(s16), iota_raw)}
        # noise ceiling: predict iota_b from iota_a (and s against iota_b for the same target)
        if "even" in io and "odd" in io:
            a_, b_ = io["even"], io["odd"]
            ka = set(a_.argsort(descending=True)[: a_.numel() // 4].tolist())
            kb = set(b_.argsort(descending=True)[: b_.numel() // 4].tolist())
            rec["ceiling"] = dict(spearman_ab=spear(a_, b_), jacc25=len(ka & kb) / len(ka | kb),
                                  iota_a=score(a_, b_), s=score(s16, b_))
        if "early" in io and "late" in io:
            rec["steps"] = {nm: score(s16, io[nm]) for nm in ("early", "late")}
        # curve over s percentiles
        rk = s16.argsort().argsort().float() / max(s16.numel() - 1, 1)
        bi = (rk * NBIN).long().clamp(max=NBIN - 1)
        top25 = torch.zeros_like(iota, dtype=torch.bool); top25[iota.argsort(descending=True)[: iota.numel() // 4]] = True
        rec["curve"] = dict(iota=[float(iota[bi == b].mean()) for b in range(NBIN)],
                            informative=[float(top25[bi == b].float().mean()) for b in range(NBIN)])
        ST["out"].append(rec)
    return out.transpose(1, 2).contiguous(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-8b")
    ap.add_argument("--data", default="datasets/longbench", help="LongBench jsonl directory")
    ap.add_argument("--per", type=int, default=20, help="prompts per task (20 -> the paper's 315 prompts)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); out_dir = os.path.abspath(a.out); data = os.path.abspath(a.data)
    chdir_root()
    import utils as U_
    from transformers import AttentionInterface
    AttentionInterface.register("link_capture", hook)
    torch.set_grad_enabled(False)
    maxgen = json.load(open("config/dataset2maxlen.json"))
    rows = load_rows(data, a.per, a.shard)
    model, tok = U_.load_model(a.model)
    os.makedirs(out_dir, exist_ok=True)
    for d, j, r in tqdm(rows, desc=f"shard {a.shard}"):
        fn = f"{out_dir}/{d}_{j}.json"
        if os.path.exists(fn):
            continue
        x = encode_prompt(tok, a.model, d, r, maxgen, model.device)
        N = x.shape[1]
        if N < 1024:
            continue
        model.init_cache(None)
        with torch.inference_mode():
            g = model.generate(input_ids=x, max_new_tokens=int(maxgen[d]), do_sample=False,
                               pad_token_id=tok.eos_token_id, tokenizer=tok, stop_strings=U_.chat_stop_strings(a.model))
        if g.shape[1] - N < 4:
            continue
        ST.update(N=N, out=[])
        model.config._attn_implementation = "link_capture"
        try:
            with torch.inference_mode():
                model(input_ids=g, use_cache=False, logits_to_keep=1)
        finally:
            model.config._attn_implementation = ATTN_IMPL
        json.dump(dict(dataset=d, idx=j, N=N, gen=int(g.shape[1] - N), pairs=ST["out"]), open(fn, "w"))
        torch.cuda.empty_cache()
    print("[link] SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
