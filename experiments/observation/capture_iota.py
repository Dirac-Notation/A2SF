"""Section 2 (Figure 1a/c, Section 2.2-2.3): per-query informativeness iota on sampled LongBench (Llama-3.1-8B).

    iota(q) = cos( A(q,.) - proj_{m_d} A(q,.),  u - proj_{m_d} u )

m_d is the mean attention row of the queries in q's own log-distance band, A is measured on the contested
keys [4, N-16), and u is the attention the full-cache generation (the task's own length) gives to them.
The band mean carries what every query at that distance shares, so sinks and RoPE locality are removed by
construction and the null value is 0.

Every layer and every KV head is measured, so nothing depends on which layers or heads were sampled.
Per (layer, KV head) we record
  iota    distribution over document queries, and for the window queries
  where   mean iota per distance band, the share of the top 1% that falls in each band, and the share of
          the positions in that band (so the report can be density-normalized)
  w       for each method the deployed weighting over ALL positions (sinks and window included, exactly as
          the method runs): E_w[iota], n_eff, and the mass it puts on the top 1% / 10% of iota
  link    spearman(s, iota) over document queries, what the top 1% / 10% of s recalls, and the snap mass lift
  proxy   the kept set each method's score produces: recall of the top-112 of u, and the share of u it keeps
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F_
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ATTN_IMPL, chdir_root, encode_prompt, load_rows

NSINK, WIN, KEEP = 4, 16, 112
EDGES = [0, 16, 64, 256, 512, 1024, 2048, 4096, 10 ** 9]
HLO, HHI, HBINS = -0.30, 0.90, 240          # histogram of per-query iota (bin 0.005), for the long-tail figure
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
    band_id = torch.bucketize(dist, torch.tensor(EDGES[1:], device=dev, dtype=torch.float), right=True)
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
        s_raw = A[C1:].sum(0)                                            # snap score over contested keys
        s = torch.cat([torch.zeros(C0, device=dev), s_raw, torch.zeros(N - C1, device=dev)])
        An = A / A.sum(-1, keepdim=True).clamp_min(1e-30)
        u = U / U.sum().clamp_min(1e-30)
        doc = (qpos >= C0) & (qpos < C1)
        band = torch.zeros(int(bins.max()) + 1, C1 - C0, device=dev)
        for b in range(band.shape[0]):
            m = doc & (bins == b)                   # document queries only, as in the intervention
            if m.any():
                band[b] = An[m].mean(0)
        r = band[bins.clamp_min(0)]
        r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        iota = cos(An - (An * r).sum(-1, keepdim=True) * r, u - (u * r).sum(-1, keepdim=True) * r)
        gen = torch.Generator(device="cpu").manual_seed(0)
        sh = An[doc][torch.randint(0, int(doc.sum()), (16,), generator=gen).to(dev)]
        sh = sh[:, torch.randperm(sh.shape[1], device=dev)]
        rsh = band[bins[doc][:16].clamp_min(0)]
        rsh = rsh / rsh.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        null_shuf = float(cos(sh - (sh * rsh).sum(-1, keepdim=True) * rsh, u - (u * rsh).sum(-1, keepdim=True) * rsh).mean())

        dv = torch.where(doc, iota, torch.full_like(iota, -1e9))
        nd = int(doc.sum())
        order = dv.argsort(descending=True)
        top1, top10, top25 = order[:max(1, nd // 100)], order[:max(1, nd // 10)], order[:max(1, nd // 4)]
        hist = torch.histc(iota[doc].clamp(HLO, HHI - 1e-6), bins=HBINS, min=HLO, max=HHI)
        rec = dict(layer=module.layer_idx, head=h, N=N, null_shuffled=null_shuf,
                   hist=[int(x) for x in hist.tolist()],
                   iota=dict(median=float(iota[doc].median()), mean=float(iota[doc].mean()),
                             top1=float(iota[top1].mean()), top10=float(iota[top10].mean()), top25=float(iota[top25].mean()),
                             bottom10=float(iota[order[-max(1, nd // 10):]].mean()),
                             window=float(iota[C1:].mean()), last=float(iota[N - 1]),
                             pos_frac=float((iota[doc] > 0).float().mean())))
        # where the informative queries are
        prof, share, share10, share25, pos = [], [], [], [], []
        for j in range(len(EDGES) - 1):
            m = (band_id == j) & doc
            prof.append(float(iota[m].mean()) if m.any() else None)
            share.append(float(band_id[top1].eq(j).float().mean()))
            share10.append(float(band_id[top10].eq(j).float().mean()))
            share25.append(float(band_id[top25].eq(j).float().mean()))
            pos.append(float(m.float().sum() / nd))
        rec["dist"] = dict(edges=EDGES, iota=prof, top1_share=share, top10_share=share10, top25_share=share25,
                           pos_share=pos, top1_dist_median=float(dist[top1].median()),
                           top10_dist_median=float(dist[top10].median()), top25_dist_median=float(dist[top25].median()),
                           top25_beyond64=float((dist[top25] >= 64).float().mean()))
        # equal-width views: relative position inside the document (0 = first document token, 1 = last) in
        # 10 bins, and absolute distance from the end in 512-token bins. The log bands above grow with
        # distance, so their shares mostly track how many positions a band holds; these do not.
        rel = ((qpos - C0).float() / max(C1 - C0, 1)).clamp(0, 1 - 1e-6)
        rb = (rel * 10).long()
        ab = (dist / 512).long().clamp(max=15)
        ab2 = (dist / 256).long().clamp(max=31)
        ab16 = (dist / 16).long().clamp(max=128)                           # 16-token bins to 2048, then one 2K+ bin
        eq = {}
        for nm, b_, K in (("rel", rb, 10), ("abs512", ab, 16), ("abs256", ab2, 32), ("abs16", ab16, 129)):
            eq[nm] = dict(pos=[float(((b_ == k) & doc).float().sum() / nd) for k in range(K)],
                          **{t: [float(b_[ix].eq(k).float().mean()) for k in range(K)]
                             for t, ix in (("top1", top1), ("top10", top10), ("top25", top25))})
        rec["dist_eq"] = eq
        # window queries against the document threshold: share of the last WIN queries whose iota is at least the
        # smallest iota among the document top 25% (the same cut that defines an informative document query)
        # no document query shares the window's distance bands (0-15), so the shared part of the window rows is
        # removed with the nearest document band (distance 16-31, log2 band 4) to keep the same yardstick
        thr = iota[top25].min()
        rw = band[4] / band[4].norm().clamp_min(1e-30)
        Aw = An[C1:N]
        iw = cos(Aw - (Aw @ rw).unsqueeze(-1) * rw, u - (u @ rw) * rw)
        rec["window_inf"] = dict(share=float((iw >= thr).float().mean()), mean=float(iw.mean()), n=int(N - C1))
        # deployed weightings, over all positions
        W = {"TOVA": torch.zeros(N, device=dev), "SnapKV": torch.zeros(N, device=dev),
             "H2O": torch.ones(N, device=dev), "Q-Trust": s.clamp_min(0), "oracle": (iota * doc).clamp_min(0)}
        W["TOVA"][N - 1] = 1.0; W["SnapKV"][C1:] = 1.0
        W = {m: w / w.sum().clamp_min(1e-30) for m, w in W.items()}
        topU = torch.zeros(C1 - C0, device=dev); topU.scatter_(0, u.topk(KEEP).indices, 1.0)
        rec["methods"] = {}
        for m, w in W.items():
            score = w @ A
            kept = torch.zeros(C1 - C0, device=dev); kept.scatter_(0, score.topk(KEEP).indices, 1.0)
            rec["methods"][m] = dict(Ew=float((w * iota).sum()), neff=float(1 / (w ** 2).sum()),
                                     top1_mass=float(w[top1].sum()), top10_mass=float(w[top10].sum()),
                                     top25_mass=float(w[top25].sum()), doc_mass=float(w[doc].sum()),
                                     recall=float((kept * topU).sum() / KEEP),
                                     u_kept=float((u * kept).sum()))
        sv = torch.where(doc, s, torch.full_like(s, -1e9))
        rec["link"] = dict(spearman=spear(sv[doc], iota[doc]))
        for p, tt in ((1, top1), (10, top10), (25, top25)):
            kk = max(1, int(nd * p / 100))
            tops = set(sv.argsort(descending=True)[:kk].tolist())
            rec["link"][f"recall{p}"] = len(tops & set(tt.tolist())) / kk
            rec["link"][f"lift{p}"] = float(W["Q-Trust"][tt].sum() / (p / 100))
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
    AttentionInterface.register("iota_capture", hook)
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
        model.config._attn_implementation = "iota_capture"
        try:
            with torch.inference_mode():
                model(input_ids=g, use_cache=False, logits_to_keep=1)
        finally:
            model.config._attn_implementation = ATTN_IMPL
        json.dump(dict(dataset=d, idx=j, N=N, gen=int(g.shape[1] - N), pairs=ST["out"]), open(fn, "w"))
        torch.cuda.empty_cache()
    print("[iota] SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
