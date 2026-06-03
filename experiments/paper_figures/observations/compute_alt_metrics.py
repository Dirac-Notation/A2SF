"""Compute Ranked (DCG-style) and Tanimoto similarity metrics from saved prefill data.

Loads oracle_score.npz + prefill_chunked.pt (saved by coord_descent.py) for each task.
Does NOT re-run the model — all computation is pure numpy/torch.

Extends coord_descent.npz with:
  j_uniform_ranked, j_snap16_ranked, j_sigmoid_ranked, j_optimal_ranked
  j_uniform_tanimoto, j_snap16_tanimoto, j_sigmoid_tanimoto, j_optimal_tanimoto

Then regenerates all three figure types:
  obs1_jaccard_recovery.pdf  — original Jaccard (unchanged keys)
  obs1_ranked_recovery.pdf   — Ranked DCG similarity (same layout)
  obs1_tanimoto_recovery.pdf — Tanimoto similarity (same layout)
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 13, "axes.linewidth": 1.0,
    "figure.dpi": 150,
})

BUDGET      = int(os.environ.get("BUDGET", "128"))
_PLOTS      = "/home/smp9898/A2SF/experiments/temporal_bias/plots"
BASE        = _PLOTS if BUDGET == 128 else os.path.join(_PLOTS, f"b{BUDGET}")
SUFFIX      = "" if BUDGET == 128 else f"_b{BUDGET}"
N_KV_HEADS  = 8     # llama3.2-1b
GROUP_SIZE  = 4     # gs = n_heads / n_kv_heads
LOCAL_RATIO = 0.125

TASKS = [
    ("Single-doc_QA/qasper",     "Single-doc QA"),
    ("Multi-doc_QA/hotpotqa",    "Multi-doc QA"),
    ("Summarization/gov_report", "Summarization"),
    ("Few_Shot/samsum",          "Few-Shot"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(np.clip(a * (x - b), -30.0, 30.0)))


def fit_sigmoid_to_row(w_row, d_centers, W):
    """Fit descending sigmoid to a single-sample weight vector. Returns sig_w (G,)."""
    try:
        popt, _ = curve_fit(sigmoid, d_centers, w_row,
                            p0=[0.05, float(W) / 4.0],
                            bounds=([0.0, 0.0], [5.0, float(W)]),
                            maxfev=5000)
        return sigmoid(d_centers, *popt).astype(np.float32)
    except Exception:
        return np.ones_like(w_row) * 0.5


def idcg_table(max_k: int) -> np.ndarray:
    """Precomputed IDCG values for k=0..max_k."""
    arr = np.zeros(max_k + 1, dtype=np.float64)
    for k in range(1, max_k + 1):
        arr[k] = arr[k - 1] + 1.0 / np.log2(k + 1)  # 1/log2(rank+2) where rank=k-1
    return arr


IDCG_TABLE = idcg_table(300)


def _ranked_sim(oracle_kv: torch.Tensor, method_idx: torch.Tensor,
                sel_b: int) -> float:
    """Ranked (DCG-style) similarity.

    oracle_kv:  (L, n_kv_heads, S) float32 — oracle attention scores in kv-head space
    method_idx: (L, n_kv_heads, sel_b) int64 — method-selected token indices
    Returns scalar ranked similarity (mean over L × n_kv_heads, normalized by IDCG).
    """
    L, K, S = oracle_kv.shape
    # rank_of_token[l, kv_h, t] = oracle rank of token t (0 = most important)
    rank_of_token = oracle_kv.argsort(dim=-1, descending=True).argsort(dim=-1).float()
    # ranks of each method-selected token
    ranks = rank_of_token.gather(-1, method_idx)          # (L, K, sel_b)
    dcg   = (1.0 / torch.log2(ranks + 2.0)).sum(dim=-1)  # (L, K)
    idcg  = float(IDCG_TABLE[sel_b])
    if idcg < 1e-12:
        return 0.0
    return float((dcg / idcg).mean().item())


def _tanimoto_sim(oracle_kv: torch.Tensor, method_kv: torch.Tensor,
                  seq_len: int) -> float:
    """Tanimoto similarity between normalized oracle and method score vectors.

    oracle_kv, method_kv: (L, n_kv_heads, S) float32 — scores in kv-head space.
    Uses only tokens 0..seq_len (mask out padding).
    Returns scalar Tanimoto averaged over L × n_kv_heads.
    """
    o = oracle_kv[:, :, :seq_len].float().clamp(min=0.0)
    m = method_kv[:, :, :seq_len].float().clamp(min=0.0)
    # L1-normalise per head so scales are comparable
    o = o / (o.sum(dim=-1, keepdim=True) + 1e-12)
    m = m / (m.sum(dim=-1, keepdim=True) + 1e-12)
    inter = torch.minimum(o, m).sum(dim=-1)              # (L, n_kv_heads)
    union = torch.maximum(o, m).sum(dim=-1).clamp(min=1e-12)
    return float((inter / union).mean().item())


def _to_kv_head(t: torch.Tensor, n_kv: int, gs: int) -> torch.Tensor:
    """Reduce query-head tensor (..., n_heads, S) to (..., n_kv_heads, S) via GQA sum."""
    *batch, H, S = t.shape
    return t.view(*batch, n_kv, gs, S).sum(dim=-2)


def _resolve_N_seqlens(cd, os_npz, pf_list):
    """Return (N, seq_lens) robust to coord_descent.npz missing seq_lens (e.g. b512)."""
    N = len(pf_list)
    if "seq_lens" in cd.files:
        seq_lens = cd["seq_lens"].astype(int)[:N]
    else:
        seq_lens = os_npz["seq_lens"].astype(int)[:N]
    return N, seq_lens


def compute_alt_metrics(path: str, budget: int):
    """Compute ranked & tanimoto running curves for all weighting schemes.

    Loads oracle_score.npz + coord_descent.npz + prefill_chunked.pt from `path`.
    Returns dict with arrays of shape (N, G) for each new metric key.

    Speed notes:
    - Pre-reduces prefill chunks to kv_head space once per sample (avoids repeat inside loop).
    - Processes all 4 schemes in a single joint pass per chunk (4× speedup vs sequential).
    - oracle_kv normalisation computed once per sample (not per chunk).
    """
    import gc, time
    cd_path  = os.path.join(path, "coord_descent.npz")
    os_path  = os.path.join(path, "oracle_score.npz")
    pf_path  = os.path.join(path, "prefill_chunked.pt")

    cd      = np.load(cd_path)
    os_npz  = np.load(os_path)
    t0 = time.time()
    print("    loading prefill_chunked.pt ...", flush=True)
    pf_list = torch.load(pf_path, map_location="cpu", weights_only=False)
    print(f"    loaded in {time.time()-t0:.1f}s", flush=True)

    G         = cd["w_cd"].shape[1]
    chunk     = int(cd["chunk"])
    W         = int(cd["window"])
    N, seq_lens = _resolve_N_seqlens(cd, os_npz, pf_list)
    w_cds     = cd["w_cd"][:N]                   # (N, G)
    kv        = N_KV_HEADS
    gs        = GROUP_SIZE

    local_b   = max(1, int(budget * LOCAL_RATIO))
    sel_b     = budget - local_b

    snap16_G  = max(1, 16 // chunk)
    d_centers = np.arange(G, dtype=np.float32) * chunk + (chunk - 1) / 2.0

    answer_score_all = os_npz["answer_score"]     # (N, L, n_heads, max_S) fp16

    S_NAMES = ["uniform", "snap16", "sigmoid", "optimal"]
    N_SCH   = len(S_NAMES)
    out = {f"j_{s}_ranked":   np.zeros((N, G), dtype=np.float32) for s in S_NAMES}
    out.update({f"j_{s}_tanimoto": np.zeros((N, G), dtype=np.float32) for s in S_NAMES})

    idcg = float(IDCG_TABLE[min(sel_b, 300)])

    for i in range(N):
        t1 = time.time()
        S_i  = seq_lens[i]
        pf_i = pf_list[i]                          # (G, L, n_heads, S_i) fp16

        # ── pre-reduce pf to kv-head space once per sample ─────────────────
        G_i, L, n_heads, S_pf = pf_i.shape
        # (G, L, n_heads, S_pf) fp16 → (G, L, n_kv_heads, S_pf) float32
        _pf_f32 = pf_i.float()
        pf_kv   = _pf_f32.view(G_i, L, kv, gs, S_pf).sum(dim=3)  # (G, L, kv, S_pf)
        del _pf_f32

        # ── oracle in kv space ──────────────────────────────────────────────
        oracle_qh = torch.as_tensor(answer_score_all[i, :, :, :S_i],
                                    dtype=torch.float32)              # (L, n_heads, S_i)
        oracle_kv = oracle_qh.view(L, kv, gs, S_i).sum(dim=2)       # (L, kv, S_i)

        rank_of_token = (oracle_kv.argsort(dim=-1, descending=True)
                                  .argsort(dim=-1).float())           # (L, kv, S_i)

        # ── normalised oracle for tanimoto ──────────────────────────────────
        o_clamped = oracle_kv.clamp(min=0.0)
        o_norm    = o_clamped / (o_clamped.sum(-1, keepdim=True) + 1e-12)  # (L, kv, S_i)

        # ── build weights for all 4 schemes: (N_SCH, G) ─────────────────────
        ws_all = np.stack([
            np.ones(G, dtype=np.float32),
            np.where(np.arange(G) < snap16_G, 1.0, 0.0).astype(np.float32),
            fit_sigmoid_to_row(w_cds[i], d_centers, W),
            w_cds[i].astype(np.float32),
        ])                                                             # (4, G)
        ws_tensor = torch.from_numpy(ws_all)                          # (4, G)

        # ── 4-scheme accumulator in kv-head space ───────────────────────────
        accums = torch.zeros(N_SCH, L, kv, S_pf, dtype=torch.float32)

        for g in range(G):
            block = pf_kv[g]                           # (L, kv, S_pf)
            for si in range(N_SCH):
                w = float(ws_tensor[si, g])
                if w > 1e-9:
                    accums[si] += block * w             # in-place (L, kv, S_pf)

            # ── slice to actual seq_len ──────────────────────────────────────
            acc = accums[:, :, :, :S_i].clone()        # (4, L, kv, S_i)

            # ── local forcing: set last local_b to max per scheme-head ───────
            if local_b > 0 and S_i > local_b:
                big = acc.max().item() + 1.0
                acc[:, :, :, -local_b:] += big

            # ── top-sel_b indices — ranked similarity ─────────────────────
            actual_k = min(sel_b, S_i)
            _, idx = acc.topk(actual_k, dim=-1)         # (4, L, kv, actual_k)

            # rank of each selected token under oracle ordering
            rot_exp = rank_of_token.unsqueeze(0).expand(N_SCH, -1, -1, -1)
            ranks   = rot_exp.gather(-1, idx)            # (4, L, kv, actual_k)
            dcg     = (1.0 / torch.log2(ranks + 2.0)).sum(-1)         # (4, L, kv)
            # mean over L × kv, normalise
            ranked_vals = (dcg / idcg).mean(dim=(1, 2))               # (4,)

            # ── Tanimoto similarity ───────────────────────────────────────
            m_raw  = accums[:, :, :, :S_i].clamp(min=0.0)             # (4, L, kv, S_i)
            m_norm = m_raw / (m_raw.sum(-1, keepdim=True) + 1e-12)
            o_exp  = o_norm.unsqueeze(0)                               # (1, L, kv, S_i)
            inter  = torch.minimum(o_exp, m_norm).sum(-1)             # (4, L, kv)
            union  = torch.maximum(o_exp, m_norm).sum(-1).clamp(1e-12)
            tan_vals = (inter / union).mean(dim=(1, 2))                # (4,)

            for si, sname in enumerate(S_NAMES):
                out[f"j_{sname}_ranked"][i, g]   = float(ranked_vals[si])
                out[f"j_{sname}_tanimoto"][i, g] = float(tan_vals[si])

        del pf_kv, accums
        print(f"  [{i+1}/{N}] seq_len={S_i}  t={time.time()-t1:.1f}s  "
              f"ranked={out['j_optimal_ranked'][i,-1]:.3f}  "
              f"tanimoto={out['j_optimal_tanimoto'][i,-1]:.3f}",
              flush=True)

    del pf_list
    gc.collect()
    return out


_GRID_TAN  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_GRID_NZERO = torch.tensor(_GRID_TAN[1:], dtype=torch.float32)  # 10 non-zero candidates


def compute_tanimoto_optimal(path: str):
    """Coord-descent optimal weights that directly maximise Tanimoto similarity.

    For each chunk g (recent → distant), sweeps 10 weight candidates in a
    vectorised pass (10 accumulators at once) and picks the best.  No top-k
    selection: Tanimoto is evaluated on the continuous normalised score vectors.

    Returns dict:
      w_tan         (N, G) — per-sample Tanimoto-optimal chunk weights
      j_tan_optimal (N, G) — running Tanimoto with w_tan (non-decreasing)
      j_tan_sigmoid (N, G) — running Tanimoto with sigmoid fit to w_tan
    """
    import gc, time

    cd_path = os.path.join(path, "coord_descent.npz")
    os_path = os.path.join(path, "oracle_score.npz")
    pf_path = os.path.join(path, "prefill_chunked.pt")

    cd      = np.load(cd_path)
    os_npz  = np.load(os_path)
    print("    loading prefill_chunked.pt ...", flush=True)
    t0 = time.time()
    pf_list = torch.load(pf_path, map_location="cpu", weights_only=False)
    print(f"    loaded in {time.time()-t0:.1f}s", flush=True)

    G        = cd["w_cd"].shape[1]
    chunk    = int(cd["chunk"])
    W        = int(cd["window"])
    N, seq_lens = _resolve_N_seqlens(cd, os_npz, pf_list)
    kv = N_KV_HEADS; gs = GROUP_SIZE
    d_centers = np.arange(G, dtype=np.float32) * chunk + (chunk - 1) / 2.0

    answer_score_all = os_npz["answer_score"]      # (N, L, n_heads, max_S) fp16

    w_tan_all     = np.zeros((N, G), dtype=np.float32)
    j_tan_opt_all = np.zeros((N, G), dtype=np.float32)
    j_tan_sig_all = np.zeros((N, G), dtype=np.float32)

    for i in range(N):
        t1   = time.time()
        S_i  = seq_lens[i]
        pf_i = pf_list[i]                               # (G, L, n_heads, S_pf) fp16
        G_i, L, n_heads, S_pf = pf_i.shape

        _pf_f32 = pf_i.float()
        pf_kv   = (_pf_f32.view(G_i, L, kv, gs, S_pf)
                          .sum(dim=3)[:, :, :, :S_i])   # (G, L, kv, S_i)
        del _pf_f32

        oracle_qh  = torch.as_tensor(answer_score_all[i, :, :, :S_i],
                                     dtype=torch.float32)
        oracle_kv  = oracle_qh.view(L, kv, gs, S_i).sum(dim=2)   # (L, kv, S_i)
        o_raw      = oracle_kv.clamp(min=0.0)
        oracle_norm = o_raw / (o_raw.sum(-1, keepdim=True) + 1e-12)

        def tan_scalar(acc):
            m  = acc.clamp(min=0.0)
            mn = m / (m.sum(-1, keepdim=True) + 1e-12)
            return float((torch.minimum(oracle_norm, mn).sum(-1) /
                          torch.maximum(oracle_norm, mn).sum(-1).clamp(1e-12)).mean())

        # ── Tanimoto coord descent ──────────────────────────────────────────
        w_tan    = np.zeros(G, dtype=np.float32)
        accum    = torch.zeros(L, kv, S_i, dtype=torch.float32)
        j_weight = np.zeros(G, dtype=np.float32)

        for g in range(G):
            block = pf_kv[g]                             # (L, kv, S_i)

            if g == 0:
                best_w = 1.0  # force most recent chunk to weight 1.0
            else:
                base_j = tan_scalar(accum)
                best_w = 0.0

                # All 10 non-zero candidates in one vectorised pass
                temps  = (accum.unsqueeze(0) +
                          block.unsqueeze(0) * _GRID_NZERO.view(-1, 1, 1, 1))
                m      = temps.clamp(min=0.0)
                mn     = m / (m.sum(-1, keepdim=True) + 1e-12)
                o_exp  = oracle_norm.unsqueeze(0)
                scores = ((torch.minimum(o_exp, mn).sum(-1) /
                           torch.maximum(o_exp, mn).sum(-1).clamp(1e-12))
                          .mean(dim=(1, 2)))              # (10,)

                best_idx = int(scores.argmax())
                if float(scores[best_idx]) > base_j + 1e-9:
                    best_w = float(_GRID_NZERO[best_idx])

            if best_w > 1e-9:
                accum = accum + block * best_w
            w_tan[g]    = best_w
            j_weight[g] = tan_scalar(accum)

        w_tan_all[i]     = w_tan
        j_tan_opt_all[i] = j_weight

        # ── sigmoid fit to w_tan, then running Tanimoto ─────────────────────
        sig_w     = fit_sigmoid_to_row(w_tan, d_centers, W)
        accum_sig = torch.zeros(L, kv, S_i, dtype=torch.float32)
        for g in range(G):
            if sig_w[g] > 1e-9:
                accum_sig = accum_sig + pf_kv[g] * float(sig_w[g])
            mn = accum_sig.clamp(0) / (accum_sig.clamp(0).sum(-1, keepdim=True) + 1e-12)
            j_tan_sig_all[i, g] = float(
                (torch.minimum(oracle_norm, mn).sum(-1) /
                 torch.maximum(oracle_norm, mn).sum(-1).clamp(1e-12)).mean())

        del pf_kv, accum, accum_sig
        print(f"  [{i+1}/{N}] seq_len={S_i}  t={time.time()-t1:.1f}s  "
              f"j_tan_opt={j_weight[-1]:.3f}  j_tan_sig={j_tan_sig_all[i,-1]:.3f}",
              flush=True)

    del pf_list; gc.collect()
    return {
        "w_tan":         w_tan_all,
        "j_tan_optimal": j_tan_opt_all,
        "j_tan_sigmoid": j_tan_sig_all,
    }


def compute_jaccard_optimal_from_prefill(path: str, budget: int = BUDGET):
    """Coord-descent optimal weights maximising Jaccard similarity, from saved prefill data.

    Forces g=0 (most recent chunk) to weight 1.0, then greedily searches g=1..G-1.
    Uses the same 10-candidate vectorised GRID as compute_tanimoto_optimal.

    Returns dict:
      w_cd_f1  (N, G) — per-sample Jaccard-optimal chunk weights (g0 forced to 1.0)
      j_cd_f1  (N, G) — running Jaccard with w_cd_f1
    """
    import gc, time

    cd_path = os.path.join(path, "coord_descent.npz")
    os_path = os.path.join(path, "oracle_score.npz")
    pf_path = os.path.join(path, "prefill_chunked.pt")

    cd     = np.load(cd_path)
    os_npz = np.load(os_path)
    print("    loading prefill_chunked.pt ...", flush=True)
    t0 = time.time()
    pf_list = torch.load(pf_path, map_location="cpu", weights_only=False)
    print(f"    loaded in {time.time()-t0:.1f}s", flush=True)

    G        = cd["w_cd"].shape[1]
    chunk    = int(cd["chunk"])
    W        = int(cd["window"])
    N, seq_lens = _resolve_N_seqlens(cd, os_npz, pf_list)
    kv = N_KV_HEADS; gs = GROUP_SIZE

    local_b = max(1, int(budget * LOCAL_RATIO))
    sel_b   = budget - local_b

    answer_score_all = os_npz["answer_score"]

    w_cd_f1_all = np.zeros((N, G), dtype=np.float32)
    j_cd_f1_all = np.zeros((N, G), dtype=np.float32)

    for i in range(N):
        t1   = time.time()
        S_i  = seq_lens[i]
        pf_i = pf_list[i]
        G_i, L, n_heads, S_pf = pf_i.shape

        _pf_f32 = pf_i.float()
        pf_kv   = (_pf_f32.view(G_i, L, kv, gs, S_pf)
                          .sum(dim=3)[:, :, :, :S_i])   # (G, L, kv, S_i)
        del _pf_f32

        oracle_qh = torch.as_tensor(answer_score_all[i, :, :, :S_i], dtype=torch.float32)
        oracle_kv = oracle_qh.view(L, kv, gs, S_i).sum(dim=2)   # (L, kv, S_i)

        actual_k = min(sel_b, S_i)

        # Oracle top-k mask with local forcing (fixed per sample)
        o_acc = oracle_kv.clone()
        if local_b > 0 and S_i > local_b:
            o_acc[:, :, -local_b:] += o_acc.max().item() + 1.0
        _, o_idx = o_acc.topk(actual_k, dim=-1)         # (L, kv, actual_k)
        oracle_mask = torch.zeros(L, kv, S_i, dtype=torch.float32)
        oracle_mask.scatter_(-1, o_idx, 1.0)

        def jaccard_scalar(acc):
            a = acc.clone()
            bias = a.max().item() + 1.0 if a.max().item() > 0 else 1.0
            if local_b > 0 and S_i > local_b:
                a[:, :, -local_b:] += bias
            _, idx = a.topk(actual_k, dim=-1)
            m_mask = torch.zeros(L, kv, S_i, dtype=torch.float32)
            m_mask.scatter_(-1, idx, 1.0)
            inter = (m_mask * oracle_mask).sum(-1)       # (L, kv)
            union = 2 * actual_k - inter
            return float((inter / union.clamp(1e-12)).mean())

        w_cd_f1  = np.zeros(G, dtype=np.float32)
        accum    = torch.zeros(L, kv, S_i, dtype=torch.float32)
        j_weight = np.zeros(G, dtype=np.float32)

        for g in range(G):
            block = pf_kv[g]                             # (L, kv, S_i)

            if g == 0:
                best_w = 1.0  # force most recent chunk to weight 1.0
            else:
                base_j = jaccard_scalar(accum)
                best_w = 0.0

                # Vectorised: 10 non-zero candidates at once
                temps = (accum.unsqueeze(0) +
                         block.unsqueeze(0) * _GRID_NZERO.view(-1, 1, 1, 1))  # (10, L, kv, S_i)
                t_local = temps.clone()
                if local_b > 0 and S_i > local_b:
                    big = t_local.max().item() + 1.0
                    t_local[:, :, :, -local_b:] += big
                _, m_idx = t_local.topk(actual_k, dim=-1)  # (10, L, kv, actual_k)
                del t_local
                m_masks = torch.zeros(10, L, kv, S_i, dtype=torch.float32)
                m_masks.scatter_(-1, m_idx, 1.0)
                o_exp = oracle_mask.unsqueeze(0)           # (1, L, kv, S_i)
                inter = (m_masks * o_exp).sum(-1)          # (10, L, kv)
                union = 2 * actual_k - inter
                jac_vals = (inter / union.clamp(1e-12)).mean(dim=(1, 2))  # (10,)
                del temps, m_masks

                best_idx = int(jac_vals.argmax())
                if float(jac_vals[best_idx]) > base_j + 1e-6:
                    best_w = float(_GRID_NZERO[best_idx])

            if best_w > 1e-9:
                accum = accum + block * best_w
            w_cd_f1[g]  = best_w
            j_weight[g] = jaccard_scalar(accum)

        w_cd_f1_all[i] = w_cd_f1
        j_cd_f1_all[i] = j_weight
        del pf_kv, accum
        print(f"  [{i+1}/{N}] seq_len={S_i}  t={time.time()-t1:.1f}s  "
              f"j_cd_f1={j_weight[-1]:.3f}",
              flush=True)

    del pf_list; gc.collect()
    return {"w_cd_f1": w_cd_f1_all, "j_cd_f1": j_cd_f1_all}


def compute_single_query_tanimoto(path: str):
    """Per-chunk (non-accumulated) Tanimoto similarity.

    For each chunk g uses only that chunk's raw attention scores (no accumulation),
    normalises them, and computes Tanimoto vs oracle.

    Returns {"j_single_tanimoto": (N, G) float32}.
    """
    import gc, time
    cd_path = os.path.join(path, "coord_descent.npz")
    os_path = os.path.join(path, "oracle_score.npz")
    pf_path = os.path.join(path, "prefill_chunked.pt")

    cd     = np.load(cd_path)
    os_npz = np.load(os_path)
    print("    loading prefill_chunked.pt ...", flush=True)
    t0 = time.time()
    pf_list = torch.load(pf_path, map_location="cpu", weights_only=False)
    print(f"    loaded in {time.time()-t0:.1f}s", flush=True)

    G        = cd["w_cd"].shape[1]
    N, seq_lens = _resolve_N_seqlens(cd, os_npz, pf_list)
    kv = N_KV_HEADS; gs = GROUP_SIZE

    answer_score_all = os_npz["answer_score"]
    j_single = np.zeros((N, G), dtype=np.float32)

    for i in range(N):
        t1   = time.time()
        S_i  = seq_lens[i]
        pf_i = pf_list[i]
        G_i, L, n_heads, S_pf = pf_i.shape

        _pf_f32 = pf_i.float()
        pf_kv   = (_pf_f32.view(G_i, L, kv, gs, S_pf)
                          .sum(dim=3)[:, :, :, :S_i])   # (G, L, kv, S_i)
        del _pf_f32

        oracle_qh  = torch.as_tensor(answer_score_all[i, :, :, :S_i], dtype=torch.float32)
        oracle_kv  = oracle_qh.view(L, kv, gs, S_i).sum(dim=2)
        o_raw      = oracle_kv.clamp(min=0.0)
        oracle_norm = o_raw / (o_raw.sum(-1, keepdim=True) + 1e-12)

        for g in range(G):
            m_raw  = pf_kv[g].clamp(min=0.0)
            m_norm = m_raw / (m_raw.sum(-1, keepdim=True) + 1e-12)
            inter  = torch.minimum(oracle_norm, m_norm).sum(-1)
            union  = torch.maximum(oracle_norm, m_norm).sum(-1).clamp(1e-12)
            j_single[i, g] = float((inter / union).mean())

        del pf_kv
        print(f"  [{i+1}/{N}] seq_len={S_i}  t={time.time()-t1:.1f}s  "
              f"j_single={j_single[i,-1]:.3f}", flush=True)

    del pf_list; gc.collect()
    return {"j_single_tanimoto": j_single}


def extend_npz(path: str, extra: dict):
    """Merge extra arrays into existing coord_descent.npz and re-save."""
    cd_path = os.path.join(path, "coord_descent.npz")
    existing = dict(np.load(cd_path))
    existing.update(extra)
    np.savez_compressed(cd_path, **existing)
    print(f"  updated → {cd_path}")


# ── plotting helpers ──────────────────────────────────────────────────────────

_SCHEME_STYLE = {
    "single":   dict(color="gray",        lw=1.5, ls="--"),
    "optimal":  dict(color="black",       lw=2.0, ls="-"),
    "uniform":  dict(color="steelblue",   lw=1.5, ls="-"),
    "snap16":   dict(color="forestgreen", lw=1.5, ls="-"),
    "sigmoid":  dict(color="darkorange",  lw=1.5, ls="--"),
}


_ALL_SCHEMES = ["optimal", "uniform", "snap16", "sigmoid"]


def _auto_ylim(metric_key: str, pad: float = 0.05, custom_keys: dict = None,
               schemes: list = None):
    """Compute global ylim from all task data for `metric_key`."""
    if schemes is None:
        schemes = _ALL_SCHEMES
    all_vals = []
    for path, _ in TASKS:
        ds_path = os.path.join(BASE, path)
        cd = np.load(os.path.join(ds_path, "coord_descent.npz"))
        chunk = int(cd["chunk"]) if "chunk" in cd.files else 1
        G     = cd["w_cd"].shape[1]
        W     = G * chunk
        for scheme in schemes:
            if custom_keys and scheme in custom_keys:
                key = custom_keys[scheme]
            elif metric_key == "jaccard":
                key = "j_weight" if scheme == "optimal" else f"j_{scheme}"
            else:
                key = f"j_{scheme}_{metric_key}"
            if key not in cd.files:
                continue
            vals = np.repeat(cd[key].mean(axis=0), chunk)[:W]
            all_vals.extend(vals.tolist())
    if not all_vals:
        return (0.0, 1.0)
    lo, hi = min(all_vals), max(all_vals)
    span = hi - lo
    return (lo - pad * span, hi + pad * span)


def _make_figure(metric_key: str, ylabel: str, ylim, out_dir: str, fname: str,
                 custom_keys: dict = None, schemes: list = None):
    """Generate a 1×4 figure with running `metric_key` curves per task.

    Pass ylim=None to auto-compute from the data range with 5% padding.
    custom_keys: optional {scheme_name: npz_key} override (e.g. Tanimoto-optimal).
    schemes: ordered list of scheme names to plot (default: all four).
    """
    if schemes is None:
        schemes = _ALL_SCHEMES
    if ylim is None:
        ylim = _auto_ylim(metric_key, custom_keys=custom_keys, schemes=schemes)
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2))

    # Legend handles built from col=0
    h_proxy = []
    all_labels = []

    for col, (path, label) in enumerate(TASKS):
        ds_path = os.path.join(BASE, path)
        cd = np.load(os.path.join(ds_path, "coord_descent.npz"))
        m  = np.load(os.path.join(ds_path, "metrics.npz"))

        chunk = int(cd["chunk"]) if "chunk" in cd.files else 1
        G     = cd["w_cd"].shape[1]
        W     = G * chunk

        def get_curve(key):
            if key not in cd.files:
                return None
            raw = cd[key].mean(axis=0)   # (G,)
            return np.repeat(raw, chunk)[:W]

        ax = axes[col]
        d  = np.arange(W)
        handles_this = []

        # single-query Jaccard from metrics.npz (always present)
        if metric_key == "jaccard":
            br = np.repeat(m["br"].mean(axis=0), chunk)[:W]
            h, = ax.plot(d, br, label="single query", **_SCHEME_STYLE["single"])
            handles_this.append(("single query", h))

        for scheme in schemes:
            style_key = scheme
            if custom_keys and scheme in custom_keys:
                key = custom_keys[scheme]
            elif metric_key == "jaccard":
                key = "j_weight" if scheme == "optimal" else f"j_{scheme}"
            else:
                key = f"j_{scheme}_{metric_key}"
            curve = get_curve(key)
            if curve is None:
                continue
            lbl = {
                "single":  "Kronecker delta weighting",
                "optimal": "Optimal Weighting",
                "uniform": "Step Weighting",
                "snap16":  "SnapKV-16",
                "sigmoid": "Sigmoid Weighting",
            }[scheme]
            h, = ax.plot(d, curve, label=lbl, **_SCHEME_STYLE[style_key])
            handles_this.append((lbl, h))

        ax.set_xlim(0, W - 1)
        ax.set_ylim(*ylim)
        ax.invert_xaxis()
        if col == 0:
            ax.set_ylabel(ylabel)
            h_proxy    = [h for _, h in handles_this]
            all_labels = [lbl for lbl, _ in handles_this]
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.16, wspace=0.30)
    fig.text(0.525, 0.03,
             r"query distance $d$   ($\leftarrow$ older  $\cdot$  recent $\rightarrow$)",
             ha="center", fontsize=13)
    fig.legend(h_proxy, all_labels,
               loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=min(5, len(h_proxy)), frameon=False)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{fname}{SUFFIX}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_dir}/{fname}{SUFFIX}.{{pdf,png}}")


_TAN_CUSTOM_KEYS = {"optimal": "j_tan_optimal", "sigmoid": "j_tan_sigmoid"}


def _draw_band_row(axes_row, w_key: str, row_label: str, colors):
    """Fill one row of the obs2 band figure for the given weight key.

    Band = mean ± std (less noisy than min-max).
    Sigmoid fitted to the mean curve.
    """
    h_proxy = []
    for col, ((path, label), c) in enumerate(zip(TASKS, colors)):
        cd = np.load(os.path.join(BASE, path, "coord_descent.npz"))
        if w_key not in cd.files:
            axes_row[col].set_visible(False)
            continue
        w     = cd[w_key]                                  # (N, G)
        G     = w.shape[1]
        chunk = int(cd["chunk"]) if "chunk" in cd.files else 1
        W     = int(cd["window"]) if "window" in cd.files else G * chunk
        d     = np.arange(G) * chunk + (chunk - 1) / 2.0

        w_mean = w.mean(axis=0)
        w_std  = w.std(axis=0)
        lo     = np.clip(w_mean - w_std, 0.0, None)
        hi     = np.clip(w_mean + w_std, None, 1.0)

        # Sigmoid fit to mean
        try:
            from scipy.optimize import curve_fit as _cf
            def _sig(x, a, b):
                return 1.0 / (1.0 + np.exp(np.clip(a * (x - b), -30.0, 30.0)))
            popt, _ = _cf(_sig, d, w_mean,
                          p0=[0.05, float(W) / 4.0],
                          bounds=([0.0, 0.0], [5.0, float(W)]),
                          maxfev=5000)
            a_fit, b_fit = popt
            fit_y_disp = _sig(d, a_fit, b_fit)
            fit_label  = f"$a={a_fit:.2f},\\ b={b_fit:.1f}$"
        except Exception:
            a_fit = b_fit = float("nan")
            fit_y_disp = np.full_like(d, np.nan)
            fit_label  = ""

        ax = axes_row[col]
        lb = ax.fill_between(d, lo, hi, color=c, alpha=0.30,
                             linewidth=0, label="mean ± std")
        lm, = ax.plot(d, w_mean, color=c, lw=2.0, label="mean")
        lf, = ax.plot(d, fit_y_disp, "--", color="black", lw=1.5,
                      label="sigmoid fit")

        if np.isfinite(a_fit):
            ax.text(0.97, 0.95, fit_label,
                    transform=ax.transAxes, fontsize=9, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor="0.8", alpha=0.85))

        ax.set_xlim(0, W - 1)
        ax.invert_xaxis()
        ax.set_ylim(-0.05, 1.10)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel(f"{row_label}\n$w_d$", fontsize=11)
            h_proxy = [lb, lm, lf]
    return h_proxy


def plot_obs2_coefficient_band(out_dir: str):
    """1-row × 4-col figure: Tanimoto-optimal coefficient band per task."""
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8),
                             sharex=False, sharey=False)

    for col, (_, label) in enumerate(TASKS):
        axes[col].set_title(label)

    h_proxy = _draw_band_row(axes, "w_tan", "optimal $w_d$", colors)

    fig.text(0.525, 0.02,
             r"query distance $d$   ($\leftarrow$ older  $\cdot$  recent $\rightarrow$)",
             ha="center", fontsize=12)

    if h_proxy:
        fig.legend(h_proxy, ["mean ± std", "mean", "sigmoid fit"],
                   loc="upper center", bbox_to_anchor=(0.5, 0.99),
                   ncol=3, frameon=False)

    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.16, wspace=0.28)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"obs1_sigmoid_band{SUFFIX}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_dir}/obs1_sigmoid_band{SUFFIX}.{{pdf,png}}")


_TAN_SCHEMES = ["single", "optimal", "uniform", "sigmoid"]


def main():
    import sys
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Flags:
    #   --tan-only    : skip ranked/Jaccard alt metrics pass (already done)
    #   --f1-only     : also skip alt metrics; run only g0-forced CD + obs2
    #   --single-only : skip all CD; compute single-query Tanimoto + regen figures
    tan_only    = "--tan-only"    in sys.argv
    f1_only     = "--f1-only"     in sys.argv
    single_only = "--single-only" in sys.argv

    if not (tan_only or f1_only or single_only):
        for path, label in TASKS:
            ds_path = os.path.join(BASE, path)
            print(f"\n=== {label} ({path}) ===")
            extra = compute_alt_metrics(ds_path, BUDGET)
            extend_npz(ds_path, extra)

    if not single_only:
        print("\n--- Jaccard-optimal coord descent (from prefill, g=0 forced to 1.0) ---")
        for path, label in TASKS:
            ds_path = os.path.join(BASE, path)
            print(f"\n=== {label} ({path}) ===")
            extra = compute_jaccard_optimal_from_prefill(ds_path, BUDGET)
            extend_npz(ds_path, extra)

        print("\n--- Tanimoto-optimal coord descent (g=0 forced to 1.0) ---")
        for path, label in TASKS:
            ds_path = os.path.join(BASE, path)
            print(f"\n=== {label} ({path}) ===")
            extra = compute_tanimoto_optimal(ds_path)
            extend_npz(ds_path, extra)

    print("\n--- Single-query Tanimoto (per chunk, no accumulation) ---")
    for path, label in TASKS:
        ds_path = os.path.join(BASE, path)
        print(f"\n=== {label} ({path}) ===")
        extra = compute_single_query_tanimoto(ds_path)
        extend_npz(ds_path, extra)

    print("\n--- Generating figures ---")
    if not (tan_only or f1_only or single_only):
        _make_figure("jaccard",  "Jaccard similarity",
                     (0.0, 0.45), out_dir, "obs1_jaccard_recovery")
        _make_figure("ranked",   "Ranked similarity (DCG-norm)",
                     (0.0, 1.0),  out_dir, "obs1_ranked_recovery")
    _make_figure("tanimoto", "Tanimoto similarity",
                 None, out_dir, "obs1_jaccard_recovery",
                 custom_keys=_TAN_CUSTOM_KEYS,
                 schemes=_TAN_SCHEMES)
    plot_obs2_coefficient_band(out_dir)

    print(f"\nDone. Outputs → {out_dir}/obs1_{{jaccard,ranked,tanimoto}}_recovery{SUFFIX}.pdf")


if __name__ == "__main__":
    main()
