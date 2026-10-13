"""Figure 1: informative queries exist (long tail), they decide accuracy, and they are found at every distance.

    python experiments/observation/fig_infoquery.py --iota runs/iota_llama3-8b --quartile runs/intervene_quartile --out fig_infoquery.pdf
(a), (c) from capture_iota.py; (b) from intervene.py --design quartile. Also prints the Section 2.3 shares.
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42          # embed TrueType, not Type 3 (venue font checks)
plt.rcParams["ps.fonttype"] = 42
import longbench_eval as E2

ap = argparse.ArgumentParser()
ap.add_argument("--iota", required=True); ap.add_argument("--quartile", required=True); ap.add_argument("--out", default="fig_infoquery.pdf")
A_ = ap.parse_args()
HLO, HHI, HBINS = -0.30, 0.90, 240
R = [json.load(open(p)) for p in sorted(glob.glob(f"{A_.iota}/*.json"))]
P = [p for r in R for p in r["pairs"]]
BLUE, GREY, RED = "#2b6cb0", "#a0aec0", "#c05621"
plt.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9})
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2), dpi=100, gridspec_kw=dict(width_ratios=[1.3, 1, 1]))

# (a) long tail as a rank plot: document queries sorted by informativeness (highest first), pooled over
# every prompt, layer and KV head. The quantile function comes from the stored histograms, linearly
# interpolated inside each 0.005-wide bin so the flat part near zero does not show as steps.
a = ax[0]
h = np.sum([p["hist"] for p in P], axis=0).astype(float)
edges = np.linspace(HLO, HHI, HBINS + 1)
cdfE = np.concatenate([[0.0], np.cumsum(h / h.sum())])
rank = np.linspace(0.0, 1.0, 4001)                      # 0 = most informative
val = np.interp(1.0 - rank, cdfE, edges)
QCOL = [BLUE, "#8fa9c7", "#b3bfcf", GREY]           # same colours as the quartile bars in (b)
x = 100 * rank
for k in range(4):
    m = (rank >= k * 0.25) & (rank <= (k + 1) * 0.25)
    a.fill_between(x[m], val[m], 0, color=QCOL[k], alpha=0.9, linewidth=0)
a.plot(x, val, color="#0b2447", lw=1.4)
a.axhline(0, color="#444", lw=0.8)
for q in (25, 50, 75):
    a.axvline(q, color="#555", ls="--", lw=0.9)
top = 0.8
a.set_ylim(min(val) - 0.02, top)          # the whole range: the positive tail reaches about twice as far
for k, lab in enumerate(["top", "25-50%", "50-75%", "bottom"]):
    a.text(12.5 + 25 * k, top * 0.9, lab, ha="center", fontsize=7.5, color="#111")
a.set_xlim(0, 100)
a.set_xticks([0, 25, 50, 75, 100])
a.set_xlabel("document queries, sorted (%)", fontsize=8.5)
a.set_ylabel("informativeness", fontsize=8.5)
a.set_title("(a) long tail of informativeness", fontsize=9)
q25, q50, q75 = (np.interp(t, cdfE, edges) for t in (0.25, 0.50, 0.75))

# (b) intervention: keep only one informativeness quartile of the document queries in the sum
a = ax[1]
T2 = [json.load(open(p)) for p in sorted(glob.glob(f"{A_.quartile}/*.json"))]
def sc(r, arm):
    d, p_ = r["dataset"], r["preds"][arm]
    if d in ("trec", "triviaqa", "samsum", "lsht"):
        p_ = p_.lstrip("\n").split("\n")[0]
    return 100 * max(E2.dataset2metric[d](p_, g, all_classes=r.get("all_classes")) for g in r["answers"])
dsv = np.array([r["dataset"] for r in T2])
def eq(arm):
    v = np.array([sc(r, arm) for r in T2])
    return float(np.mean([v[dsv == d].mean() for d in sorted(set(dsv))]))
ys = [eq(b) for b in ["q1", "q2", "q3", "q4"]]
a.bar(range(4), ys, color=[BLUE, "#8fa9c7", "#b3bfcf", GREY], width=0.65)
lo = min(ys) - 3
for ref, lab, st in [("random", "random 25%", "--"), ("all", "all queries (H2O)", ":")]:
    y = eq(ref)
    a.axhline(y, color="#444", ls=st, lw=1)
    a.text(3.45, y + .3, lab, fontsize=7.5, ha="right", color="#111")
for i, v in enumerate(ys):
    a.text(i, v + .3, f"{v:.1f}", ha="center", fontsize=7.5)
a.set_ylim(lo, max(ys) + 2.5)
a.set_xticks(range(4)); a.set_xticklabels(["top\n25%", "25-\n50%", "50-\n75%", "bottom\n25%"], fontsize=7.5)
a.set_xlabel("queries kept in the sum", fontsize=8.5)
a.set_ylabel("sampled LongBench", fontsize=8.5)
a.set_title("(b) informative help, noisy hurt", fontsize=9)

# (c) at each distance from the prompt end, 16-token bins up to 2K and one 2K+ bin: the share of the bin's
# document queries that are informative (top 25%). Counts are pooled over prompts, layers and KV heads before
# dividing; the document-wide mean is 25% by definition.
a = ax[2]
RR = R
K, WB = 129, 16
inf, allq = np.zeros(K), np.zeros(K)
for r in RR:
    nd = r["N"] - 20
    for p in r["pairs"]:
        e = p["dist_eq"]["abs16"]
        inf += np.array(e["top25"]) * (nd // 4); allq += np.array(e["pos"]) * nd
ratio = 100 * inf / np.maximum(allq, 1e-9)
xs = (np.arange(1, K - 1) + 0.5) * WB                   # bins 1..127 (bin 0 = the window, not document queries)
# 128-token groups of the 16-token bins (the first group is 16-128: 0-16 is the window, not document queries)
G = 8
gx = np.array([(g * G * WB + (g + 1) * G * WB) / 2 for g in range(16)])
gy = np.array([100 * inf[max(1, g * G):(g + 1) * G].sum() / allq[max(1, g * G):(g + 1) * G].sum() for g in range(16)])
gx[0] = (16 + 128) / 2
a.plot(gx, gy, "-o", color=BLUE, lw=1.8, ms=3.5)
raw = gy
a.set_xlim(0, 2048)
a.set_xticks([0, 512, 1024, 1536, 2048]); a.set_xticklabels(["0", "512", "1K", "1.5K", "2K"], fontsize=8)
a.set_ylim(0, max(raw) * 1.12)
a.set_xlabel("distance from end (tokens)", fontsize=8.5)
a.set_ylabel("informative queries (%)", fontsize=8.5)
a.set_title("(c) share by distance", fontsize=9)
print("16-64", round(100 * inf[1:4].sum() / allq[1:4].sum(), 1), "| 1-2K", round(100 * inf[64:128].sum() / allq[64:128].sum(), 1),
      "| 2K+", round(ratio[K - 1], 1), "| 2K+ share of queries", round(100 * allq[K - 1] / allq.sum(), 1),
      "| 2K+ share of informative", round(100 * inf[K - 1] / inf.sum(), 1))

fig.tight_layout(pad=0.3, w_pad=0.8); fig.savefig(A_.out)
print("informative queries more than 64 tokens from the end", round(100 * inf[4:].sum() / inf.sum(), 1), "%")
print(f"saved   pooled quartiles {q25:+.3f} / {q50:+.3f} / {q75:+.3f}, max plotted {val.max():+.3f}")
