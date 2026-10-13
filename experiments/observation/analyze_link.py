"""Table 1 and the Section 2.2 / 2.4 link statistics from capture_link.py outputs (one directory per model).

    python experiments/observation/analyze_link.py runs/link_llama3-8b runs/link_qwen2 runs/link_mistral-7b-v03

Table 1 = section [1]: "lift1" (mass on the top 1% of iota, as a multiple of uniform) and "top25 share",
rows "s (W=16)", "H2O transfer" (accumulated attention) and "recency 1/d" (distance).
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT

TASK = json.load(open(os.path.join(ROOT, "config/task2dataset.json"))); D2T = {d: t for t, ds in TASK.items() for d in ds}


def mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def report(path):
    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(path, "*.json")))]
    P = [p for r in R for p in r["pairs"]]
    print(f"\n==================== {path}: {len(R)} prompts, {len(P)} (layer, KV head) pairs")
    names = list(P[0]["rivals"])
    print("\n[1] query weighting vs informative queries (lift = multiple of uniform) / keys kept by its score")
    print(f"  {'weighting':16s} {'top1%':>7s} {'top10%':>7s} {'top25 share':>11s} {'E_w[iota]':>9s} {'Spear':>6s} | "
          f"{'recall':>6s} {'u mass':>6s} {'far':>6s} {'ESS':>7s}")
    for w in names:
        g = lambda k: mean([p["rivals"][w][k] for p in P])
        kr = [p["keyrec"].get(w) for p in P if "keyrec" in p and w in p["keyrec"]]
        k = lambda x: mean([q[x] for q in kr if q]) if kr else float("nan")
        print(f"  {w:16s} {g('lift1'):7.2f} {g('lift10'):7.2f} {g('lift25') / 4:11.3f} {g('Ew'):+9.3f} {g('spearman'):6.3f} | "
              f"{k('recall'):6.3f} {k('umass'):6.3f} {k('far_recall'):6.3f} {k('ess'):7.0f}")
    for w in ("SnapKV", "Q-Trust (actual)", "H2O (actual)", "iota oracle"):
        kr = [p["keyrec"][w] for p in P if "keyrec" in p and w in p["keyrec"]]
        if kr:
            print(f"  {w:16s} {'':>45s} | {mean([q['recall'] for q in kr]):6.3f} {mean([q['umass'] for q in kr]):6.3f} "
                  f"{mean([q['far_recall'] for q in kr]):6.3f}  (n={len(kr)})")

    print("\n[2] informative share (%) in 20 bins of the s percentile (lowest s first)")
    print("  " + " ".join(f"{100 * x:4.1f}" for x in np.nanmean([p["curve"]["informative"] for p in P], 0)))

    C = [p for p in P if "ceiling" in p]
    if C:
        print(f"\n[3] noise ceiling (even / odd generated tokens): Spearman(iota_a, iota_b) {mean([p['ceiling']['spearman_ab'] for p in C]):.3f}")
        for k in ("lift1", "lift10", "lift25"):
            a, s = mean([p["ceiling"]["iota_a"][k] for p in C]), mean([p["ceiling"]["s"][k] for p in C])
            print(f"  {k:7s} ceiling {a:6.2f}  s {s:6.2f}  s relative to ceiling {(s - 1) / (a - 1):.2f}")
    M = [p for p in P if "mech" in p]
    if M:
        print(f"\n[4] mechanism: Spearman(s, rho) {mean([p['mech']['sp_s_rho'] for p in M]):.3f}, "
              f"Spearman(rho, iota) {mean([p['mech']['sp_rho_iota'] for p in M]):.3f}, "
              f"Spearman(s, iota) {mean([p['mech']['sp_s_iota'] for p in M]):.3f}")
    L = [p for p in P if "nonlocal" in p]
    if L:
        print(f"\n[5] locality control (keys within +-32 tokens of the query removed), "
              f"Spearman(iota, iota_nl) {mean([p['nonlocal']['sp_iota_nl'] for p in L]):.3f}")
        for w in ("s (W=16)", "recency 1/d", "H2O transfer", "uniform"):
            print(f"  {w:14s} lift1 {mean([p['nonlocal'][w]['lift1'] for p in L]):6.2f}  lift10 {mean([p['nonlocal'][w]['lift10'] for p in L]):5.2f}  "
                  f"top25 share {mean([p['nonlocal'][w]['lift25'] for p in L]) / 4:.3f}")
    V = [p for p in P if "variants" in p]
    if V:
        print("\n[6] s under other iota definitions")
        for k in ("iota_raw", "iota_glob"):
            print(f"  {k:10s} lift1 {mean([p['variants'][k]['lift1'] for p in V]):6.2f}  lift10 {mean([p['variants'][k]['lift10'] for p in V]):5.2f}  "
                  f"top25 share {mean([p['variants'][k]['lift25'] for p in V]) / 4:.3f}   "
                  f"(uniform {mean([p['variants']['iota_raw_uniform']['lift25'] for p in V]) / 4:.3f})")
    print("\n[7] top25 share of s by task group")
    for t in TASK:
        rr = [p for r in R if D2T.get(r["dataset"]) == t for p in r["pairs"]]
        if rr:
            print(f"  {t:18s} {mean([p['rivals']['s (W=16)']['lift25'] for p in rr]) / 4:.3f}")
    W = [p for p in P if "rawcos" in p]
    if W:
        print(f"\n[8] raw cosine with u: mean document row {mean([p['rawcos']['meanrow'] for p in W]):.3f}, "
              f"top 1% of rows {mean([p['rawcos']['top1'] for p in W]):.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        report(d)
