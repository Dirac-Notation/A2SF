"""Section 2.2 / 2.3 statistics from capture_iota.py outputs (informativeness distribution, where informative
queries are, and how much weight each method puts on them).

    python experiments/observation/analyze_iota.py runs/iota_llama3-8b
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT

TASK = json.load(open(os.path.join(ROOT, "config/task2dataset.json"))); D2T = {d: t for t, ds in TASK.items() for d in ds}
R = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json")))]
P = [p for r in R for p in r["pairs"]]
m = lambda f, rows=None: float(np.nanmean([f(p) for p in (rows or P)]))
print(f"{len(R)} prompts, {len(P)} (layer, KV head) pairs, median N {int(np.median([r['N'] for r in R]))}")

print("\n[distribution of iota over document queries]")
for k, lab in [("bottom10", "bottom 10%"), ("median", "median"), ("top25", "top 25%"), ("top10", "top 10%"), ("top1", "top 1%")]:
    print(f"  {lab:10s} {m(lambda p: p['iota'][k]):+.3f}")
print(f"  shuffled rows (null) {m(lambda p: p['null_shuffled']):+.3f}   share iota>0 {m(lambda p: p['iota']['pos_frac']):.3f}")
print("  by task group (median / top 25% / top 1%)")
for t in TASK:
    rr = [p for r in R if D2T.get(r["dataset"]) == t for p in r["pairs"]]
    if rr:
        print(f"    {t:18s} {m(lambda p: p['iota']['median'], rr):+.3f} / {m(lambda p: p['iota']['top25'], rr):+.3f} / "
              f"{m(lambda p: p['iota']['top1'], rr):+.3f}")

print("\n[where: share by distance from the end of the prompt]")
e = P[0]["dist"]["edges"]
print(f"  {'distance':>12s} {'top25%':>8s} {'top10%':>8s} {'top1%':>8s} {'positions':>9s}")
for j in range(len(e) - 1):
    lab = f"{e[j]}-{e[j+1] if e[j+1] < 10**8 else 'inf'}"
    ps = m(lambda p: p["dist"]["pos_share"][j])
    if ps > 0:
        print(f"  {lab:>12s} {m(lambda p: p['dist']['top25_share'][j]):8.3f} {m(lambda p: p['dist']['top10_share'][j]):8.3f} "
              f"{m(lambda p: p['dist']['top1_share'][j]):8.3f} {ps:9.3f}")

print("\n[weight each method puts on informative queries (share of its document-query weight)]")
print(f"  {'method':8s} {'E_w[iota]':>9s} {'top25 share':>11s} {'lift':>6s} {'top10 lift':>10s}")
for me in ["H2O", "Q-Trust", "oracle"]:
    sh25 = m(lambda p: p["methods"][me]["top25_mass"] / max(p["methods"][me]["doc_mass"], 1e-30))
    sh10 = m(lambda p: p["methods"][me]["top10_mass"] / max(p["methods"][me]["doc_mass"], 1e-30))
    print(f"  {me:8s} {m(lambda p: p['methods'][me]['Ew']):+9.3f} {sh25:11.3f} {sh25/0.25:6.2f} {sh10/0.10:10.2f}")
print(f"\n  Spearman(s, iota) {m(lambda p: p['link']['spearman']):.3f}")
