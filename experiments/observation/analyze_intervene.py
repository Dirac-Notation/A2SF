"""Score the intervention runs of intervene.py (dataset-equal LongBench average, bootstrap CIs resampled within
each dataset, 4,000 resamples, seed 0).

    python experiments/observation/analyze_intervene.py --design quartile   <dir>    # Figure 1b, Section 2.2
    python experiments/observation/analyze_intervene.py --design percentile <dir>    # Section 2.4 (top 10% vs s)
    python experiments/observation/analyze_intervene.py --design weights    <dir>    # Section 2.4 (s vs iota oracle)
"""
import argparse, difflib, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common import ROOT
import longbench_eval as E

TASK = json.load(open(os.path.join(ROOT, "config/task2dataset.json")))
D2T = {d: t for t, ds in TASK.items() for d in ds}


def score(rec, arm):
    d, p = rec["dataset"], rec["preds"][arm]
    if d in ("trec", "triviaqa", "samsum", "lsht"):
        p = p.lstrip("\n").split("\n")[0]
    return 100 * max(E.dataset2metric[d](p, g, all_classes=rec.get("all_classes")) for g in rec["answers"])


class Stats:
    def __init__(self, R, arms):
        self.S = {a: np.array([score(r, a) for r in R]) for a in arms}
        self.ds = np.array([r["dataset"] for r in R])
        self.keys = sorted(set(self.ds))
        self.rng = np.random.default_rng(0)

    def mean(self, v, excl=()):
        return float(np.mean([v[self.ds == d].mean() for d in self.keys if d not in excl]))

    def boot(self, x, y, n=4000, excl=(), rng=None):
        rng = rng or self.rng
        d, ds = x - y, self.ds
        out = []
        for _ in range(n):
            out.append(np.mean([d[np.where(ds == k)[0]][rng.integers(0, int((ds == k).sum()), int((ds == k).sum()))].mean()
                                for k in self.keys if k not in excl]))
        return np.quantile(out, [.025, .975])


def quartile(R):
    arms = ["q1", "q2", "q3", "q4", "random", "all"]
    lab = {"q1": "iota top 25%", "q2": "iota 25-50%", "q3": "iota 50-75%", "q4": "iota bottom 25%",
           "random": "random 25%", "all": "all document queries (H2O)"}
    st = Stats(R, arms); S = st.S
    F = {a: np.array([difflib.SequenceMatcher(None, r["preds"][a], r["fc"]).ratio() for r in R]) for a in arms}
    print(f"{len(R)} prompts, {len(st.keys)} tasks, 25% of document queries (median k {int(np.median([r['k'] for r in R]))})")
    print(f"\n{'arm':28s} {'LongBench':>9s} {'vs random':>10s} {'95% CI':>18s} {'sim. to full':>12s}")
    for a in arms:
        lo, hi = st.boot(S[a], S["random"])
        print(f"{lab[a]:28s} {st.mean(S[a]):9.2f} {st.mean(S[a]) - st.mean(S['random']):+10.2f} "
              f"[{lo:+6.2f}, {hi:+6.2f}] {st.mean(F[a]):12.3f}")
    for x, y in [("q1", "q4"), ("q4", "random"), ("q1", "all"), ("q1", "random")]:
        lo, hi = st.boot(S[x], S[y]); lo2, hi2 = st.boot(S[x], S[y], excl=("trec",))
        print(f"\n{lab[x]} - {lab[y]} = {st.mean(S[x]) - st.mean(S[y]):+.2f} [{lo:+.2f}, {hi:+.2f}]"
              f"   without trec {st.mean(S[x], ('trec',)) - st.mean(S[y], ('trec',)):+.2f} [{lo2:+.2f}, {hi2:+.2f}]")
    print("\nper task group (top 25% - bottom 25%)")
    for t in TASK:
        m = np.array([D2T.get(d) == t for d in st.ds])
        if m.sum() >= 10:
            print(f"  {t:18s} {S['q1'][m].mean() - S['q4'][m].mean():+6.2f}  (n={int(m.sum())})")
    print("\nper dataset (" + " / ".join(lab[a] for a in arms) + ")")
    for d in st.keys:
        m = st.ds == d
        print(f"  {d:22s} " + " ".join(f"{S[a][m].mean():6.2f}" for a in arms))


def percentile(R):
    arms = ["top10", "q10_25", "q25_50", "q50_75", "bottom10", "snap10", "random10", "dense"]
    lab = {"top10": "iota top 10%", "q10_25": "iota 10-25%", "q25_50": "iota 25-50%", "q50_75": "iota 50-75%",
           "bottom10": "iota bottom 10%", "snap10": "s top 10%", "random10": "random 10%", "dense": "no restriction (s only)"}
    st = Stats(R, arms); S = st.S
    print(f"{len(R)} prompts, {len(st.keys)} tasks, 10% of document queries (median k {int(np.median([r['k'] for r in R]))})")
    print(f"\n{'arm':26s} {'LongBench':>9s} {'w/o trec':>9s} {'vs random':>10s} {'95% CI':>18s}")
    for a in arms:
        lo, hi = st.boot(S[a], S["random10"])
        print(f"{lab[a]:26s} {st.mean(S[a]):9.2f} {st.mean(S[a], ('trec',)):9.2f} "
              f"{st.mean(S[a]) - st.mean(S['random10']):+10.2f} [{lo:+.2f}, {hi:+.2f}]")
    for x, y in [("top10", "bottom10"), ("snap10", "random10"), ("top10", "snap10")]:
        lo, hi = st.boot(S[x], S[y])
        print(f"  {lab[x]} - {lab[y]} = {st.mean(S[x]) - st.mean(S[y]):+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
    lo, hi = st.boot(S["top10"], S["dense"], rng=np.random.default_rng(0))
    print(f"  {lab['top10']} - {lab['dense']} = {st.mean(S['top10']) - st.mean(S['dense']):+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")


def weights(R):
    lab = {"iota": "iota-proportional (oracle)", "s": "s-proportional (Q-Trust sum)", "snapkv": "SnapKV", "h2o": "H2O"}
    arms = [a for a in lab if all(a in r["preds"] for r in R)]
    st = Stats(R, arms); S = st.S
    print(f"{len(R)} prompts, {len(st.keys)} tasks (dataset-equal average)")
    for a in arms:
        lo, hi = st.boot(S[a], S["s"])
        print(f"  {lab[a]:30s} {st.mean(S[a]):6.2f}   vs s {st.mean(S[a]) - st.mean(S['s']):+6.2f} [{lo:+.2f}, {hi:+.2f}]")
    if {"iota", "h2o", "s", "snapkv"} <= set(arms):
        gap = st.mean(S["iota"]) - st.mean(S["h2o"])
        print(f"  share of the H2O -> iota-oracle gap closed: s {(st.mean(S['s']) - st.mean(S['h2o'])) / gap:.2f}, "
              f"SnapKV {(st.mean(S['snapkv']) - st.mean(S['h2o'])) / gap:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["quartile", "percentile", "weights"], required=True)
    ap.add_argument("dir")
    a = ap.parse_args()
    R = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.dir, "*.json")))]
    {"quartile": quartile, "percentile": percentile, "weights": weights}[a.design](R)
