"""Text-level comparison of v5 vs 4.46.2-backup predictions (same SnapKV-16 config).

Aligns per-sample by (answers, all_classes, length) — a generation-independent key
from the source dataset — then reports, per dataset: exact-match rate of the pred
strings + mean difflib similarity. Tells us whether v5 reproduces the 4.46.2 text,
not just the aggregate score."""
import os, json, sys
from collections import defaultdict
from difflib import SequenceMatcher

# usage: compare_v5_vs_446_preds.py [v5_dir] [backup_dir]
V5 = sys.argv[1] if len(sys.argv) > 1 else "result_txt/pred/128/llama3-1b_snap_16_128"
BK = sys.argv[2] if len(sys.argv) > 2 else "result_txt/backup/llama3-1b/128/llama3-1b_SnapKV-16_128"
print(f"v5={V5}\nbackup={BK}\n")


def load(path):
    by_key = defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            key = (str(d.get("answers")), str(d.get("all_classes")), str(d.get("length")))
            by_key[key].append(d.get("pred", ""))
    return by_key


def pair_preds(bk, v5):
    """Yield (bk_pred, v5_pred) pairs aligned within each key group."""
    pairs = []
    for key, bks in bk.items():
        v5s = list(v5.get(key, []))
        if not v5s:
            continue
        bks = list(bks)
        # match exact-identical first, then pair leftovers by order
        bk_rem, v5_rem = [], list(v5s)
        for p in bks:
            if p in v5_rem:
                v5_rem.remove(p); pairs.append((p, p))
            else:
                bk_rem.append(p)
        for a, b in zip(bk_rem, v5_rem):
            pairs.append((a, b))
    return pairs


datasets = sorted(f[:-6] for f in os.listdir(V5) if f.endswith(".jsonl"))
tot_n = tot_exact = 0
sim_sum = 0.0
print(f"{'dataset':22s} {'n':>4} {'exact%':>7} {'mean_sim':>9} {'sim>=0.9%':>9}")
for ds in datasets:
    pv, pb = os.path.join(V5, ds + ".jsonl"), os.path.join(BK, ds + ".jsonl")
    if not os.path.exists(pb):
        continue
    pairs = pair_preds(load(pb), load(pv))
    if not pairs:
        continue
    n = len(pairs)
    exact = sum(1 for a, b in pairs if a == b)
    sims = [SequenceMatcher(None, a, b).ratio() for a, b in pairs]
    hi = sum(1 for s in sims if s >= 0.9)
    msim = sum(sims) / n
    print(f"{ds:22s} {n:>4} {100*exact/n:>6.1f}% {msim:>9.3f} {100*hi/n:>8.1f}%")
    tot_n += n; tot_exact += exact; sim_sum += sum(sims)

print("-" * 56)
print(f"{'OVERALL':22s} {tot_n:>4} {100*tot_exact/tot_n:>6.1f}% {sim_sum/tot_n:>9.3f}")
