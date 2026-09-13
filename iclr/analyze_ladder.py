"""Ladder result analysis: 4-row accuracy table + head-choice structure.

Usage: python iclr/analyze_ladder.py --tag feas1b --model llama3-1b
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

ROWS = ["fixed", "oracle", "shuffled", "agent"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="feas1b")
    ap.add_argument("--model", default="llama3-1b")
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print(f"== {args.tag} accuracy ladder ==")
    per_row = {}
    for row in ROWS:
        p = f"result_txt/pred/128/{args.tag}_{row}/result.json"
        if not os.path.exists(p):
            print(f"  {row:9s}: (incomplete)")
            continue
        r = json.load(open(p))
        per_row[row] = r
        print(f"  {row:9s}: overall {r['overall_average']:.2f}")
    if "fixed" in per_row:
        base = per_row["fixed"]["individual_scores"]
        for row in ["oracle", "shuffled", "agent"]:
            if row not in per_row:
                continue
            diffs = {d: per_row[row]["individual_scores"][d] - base[d]
                     for d in base if d in per_row[row]["individual_scores"]}
            top = sorted(diffs.items(), key=lambda x: -abs(x[1]))[:5]
            print(f"  Δ({row}-fixed) top: " +
                  ", ".join(f"{d} {v:+.1f}" for d, v in top))

    # head-choice structure from pass-1 tables
    cache_root = f"{os.environ.get('ICLR_TRACES', '/data2/smp9898/iclr_traces')}/lb_pass1/{args.model}"
    for row in ["oracle", "agent"]:
        files = glob.glob(os.path.join(cache_root, "*", f"s*_{row}.json"))
        if not files:
            continue
        cnt, per_head_mode = Counter(), None
        choices = []
        for f in files:
            ch = np.array(json.load(open(f))["choice"])
            choices.append(ch)
            cnt.update(ch.flatten().tolist())
        ch_all = np.stack(choices)                     # [S, L, H]
        consist = np.array([[Counter(ch_all[:, l, h]).most_common(1)[0][1] / len(ch_all)
                             for h in range(ch_all.shape[2])]
                            for l in range(ch_all.shape[1])])
        total = sum(cnt.values())
        top_actions = ", ".join(f"a{k}:{v*100//total}%" for k, v in cnt.most_common(5))
        print(f"  [{row}] {len(files)} samples | action mix {top_actions} | "
              f"mean cross-sample head-selection consistency {consist.mean():.2f}")


if __name__ == "__main__":
    main()
