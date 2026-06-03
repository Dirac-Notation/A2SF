"""Q3 rebuttal: Multi-seed stability of the agent.

Three seeds:
  seed=42 (already-trained exp_pre_rope_mean)
  seed=1  (fresh)
  seed=2  (fresh)

Each evaluated via fast_lb_eval → result.json with per-dataset / group scores.

Output:
  experiments/paper_figures/rebuttal/q3_multiseed_boxplot.{pdf,png}
  experiments/paper_figures/rebuttal/q3_summary.json
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 10, "ytick.labelsize": 11,
    "legend.fontsize": 10, "axes.linewidth": 1.0,
    "figure.dpi": 150,
})

ROOT    = "/home/smp9898/A2SF"
OUT_DIR = f"{ROOT}/experiments/paper_figures/rebuttal"

OUR_RUNS = [
    ("seed=42", f"{ROOT}/result_txt/pred/128/exp_pre_rope_mean_fast"),
    ("seed=1",  f"{ROOT}/result_txt/pred/128/exp_pre_rope_mean_s1_fast"),
    ("seed=2",  f"{ROOT}/result_txt/pred/128/exp_pre_rope_mean_s2_fast"),
]

BASELINES = [
    ("SnapKV-16", f"{ROOT}/result_txt/backup/llama3-1b/128/llama3-1b_SnapKV-16_128"),
    ("SnapKV-32", f"{ROOT}/result_txt/backup/llama3-1b/128/llama3-1b_SnapKV-32_128"),
    ("TOVA",      f"{ROOT}/result_txt/backup/llama3-1b/128/llama3-1b_TOVA_128"),
    ("H2O",       f"{ROOT}/result_txt/backup/llama3-1b/128/llama3-1b_H2O_128"),
]


def load_result(run_dir):
    p = f"{run_dir}/result.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    seeds_data = {}
    for label, run in OUR_RUNS:
        r = load_result(run)
        if r is None:
            print(f"  [missing] {label}: {run}")
            continue
        seeds_data[label] = r

    base_data = {}
    for label, run in BASELINES:
        r = load_result(run)
        if r is None:
            print(f"  [missing baseline] {label}")
            continue
        base_data[label] = r

    if len(seeds_data) < 2:
        print("ERROR: need at least 2 seeds"); return

    overall = np.array([seeds_data[s]["overall_average"] for s in seeds_data])
    overall_mean = float(overall.mean())
    overall_std  = float(overall.std(ddof=1)) if len(overall) > 1 else 0.0

    # 95% CI via bootstrap (or t-CI for n=3)
    if len(overall) >= 3:
        ci_t = stats.t.interval(0.95, len(overall)-1, loc=overall_mean,
                                scale=overall_std / np.sqrt(len(overall)))
        ci_lo, ci_hi = float(ci_t[0]), float(ci_t[1])
    else:
        ci_lo = ci_hi = float("nan")

    # Per-task: each seed contributes per-dataset scores; group by task
    with open(f"{ROOT}/config/task2dataset.json") as f:
        t2d = json.load(f)
    ds2task = {ds: t for t, dss in t2d.items() for ds in dss}
    tasks = list(t2d.keys())

    # per-task values across seeds (each seed contributes one task-group-avg)
    per_task = {t: [] for t in tasks}
    for s in seeds_data:
        for t in tasks:
            v = seeds_data[s]["group_averages"].get(t)
            if v is not None:
                per_task[t].append(v)

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.0))
    labels = [t for t in tasks if per_task[t]]
    box_data = [np.array(per_task[t]) for t in labels]
    bp = ax.boxplot(box_data, positions=range(len(labels)), widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.4))
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue"); patch.set_alpha(0.55)

    # Show individual seed points
    rng = np.random.default_rng(0)
    for i, t in enumerate(labels):
        for v in per_task[t]:
            ax.scatter(i + rng.uniform(-0.15, 0.15), v,
                       color="navy", s=20, alpha=0.7, zorder=5)

    # Overlay baselines as colored markers
    base_colors = {"SnapKV-16": "red", "SnapKV-32": "darkorange",
                   "TOVA": "magenta", "H2O": "gray"}
    for bname, br in base_data.items():
        ga = br["group_averages"]
        ys = [ga.get(t, np.nan) for t in labels]
        ax.scatter(range(len(labels)), ys, marker="D", s=65,
                   c=base_colors.get(bname, "black"), label=bname,
                   zorder=10, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=10)
    ax.set_ylabel("LB128 task-group score")
    ax.set_title(f"Multi-seed performance vs. baselines  "
                 f"(3 seeds; overall = {overall_mean:.2f} ± {overall_std:.2f})")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_pdf = f"{OUT_DIR}/q3_multiseed_boxplot.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_pdf}")

    # Paired Wilcoxon vs SnapKV-16
    wp = float("nan"); mean_diff = float("nan")
    if "SnapKV-16" in base_data:
        # Per-dataset comparison
        seed_labels = list(seeds_data.keys())
        datasets = list(seeds_data[seed_labels[0]]["individual_scores"].keys())
        our_avg = np.array([
            np.mean([seeds_data[s]["individual_scores"][ds] for s in seed_labels])
            for ds in datasets
        ])
        snap = np.array([base_data["SnapKV-16"]["individual_scores"][ds] for ds in datasets])
        diff = our_avg - snap
        mean_diff = float(diff.mean())
        try:
            wstat, wp = stats.wilcoxon(diff, alternative="greater", zero_method="zsplit")
            wp = float(wp)
        except Exception:
            wp = float("nan")

    summary = {
        "overall_mean": overall_mean,
        "overall_std":  overall_std,
        "95%_CI":       [ci_lo, ci_hi],
        "per_seed":     {s: float(seeds_data[s]["overall_average"]) for s in seeds_data},
        "baselines":    {l: float(b["overall_average"]) for l, b in base_data.items()},
        "vs_SnapKV_16": {
            "mean_per_dataset_diff": mean_diff,
            "wilcoxon_p_greater":    wp,
        },
        "per_task": {t: {"values": [float(v) for v in per_task[t]],
                          "mean":  float(np.mean(per_task[t])),
                          "std":   float(np.std(per_task[t], ddof=1)) if len(per_task[t]) > 1 else 0.0}
                      for t in labels},
    }
    out_json = f"{OUT_DIR}/q3_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved → {out_json}")
    print()
    print(f"Overall: {overall_mean:.3f} ± {overall_std:.3f}  (95% CI [{ci_lo:.2f}, {ci_hi:.2f}])")
    print(f"Per seed: {dict((s, round(float(seeds_data[s]['overall_average']), 2)) for s in seeds_data)}")
    print(f"Baselines: {dict((l, round(float(b['overall_average']), 2)) for l, b in base_data.items())}")
    if not np.isnan(mean_diff):
        print(f"vs SnapKV-16: mean per-dataset diff = +{mean_diff:.3f}, Wilcoxon p (greater) = {wp:.4f}")


if __name__ == "__main__":
    main()
