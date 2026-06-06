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

    labels = [t for t in tasks if per_task[t]]
    base_order = [b for b, _ in BASELINES if b in base_data]
    base_colors = {"SnapKV-16": "#d62728", "SnapKV-32": "#ff7f0e",
                   "TOVA": "#9467bd", "H2O": "#7f7f7f"}
    OURS_C = "#1f77b4"

    # ── One zoomed panel per metric (Overall + each task) ────────────────────────
    # A shared 0-50 axis hides the (small) seed-to-seed spread, which is the whole
    # point of this figure. Per-panel zoom makes both the seed spread and the
    # baseline gaps visible.
    def draw_panel(ax, ours_vals, base_vals, title, ylabel=False, emphasize=False):
        ours = np.array(ours_vals, dtype=float)
        o_mean = ours.mean()
        o_min, o_max = ours.min(), ours.max()
        present = [v for v in base_vals.values() if v is not None and not np.isnan(v)]
        lo, hi = min(list(ours) + present), max(list(ours) + present)
        pad = max((hi - lo) * 0.25, 0.15)

        # Ours: min-max range band + mean marker (whiskers reach min and max,
        # so the mean sits as the centre of the range).
        ax.axhspan(o_min, o_max, color=OURS_C, alpha=0.13, zorder=0)
        ax.axhline(o_mean, color=OURS_C, lw=1.4, alpha=0.8, zorder=1)
        ax.errorbar(0, o_mean, yerr=[[o_mean - o_min], [o_max - o_mean]],
                    fmt="o", ms=9, color=OURS_C, ecolor=OURS_C, elinewidth=1.6,
                    capsize=5, zorder=4, markeredgecolor="black", markeredgewidth=0.6)
        for j, b in enumerate(base_order, start=1):
            v = base_vals.get(b)
            if v is None or np.isnan(v):
                continue
            ax.scatter(j, v, marker="D", s=70, c=base_colors.get(b, "black"),
                       zorder=6, edgecolor="black", linewidth=0.6)

        ax.set_xlim(-0.6, len(base_order) + 0.6)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(range(1 + len(base_order)))
        ax.set_xticklabels(["Ours"] + base_order, fontsize=8.5, rotation=30, ha="right")
        ax.set_title(title, fontsize=13 if emphasize else 12,
                     fontweight="bold" if emphasize else "normal")
        ax.grid(True, axis="y", alpha=0.3)
        if emphasize:
            for s in ax.spines.values():
                s.set_edgecolor(OURS_C); s.set_linewidth(1.6)
        if ylabel:
            ax.set_ylabel("LB128 score")

    # Overall panel first (headline), then one panel per task.
    panels = [("Overall", list(overall),
               {b: base_data[b]["overall_average"] for b in base_order}, True)]
    panels += [(t, per_task[t],
                {b: base_data[b]["group_averages"].get(t) for b in base_order}, False)
               for t in labels]

    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).flatten()
    for idx, (title, ov, bv, emph) in enumerate(panels):
        draw_panel(axes[idx], ov, bv, title, ylabel=(idx % ncol == 0), emphasize=emph)
    for k in range(len(panels), len(axes)):
        axes[k].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=OURS_C,
                      markeredgecolor="black", markersize=9, label="Ours")]
    handles += [Line2D([0], [0], marker="D", color="w", markerfacecolor=base_colors[b],
                       markeredgecolor="black", markersize=9, label=b) for b in base_order]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=10.5)
    fig.suptitle("Multi-seed stability vs. baselines", y=1.05, fontsize=14)
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
