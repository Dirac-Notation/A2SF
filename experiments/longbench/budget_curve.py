"""Figure 3 / Figure A.1: LongBench average vs KV budget.

Reads result_txt/pred/<B>/<run>/result.json (longbench_eval.py) for the default run names of longbench.py:
    H2O     <m>_snap_32768_<B>              TOVA    <m>_snap_1_<B>
    SnapKV  <m>_snap_16_<B>                 SnapKV + pooling  <m>_snap_16_<B>_pool7
    Q-Trust <m>_qtrust_16_<B>_kern_tau1     Full    <m>_full_16_128
Missing runs are left out of the plot.

    python experiments/longbench/budget_curve.py --models llama3-8b --out budget_curve_llama.pdf
    python experiments/longbench/budget_curve.py --models qwen2,mistral-7b-v03 --out budget_curve_app.pdf
"""
import argparse, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42; plt.rcParams["ps.fonttype"] = 42

B = [128, 256, 512, 1024]
RUNS = {"H2O": "{m}_snap_32768_{b}", "TOVA": "{m}_snap_1_{b}", "SnapKV": "{m}_snap_16_{b}",
        "SnapKV + pooling": "{m}_snap_16_{b}_pool7", "Q-Trust (ours)": "{m}_qtrust_16_{b}_kern_tau1"}
NAMES = {"llama3-8b": "Llama-3.1-8B", "qwen2": "Qwen2.5-7B", "mistral-7b-v03": "Mistral-7B-v0.3"}
COL = {"H2O": "#c05621", "TOVA": "#888888", "SnapKV": "#2b6cb0", "SnapKV + pooling": "#2b6cb0", "Q-Trust (ours)": "#2f855a"}
MK = {"H2O": "v", "TOVA": "s", "SnapKV": "^", "SnapKV + pooling": "^", "Q-Trust (ours)": "o"}
LS = {"SnapKV + pooling": "--"}


def score(pred, b, run):
    f = os.path.join(pred, str(b), run, "result.json")
    return json.load(open(f))["overall_average"] if os.path.exists(f) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama3-8b"); ap.add_argument("--pred", default="result_txt/pred")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    models = a.models.split(",")
    single = len(models) == 1
    fig, axes = plt.subplots(1, len(models), figsize=(3.2, 2.35) if single else (3.0 * len(models), 2.4), dpi=130, squeeze=False)
    axes = axes[0]
    for ax, m in zip(axes, models):
        for meth, pat in RUNS.items():
            pts = [(b, score(a.pred, b, pat.format(m=m, b=b))) for b in B]
            pts = [p for p in pts if p[1] is not None]
            if not pts:
                continue
            print(m, meth, pts)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=MK[meth], ms=3.5, lw=1.6, color=COL[meth], label=meth,
                    ls=LS.get(meth, "-"), mfc="white" if meth in LS else COL[meth], zorder=3 if meth.startswith("Q-Trust") else 2)
        full = score(a.pred, 128, f"{m}_full_16_128")
        if full is not None:
            ax.axhline(full, color="#444", ls="--", lw=0.8)
            ax.text(1024, full - 0.35, "full cache", fontsize=8, ha="right", va="top", color="#111")
        ax.set_xscale("log", base=2); ax.set_xticks(B); ax.set_xticklabels([str(b) for b in B], fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(NAMES.get(m, m), fontsize=10)
        ax.set_xlabel("KV budget", fontsize=9)
        ax.grid(alpha=.25, ls=":")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("LongBench average", fontsize=9)
    if single:
        axes[0].set_title("")
        axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, frameon=False,
                       handlelength=1.6, columnspacing=1.0, borderaxespad=0.1)
        fig.tight_layout()
    else:
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    main()
