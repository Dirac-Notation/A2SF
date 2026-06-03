"""Reproduce the three paper Observation figures end-to-end, from scratch.

Run:
    conda activate A2SF
    python reproduce.py            # generates data (GPU) then plots all 3 figures
    python reproduce.py --plot-only  # re-plot only (no GPU), data must already exist

Outputs (this directory):
    obs1_sigmoid_band.pdf        Obs1: per-prompt Tanimoto-optimal weight w_d
                                 (mean ± std) with sigmoid fit, 4 tasks.
    obs1_tanimoto_recovery.pdf   Obs1: running Tanimoto recovery vs the oracle for
                                 single / optimal / uniform / sigmoid weighting.
    obs2_window_dominance.pdf    Obs2: selection-density window dominance (3 panels).

Structure (everything needed lives in this directory)
-----------------------------------------------------
    common.py   shared config + GPU/compute helpers (model & prompt loading,
                AttentionCollector, teacher-forcing oracle, Tanimoto math).
    obs1.py     generate (GPU, minimal data) + plot the two obs1 figures.
    obs2.py     generate (GPU) + plot obs2_window_dominance.
    data/       tiny per-task .npz curves written by generate (gitignored).
    reproduce.py  this driver.

Data flow
---------
    obs1:  LongBench prompts --(1B prefill, windowed attention)-->
           + teacher-forced oracle attention --> per-prompt Tanimoto curves
           (w_tan, j_tan_optimal, j_tan_sigmoid, j_uniform_tanimoto,
            j_single_tanimoto) saved to data/<Task>__<dataset>.npz.
           The large prefill/oracle tensors are consumed in memory and discarded
           (no multi-GB intermediates), so disk use is a few MB, not ~125 GB.

    obs2:  LongBench prompts --(1B all-layer attention)--> per-(layer, head)
           top-B selection densities under hard windows {1,16,128,full} -->
           cross-window / cross-prompt Tanimoto matrices, cached in
           data/obs2_data.npz, then plotted.

Exact reproduction
------------------
Prompt sampling is deterministic (global SEED=42, window=256, 40 prompts/dataset
in TASK_GROUP order) and was verified to match the archived paper run by
comparing per-prompt seq_lens against /data2/hyunrae/plots. The numerical logic
(Tanimoto coordinate descent, single-query and sigmoid-fit curves) is ported
unchanged from the original scripts, so re-running reproduces the committed
figures.
"""
import os
import sys
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                    help="skip GPU data generation; re-plot from data/")
    args = ap.parse_args()
    extra = ["--plot-only"] if args.plot_only else []

    for script in ("obs1.py", "obs2.py"):
        print(f"\n{'='*70}\n=== running {script} {' '.join(extra)}\n{'='*70}", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, script), *extra], check=True)

    print("\nDone. Figures:")
    for f in ("obs1_sigmoid_band.pdf", "obs1_tanimoto_recovery.pdf",
              "obs2_window_dominance.pdf"):
        print(f"  {os.path.join(HERE, f)}")


if __name__ == "__main__":
    main()
