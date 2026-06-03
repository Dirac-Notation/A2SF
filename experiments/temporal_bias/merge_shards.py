"""Merge per-shard partial npz files into the final metrics.npz / coord_descent.npz
expected by experiments/paper_figures/observations/fig2_optimal_coefficient.py.

Per-shard files (produced by parallel_obs2.py) live at:
  plots[/b<budget>]/<task>/<dataset>/shard_<i>_of_<S>.npz
  with keys: br, w_cd, j_cd, j_weight, sample_idx, seq_len, chunk, window

Final files written here:
  plots[/b<budget>]/<task>/<dataset>/coord_descent.npz   (w_cd, j_cd, j_weight, chunk, window)
  plots[/b<budget>]/<task>/<dataset>/metrics.npz         (br)

Usage:
  python -m experiments.temporal_bias.merge_shards --budget 128 --shard_count 2
"""
import os, sys, glob, argparse
import numpy as np

WORKPATH = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--budget",      type=int, default=128)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--cleanup",     action="store_true",
                   help="delete per-shard partial files after merge")
    return p.parse_args()


def main():
    args = parse_args()
    plot_dir = os.path.join(WORKPATH, "plots")
    if args.budget != 128:
        plot_dir = os.path.join(plot_dir, f"b{args.budget}")

    if not os.path.isdir(plot_dir):
        print(f"no plot dir: {plot_dir}"); return

    # walk task / dataset
    for task in sorted(os.listdir(plot_dir)):
        task_dir = os.path.join(plot_dir, task)
        if not os.path.isdir(task_dir): continue
        for dataset in sorted(os.listdir(task_dir)):
            ds_dir = os.path.join(task_dir, dataset)
            if not os.path.isdir(ds_dir): continue

            partials = sorted(glob.glob(
                os.path.join(ds_dir, f"shard_*_of_{args.shard_count}.npz")))
            if not partials:
                continue
            if len(partials) != args.shard_count:
                print(f"warn: {ds_dir} has {len(partials)}/{args.shard_count} partials, skipping")
                continue

            # Concatenate
            br_list, w_list, j_list, jw_list, idx_list, seq_list = [], [], [], [], [], []
            chunk = window = None
            for p in partials:
                z = np.load(p)
                br_list.append(z["br"])
                w_list.append(z["w_cd"])
                j_list.append(z["j_cd"])
                jw_list.append(z["j_weight"])
                idx_list.append(z["sample_idx"])
                seq_list.append(z["seq_len"])
                chunk = int(z["chunk"]); window = int(z["window"])

            br = np.concatenate(br_list, axis=0)
            w_cd = np.concatenate(w_list, axis=0)
            j_cd = np.concatenate(j_list, axis=0)
            j_weight = np.concatenate(jw_list, axis=0)
            sample_idx = np.concatenate(idx_list, axis=0)
            seq_len = np.concatenate(seq_list, axis=0)

            # Order by sample_idx for determinism
            order = np.argsort(sample_idx, kind="stable")
            br = br[order]; w_cd = w_cd[order]
            j_cd = j_cd[order]; j_weight = j_weight[order]
            sample_idx = sample_idx[order]; seq_len = seq_len[order]

            np.savez_compressed(
                os.path.join(ds_dir, "coord_descent.npz"),
                w_cd=w_cd, j_cd=j_cd, j_weight=j_weight,
                chunk=np.int32(chunk), window=np.int32(window),
            )
            np.savez_compressed(
                os.path.join(ds_dir, "metrics.npz"),
                br=br, sample_idx=sample_idx, seq_len=seq_len,
            )
            print(f"merged {dataset}: N={br.shape[0]} W={br.shape[1]} → "
                  f"{ds_dir}/{{coord_descent,metrics}}.npz")

            if args.cleanup:
                for p in partials:
                    os.remove(p)


if __name__ == "__main__":
    main()
