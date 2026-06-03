"""Visualize how the selected top-B token set evolves with observation
window d, for a single (layer, kv_head) on a single prompt.

For each d in --windows, the per-key score is the sum of attention from the
last d query positions (matching the gqa_topk reduction used in obs2). Local
last-(LOCAL_RATIO * B) keys are always kept.

Outputs:
  result_txt/analysis/head_window_selection/<dataset>_p<idx>_L<l>_KV<kv>_b<B>.png
    Top panel  : heatmap rows=d, cols=key position, white=in selection.
    Bottom plot: Jaccard(top-B(d), top-B(d_max)) recovery curve over d.

Usage:
  python -m experiments.temporal_bias.head_window_selection \
      --dataset hotpotqa --prompt_idx 0 --layer 8 --kv_head 4 \
      --budget 128 --max_window 256
"""
import os, sys, json, argparse, random
import numpy as np
import torch
import matplotlib.pyplot as plt

WORKPATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(WORKPATH))
sys.path.append(ROOT_PATH)

from experiments.temporal_bias.optimal import (
    AttentionCollector, MAX_SEQ_LEN, LENGTH_MIN, LENGTH_MAX,
    MODEL_NAME, SEED, LOCAL_RATIO,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="hotpotqa")
    p.add_argument("--prompt_idx", type=int, default=0)
    p.add_argument("--layer",      type=int, default=8)
    p.add_argument("--kv_head",    type=int, default=4)
    p.add_argument("--budget",     type=int, default=128)
    p.add_argument("--max_window", type=int, default=256)
    p.add_argument("--gpu",        type=int, default=0)
    p.add_argument("--windows",    default="1,4,16,32,64,128,192,256",
                   help="comma-separated d values to visualize")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    with open(os.path.join(ROOT_PATH, "config", "model2path.json")) as f:
        model_path = json.load(f)[MODEL_NAME]

    longbench_dir = os.path.join(ROOT_PATH, "datasets", "longbench")
    prompts = []
    for fname in os.listdir(longbench_dir):
        with open(os.path.join(longbench_dir, fname)) as f:
            for line in f:
                item = json.loads(line)
                if (item["dataset"] == args.dataset
                        and LENGTH_MIN <= item.get("length", 0) <= LENGTH_MAX):
                    prompts.append(item["input_prompt"])
    if not prompts:
        raise RuntimeError(f"no prompts for dataset={args.dataset}")
    random.Random(SEED).shuffle(prompts)
    if args.prompt_idx >= len(prompts):
        raise RuntimeError(f"prompt_idx={args.prompt_idx} out of range "
                           f"(have {len(prompts)})")
    prompt = prompts[args.prompt_idx]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
    ).eval()
    device = next(model.parameters()).device

    collector = AttentionCollector(model, args.max_window)
    enc = tokenizer(f"[INST]{prompt}[/INST]", return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    if input_ids.size(1) > MAX_SEQ_LEN:
        half = MAX_SEQ_LEN // 2
        input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)
    seq_len = int(input_ids.size(1))
    collector.reset(seq_len)

    with torch.no_grad():
        out = model(input_ids, use_cache=True, num_logits_to_keep=1)
        past_kv = out.past_key_values
        data = collector.compute_window_data(past_kv)
        del out, past_kv

    prefill_attn = data["prefill_attn"]            # (L, H, W, S) on CPU
    L, H, W, S = prefill_attn.shape
    num_kv = collector.num_kv_heads
    gs = collector.group_size
    if not (0 <= args.layer < L):
        raise RuntimeError(f"layer {args.layer} out of range [0,{L})")
    if not (0 <= args.kv_head < num_kv):
        raise RuntimeError(f"kv_head {args.kv_head} out of range [0,{num_kv})")

    # group query heads under kv_head (gqa)
    pa = prefill_attn.view(L, num_kv, gs, W, S).sum(dim=2)   # (L, KV, W, S)
    head_attn = pa[args.layer, args.kv_head].float()         # (W, S)

    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    windows = sorted(set(w for w in windows if 1 <= w <= W))

    B = args.budget
    local_b = max(1, int(B * LOCAL_RATIO))
    sel_b = B - local_b
    recent_idx = torch.arange(S - local_b, S)

    masks = []
    for d in windows:
        score = head_attn[-d:, :].sum(dim=0)   # (S,)
        s = score.clone()
        s[recent_idx] = -float("inf")          # exclude local from top-K race
        topk = torch.topk(s, sel_b).indices
        keep = torch.zeros(S, dtype=torch.bool)
        keep[topk] = True
        keep[recent_idx] = True
        masks.append(keep.numpy())
    masks = np.stack(masks, axis=0)            # (D, S)

    # Jaccard vs largest-d
    ref = masks[-1].astype(bool)
    j_curve = []
    for m in masks:
        m = m.astype(bool)
        inter = (m & ref).sum()
        union = (m | ref).sum()
        j_curve.append(inter / max(union, 1))

    out_dir = os.path.join(ROOT_PATH, "result_txt", "analysis",
                           "head_window_selection")
    os.makedirs(out_dir, exist_ok=True)
    fname = (f"{args.dataset}_p{args.prompt_idx}_L{args.layer}_KV{args.kv_head}"
             f"_b{B}.png")
    out_path = os.path.join(out_dir, fname)

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 6), dpi=100,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax = axes[0]
    ax.imshow(masks, aspect="auto", cmap="Greys", interpolation="nearest",
              extent=[0, S, len(windows), 0])
    ax.set_yticks([i + 0.5 for i in range(len(windows))])
    ax.set_yticklabels([f"d={d}" for d in windows])
    ax.set_xlabel(f"key position  (S={S})")
    ax.set_ylabel("observation window d")
    ax.set_title(f"top-{B} selection vs window  |  "
                 f"layer {args.layer}, kv-head {args.kv_head}  |  "
                 f"{args.dataset} #{args.prompt_idx}")

    ax = axes[1]
    ax.plot(windows, j_curve, "o-", color="C0")
    ax.set_xlabel("observation window d")
    ax.set_ylabel(f"Jaccard vs d={windows[-1]}")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {out_path}")
    print(f"S={S}  W={W}  B={B}  local_b={local_b}  sel_b={sel_b}")
    print("Jaccard vs adjacent d:")
    for i in range(1, len(windows)):
        a = masks[i-1].astype(bool); b = masks[i].astype(bool)
        j = (a & b).sum() / max((a | b).sum(), 1)
        print(f"  d={windows[i-1]:3d} -> d={windows[i]:3d}: J={j:.3f}")
    print(f"Jaccard vs d={windows[-1]}:")
    for d, j in zip(windows, j_curve):
        print(f"  d={d:3d}: J={j:.3f}")


if __name__ == "__main__":
    main()
