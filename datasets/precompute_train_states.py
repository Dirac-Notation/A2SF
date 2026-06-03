"""Pre-compute and cache MiniAttnEncoder states for training data.

After running, use --states_file in train.py to skip model loading and encoding.
This enables fast agent-only training iterations without re-encoding.

Usage:
    python datasets/precompute_train_states.py \\
        --model llama3-1b \\
        --mini_attn_ckpt runs/mini_attn_v5/mini_attn_best.pt \\
        --data_file datasets/training/scored/llama3-1b/train.jsonl \\
        --out runs/states/llama3-1b_train.pt \\
        --gpu 0

Output .pt keys:
    {prompt_id: int} → Tensor[state_dim]   float32
    state_dim         int
    num_metric_types  int
    num_task_types    int
    side_dim          int
    num_heads         int
    num_hidden_pool   int
    config            dict  (encoder settings for reproducibility)
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3-1b")
    p.add_argument("--mini_attn_ckpt", default="runs/mini_attn_v5/mini_attn_best.pt")
    p.add_argument("--encoder_topk", type=int, default=16)
    p.add_argument("--encoder_max_input_length", type=int, default=32768)
    p.add_argument("--data_file", default="datasets/training/scored/llama3-1b/train.jsonl",
                   help="Train JSONL. Val JSONL inferred by replacing train/ with validation/.")
    p.add_argument("--val_data_file", default=None)
    p.add_argument("--out", required=True, help="Output .pt file path.")
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[precompute] model={args.model}  device={device}")

    from RL.a2sf_model import ModelConfig
    from RL.env import A2SFModelRunner, A2SFEnv

    mc = ModelConfig.sigmoid(model=args.model)
    mc.mini_attn_ckpt = args.mini_attn_ckpt
    mc.encoder_topk = args.encoder_topk
    mc.encoder_max_input_length = args.encoder_max_input_length

    runner = A2SFModelRunner(mc)
    env = A2SFEnv(runner, mc)
    enc = env.context_encoder

    val_data_file = args.val_data_file or args.data_file.replace("/train/", "/validation/")

    # Collect all records with global prompt_id
    records = []
    for split, path in [("train", args.data_file), ("val", val_data_file)]:
        if not os.path.exists(path):
            print(f"  [skip] {split} file not found: {path}")
            continue
        offset = len(records)
        with open(path) as f:
            for j, line in enumerate(f):
                r = json.loads(line)
                if not r.get("input_prompt"):
                    continue
                records.append({
                    "_prompt_id": offset + j,
                    "prompt": r["input_prompt"],
                    "metric_type": str(r.get("metric_type", "qa_f1_score")),
                    "task_type": str(r.get("task_type", "unknown")),
                    "dataset": r.get("dataset"),
                    "generation_length": int(r.get("generation_length", 0)),
                })
        print(f"  {split}: {j+1} records from {path}")

    prompt_to_state = {}
    t0 = time.time()
    with torch.inference_mode():
        for r in records:
            pid = r["_prompt_id"]
            if pid in prompt_to_state:
                continue
            state = enc.encode_context(
                text=r["prompt"],
                generation_length=r["generation_length"],
                token_budget=128,
                metric_type=r["metric_type"],
                task_type=r["task_type"],
                dataset=r["dataset"],
            ).cpu()
            prompt_to_state[pid] = state
            n = len(prompt_to_state)
            if n % 100 == 0:
                print(f"  encoded {n}/{len(records)}  ({time.time()-t0:.0f}s)", flush=True)

    state_dim = int(enc.output_dim)
    print(f"[precompute] {len(prompt_to_state)} states  state_dim={state_dim}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_dict = dict(prompt_to_state)
    save_dict["state_dim"] = state_dim
    save_dict["num_metric_types"] = int(enc.num_metric_types)
    save_dict["num_task_types"] = int(getattr(enc, "num_task_types", 0))
    save_dict["side_dim"] = int(getattr(enc, "side_dim", 0))
    save_dict["num_heads"] = int(getattr(enc, "num_heads", 1))
    save_dict["num_hidden_pool"] = int(getattr(enc, "hidden_pool_dim", 0))
    save_dict["config"] = {
        "encoder_topk": args.encoder_topk,
        "mini_attn_ckpt": args.mini_attn_ckpt,
    }
    torch.save(save_dict, args.out)
    print(f"[precompute] saved → {args.out}")


if __name__ == "__main__":
    main()
