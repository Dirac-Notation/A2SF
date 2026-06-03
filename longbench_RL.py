"""LongBench evaluation for the trained RL champion (simple_ucb on sigmoid).

Workers run in separate processes, each pinned to a GPU group via
CUDA_VISIBLE_DEVICES. For 8B inference use --gpus_per_model 2 (model parallel
across two cards) to fit long contexts; 1B fits single GPU.
Cross-server: --shard_count 2 --shard_id {0,1} to round-robin tasks.

Orthogonal compression modifiers (ChunkKV, PyramidKV) are CLI-controllable
and applied via model.config; they propagate into build_selector at cache init.
"""
import os
import json
import argparse
import multiprocessing as mp
import time
from typing import Optional

import torch

from utils import set_seed
from RL.a2sf_model import A2SFModel, ModelConfig
from longbench_eval import dataset2metric


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="LongBench RL evaluation (champion ckpt only).")
    parser.add_argument("--model", type=str, required=True, choices=["llama3-8b", "llama3-1b", "qwen2"])
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--rl_checkpoint", type=str, required=True,
                        help="Path to RL policy_best.pt checkpoint.")
    parser.add_argument("--gpus_per_model", type=int, default=1,
                        help="GPUs per worker (model parallel). Use 2 for 8B on 24GB cards.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir; default = result_txt/pred/<budget>/<model>_sigmoid_<budget>_RL")
    parser.add_argument("--shard_count", type=int, default=1,
                        help="Cross-server shard count (round-robin task distribution).")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="This instance's shard id ∈ [0, shard_count).")
    # Orthogonal compression modifiers (ChunkKV / PyramidKV).
    parser.add_argument("--chunk_size", type=int, default=0,
                        help="ChunkKV: 0 = off, >0 = group head tokens into chunks of this size.")
    parser.add_argument("--pyramid_kv", action="store_true",
                        help="PyramidKV: vary per-layer budget (layer 0 max, layer L-1 min).")
    parser.add_argument("--pyramid_ratio", type=float, default=4.0,
                        help="PyramidKV bottom/top ratio (default 4.0).")
    parser.add_argument("--fixed_actions_json", type=str, default=None,
                        help="Optional path to {task_type: action_idx} JSON. When set, the RL "
                             "agent is bypassed and the action_idx for that task is used directly. "
                             "Used for ablation Config A (per-task fixed action).")
    parser.add_argument("--per_lh_lookup", type=str, default=None,
                        help="Path to per-(L,h) static lookup .pt with per_lh_a/per_lh_b "
                             "(shape (n_layers, n_kv_heads)). When set, overrides agent action: "
                             "each layer's scorer uses per-head (a, b) from lookup table.")
    parser.add_argument("--per_lh_policies_dir", type=str, default=None,
                        help="Directory containing per-(L,h) policy weights (L*h*.pt). "
                             "When set, each layer uses PerLHPolicyScorer that runs the policy "
                             "during prefill on the layer's snap to choose per-head (a, b).")
    parser.add_argument("--per_prompt_actions_json", type=str, default=None,
                        help="JSON {dataset: [action_idx, ...]} (one per prompt in order). "
                             "When set, each prompt uses its assigned action_idx (agent bypassed).")
    return parser.parse_args(args)


def load_jsonl_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# Build dataset → task_type lookup once (used by Config A fixed-action dispatch).
def _build_dataset_to_task() -> dict:
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "config", "task2dataset.json")
    with open(cfg_path) as f:
        t2d = json.load(f)
    return {ds: task for task, ds_list in t2d.items() for ds in ds_list}


_DATASET_TO_TASK = _build_dataset_to_task()


def _build_tasks(longbench_dir: str, dataset2maxlen: dict) -> list:
    tasks = []
    for dataset in dataset2maxlen.keys():
        jsonl_path = os.path.join(longbench_dir, f"{dataset}.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"Warning: {jsonl_path} not found, skipping {dataset}")
            continue
        for idx, obj in enumerate(load_jsonl_file(jsonl_path)):
            tasks.append({"dataset": dataset, "sample_idx": idx, "json_obj": obj})
    return tasks


def _load_rl_model(model_name: str, checkpoint_path: str) -> A2SFModel:
    """Load champion ckpt: simple_ucb agent with sigmoid compression."""
    model_cfg = ModelConfig.sigmoid(model=model_name)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch_config = ckpt.get("arch_config", {}) or {}
    # Encoder config from the trained ckpt (must match training).
    model_cfg.encoder_topk = int(arch_config.get("encoder_topk", 16))
    model_cfg.mini_attn_ckpt = str(arch_config.get("mini_attn_ckpt", "") or "")
    model_cfg.encoder_max_input_length = int(arch_config.get("encoder_max_input_length", 32768))
    model_cfg.encoder_include_hidden_pool = bool(arch_config.get("encoder_include_hidden_pool", True))
    model_cfg.encoder_hidden_pool_window = int(arch_config.get("encoder_hidden_pool_window", 0))
    model_cfg.single_view = bool(arch_config.get("single_view", False))
    model_cfg.encoder_feature_mode = str(arch_config.get("encoder_feature_mode", "stats"))
    model_cfg.extra_view = str(arch_config.get("extra_view", "none"))
    model_cfg.pre_rope_query_window = int(arch_config.get("pre_rope_query_window", 16))

    state_dict = ckpt.get("agent_state_dict") or ckpt.get("policy_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint missing 'agent_state_dict'.")
    return A2SFModel(config=model_cfg, state_dict=state_dict, arch_config=arch_config)


def _rl_worker(
    worker_id: int,
    gpu_group: list,
    model_name: str,
    rl_checkpoint: str,
    budget: int,
    max_length: int,
    dataset2maxlen: dict,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    chunk_size: int = 0,
    pyramid_kv: bool = False,
    pyramid_ratio: float = 4.0,
    fixed_actions: Optional[dict] = None,
    per_lh_lookup: Optional[str] = None,
    per_lh_policies_dir: Optional[str] = None,
    per_prompt_actions: Optional[dict] = None,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_group)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_grad_enabled(False)

    print(f"[worker {worker_id}] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    model = _load_rl_model(model_name, rl_checkpoint)
    # Orthogonal modifiers (ChunkKV / PyramidKV) propagate to build_selector via model.config.
    model.config.chunk_size = int(chunk_size)
    model.config.pyramid_kv = bool(pyramid_kv)
    model.config.pyramid_ratio = float(pyramid_ratio)
    if per_lh_lookup is not None:
        lk = torch.load(per_lh_lookup, map_location="cpu", weights_only=False)
        model.config.per_lh_a = lk["per_lh_a"]
        model.config.per_lh_b = lk["per_lh_b"]
        print(f"[worker {worker_id}] per-(L,h) lookup loaded: shape {lk['per_lh_a'].shape}")
    if per_lh_policies_dir is not None:
        from utils_real_drop.scorers import load_policies
        # Pick first weight file to read in_dim / hidden
        from pathlib import Path
        sample = sorted(Path(per_lh_policies_dir).glob("L*h*.pt"))[0]
        sample_ckpt = torch.load(sample, map_location="cpu", weights_only=False)
        in_dim = sample_ckpt["in_dim"]; hidden = sample_ckpt["hidden"]
        n_layers = model.model_runner.model.config.num_hidden_layers
        n_kv = model.model_runner.model.config.num_key_value_heads
        bank = load_policies(per_lh_policies_dir, n_layers, n_kv, in_dim, hidden)
        # Move policies to model device
        dev = next(model.model_runner.model.model.layers[0].parameters()).device
        for k, m in bank.items(): bank[k] = m.to(dev)
        model.config.per_lh_policy_bank = bank
        topk_from_dim = (in_dim - 3) // 2
        model.config.per_lh_topk = topk_from_dim
        print(f"[worker {worker_id}] per-(L,h) policies loaded: {len(bank)} policies, "
                f"in_dim={in_dim} (topk={topk_from_dim}), hidden={hidden}")
    if chunk_size or pyramid_kv:
        print(f"[worker {worker_id}] modifiers: chunk_size={chunk_size} pyramid_kv={pyramid_kv} ratio={pyramid_ratio}")
    tokenizer = model.model_runner.tokenizer
    print(f"[worker {worker_id}] model device={model.model_runner.model.device}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            dataset = task["dataset"]
            json_obj = task["json_obj"]
            sample_idx = int(task["sample_idx"])

            prompt = json_obj["input_prompt"]
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = (
                    tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
                    + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                )

            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                if "llama" in model_name:
                    prompt = f"[INST]{prompt}[/INST]"

            encoded = tokenizer(prompt, truncation=False, return_tensors="pt")
            context_length = int(encoded.input_ids.shape[-1])
            metric_fn = dataset2metric.get(dataset)
            metric_type = metric_fn.__name__ if metric_fn is not None else "qa_f1_score"
            max_gen = int(dataset2maxlen.get(dataset, 64))

            gen_kwargs = dict(
                prompt=prompt,
                metric_type=metric_type,
                token_budget=budget,
                answers=json_obj.get("answers", []),
                all_classes=json_obj.get("all_classes", []),
                dataset=dataset,
                task_type=json_obj.get("task_type"),
                tokenizer=tokenizer,
                stop_strings="[/INST]",
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                num_logits_to_keep=1,
            )
            if dataset == "samsum":
                gen_kwargs["min_length"] = context_length + 1
                gen_kwargs["eos_token_id"] = [
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ]

            # Config A: per-task fixed action override (bypass RL agent).
            if fixed_actions is not None:
                key = json_obj.get("task_type") or _DATASET_TO_TASK.get(dataset)
                if key is not None and key in fixed_actions:
                    fa = fixed_actions[key]
                    gen_kwargs["fixed_action_idx"] = int(
                        fa["action_idx"] if isinstance(fa, dict) else fa
                    )
            if per_prompt_actions is not None:
                ds_actions = per_prompt_actions.get(dataset)
                if ds_actions is not None and sample_idx < len(ds_actions):
                    gen_kwargs["fixed_action_idx"] = int(ds_actions[sample_idx])

            out = model.generate(**gen_kwargs)

            a_val = out.info.get("a")
            b_val = out.info.get("b")
            b_write = int(round(b_val)) if isinstance(b_val, (float, int)) else b_val

            result_queue.put((dataset, sample_idx, {
                "pred": out.pred_text,
                "answers": json_obj.get("answers", []),
                "all_classes": json_obj.get("all_classes", []),
                "length": json_obj.get("length"),
                "a": a_val,
                "b": b_write,
            }))
    except Exception as exc:
        result_queue.put(("__error__", f"worker {worker_id} gpus={gpu_group}: {exc}"))


def _run_longbench_rl_multi_gpu(args):
    set_seed(42)

    model_key = args.model.split("_")[0].lower()
    with open("config/model2maxlen.json", "r", encoding="utf-8") as f:
        max_length = int(json.load(f)[model_key])
    with open("config/dataset2maxlen.json", "r", encoding="utf-8") as f:
        dataset2maxlen = json.load(f)

    fixed_actions = None
    if args.fixed_actions_json:
        with open(args.fixed_actions_json) as f:
            fixed_actions = json.load(f)
        print(f"fixed_actions loaded ({len(fixed_actions)} tasks): {list(fixed_actions.keys())}")
    per_prompt_actions = None
    if args.per_prompt_actions_json:
        with open(args.per_prompt_actions_json) as f:
            per_prompt_actions = json.load(f)
        total = sum(len(v) for v in per_prompt_actions.values())
        print(f"per_prompt_actions loaded ({len(per_prompt_actions)} datasets, {total} prompts)")

    chunk_suffix = ""
    if int(args.chunk_size) > 1:
        chunk_suffix = f"_chunk{int(args.chunk_size)}"
    if args.pyramid_kv:
        chunk_suffix += "_pyr"
    output_dir = args.output_dir or f"result_txt/pred/{args.budget}/{args.model}_sigmoid_{args.budget}_RL{chunk_suffix}"
    os.makedirs(output_dir, exist_ok=True)

    longbench_dir = os.path.join("datasets", "longbench")
    tasks = _build_tasks(longbench_dir, dataset2maxlen)
    if not tasks:
        print("No LongBench tasks found. Exiting.")
        return

    # Length-balanced shuffle: deterministic across shards (same seed everywhere)
    # so all shards see the same shuffled order, then round-robin sharding splits
    # into balanced subsets without long-task clustering.
    import random as _random
    _random.Random(42).shuffle(tasks)

    if args.shard_count > 1:
        n_total = len(tasks)
        tasks = [t for i, t in enumerate(tasks) if i % args.shard_count == args.shard_id]
        print(f"shard {args.shard_id+1}/{args.shard_count}: {len(tasks)}/{n_total} tasks")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    visible_gpu_count = int(torch.cuda.device_count())
    if args.gpus_per_model <= 0 or args.gpus_per_model > visible_gpu_count:
        raise ValueError(f"--gpus_per_model must be in [1, {visible_gpu_count}]")

    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if parent_visible:
        all_gpu_ids = [int(g) for g in parent_visible.split(",") if g.strip()]
    else:
        all_gpu_ids = list(range(visible_gpu_count))
    gpu_groups = [
        all_gpu_ids[start : start + args.gpus_per_model]
        for start in range(0, len(all_gpu_ids), args.gpus_per_model)
    ]
    gpu_groups = [g for g in gpu_groups if len(g) == args.gpus_per_model]
    print(f"gpu_groups={gpu_groups}")

    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()

    for t in tasks:
        task_queue.put(t)
    for _ in gpu_groups:
        task_queue.put(None)

    processes = []
    for worker_id, gpu_group in enumerate(gpu_groups):
        p = ctx.Process(
            target=_rl_worker,
            args=(worker_id, gpu_group, model_key, args.rl_checkpoint,
                  int(args.budget), max_length, dataset2maxlen,
                  task_queue, result_queue),
            kwargs=dict(
                chunk_size=int(args.chunk_size),
                pyramid_kv=bool(args.pyramid_kv),
                pyramid_ratio=float(args.pyramid_ratio),
                fixed_actions=fixed_actions,
                per_lh_lookup=args.per_lh_lookup,
                per_lh_policies_dir=args.per_lh_policies_dir,
                per_prompt_actions=per_prompt_actions,
            ),
        )
        p.start()
        processes.append(p)

    remaining = len(tasks)
    done = 0
    started_at = time.time()
    print(f"total tasks: {remaining}")

    while remaining > 0:
        item = result_queue.get()
        if item[0] == "__error__":
            raise RuntimeError(item[1])

        dataset, sample_idx, payload = item
        with open(os.path.join(output_dir, f"{dataset}.jsonl"), "a", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.write("\n")

        remaining -= 1
        done += 1
        elapsed = max(1e-6, time.time() - started_at)
        rate = done / elapsed
        pct = 100.0 * done / len(tasks)
        eta = int((len(tasks) - done) / rate) if rate > 0 else 0
        print(f"progress: {done}/{len(tasks)} ({pct:.1f}%) | {rate:.2f}/s | ETA {eta}s", flush=True)

    for p in processes:
        p.join()

    print("\nRL LongBench evaluation completed!")


if __name__ == "__main__":
    args = parse_args()
    _run_longbench_rl_multi_gpu(args)
