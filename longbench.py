import os
import json
import argparse
import multiprocessing as mp
import time

from tqdm import tqdm
import torch

from utils import load_model, set_seed, CompressionConfig, build_chat_prompt, chat_stop_strings


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="LongBench end-to-end evaluation (multi-GPU, multi-process)")
    parser.add_argument("--model", type=str, required=True, choices=["llama3-8b", "llama3-1b", "qwen2", "mistral-7b"])
    parser.add_argument("--method", type=str, default="full")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--gpus_per_model", type=int, default=1, help="한 모델 인스턴스가 사용할 GPU 개수 (연속된 ID 그룹).")
    parser.add_argument(
        "--aj_weight_fn", type=str, default="aj_offset",
        choices=["aj_offset", "aj", "aj_sqrt", "aj_fastrise", "aj_quartic",
                 "aj_floor30", "aj_floor50", "aj_floor70",
                 "aj_mix25", "aj_mix50", "aj_mix75",
                 "aj_gate30", "aj_gate50", "aj_sqrt_gate30", "aj_norm_sqrt"],
        help="(AJ only) weight function applied to Jaccard signal.",
    )
    parser.add_argument("--aj_offset", type=float, default=0.1, help="(AJ only) offset for aj_offset weight.")
    parser.add_argument("--recent_budget", type=int, default=16, help="(AJ only) number of keys always kept from the tail.")
    parser.add_argument("--n_sink", type=int, default=0, help="Always-keep first n_sink (attention-sink) tokens. Helps attention-free scorers (KeyDiff/L2/TriAtt).")
    parser.add_argument("--ada_kv", action="store_true", help="Ada-KV head-wise adaptive budget allocation (orthogonal to scorer; pad-to-max sim). Composes with any --method / --waits_table.")
    parser.add_argument("--chunk_size", type=int, default=0,
                        help="ChunkKV: 0 = off (token-level select), >0 = group head tokens into chunks of this size.")
    parser.add_argument("--chunk_group_size", type=int, default=1,
                        help="ChunkKV LIR: layers per group sharing one selection (1 = no sharing).")
    parser.add_argument("--pyramid_kv", action="store_true",
                        help="PyramidKV: linearly decreasing per-layer budget.")
    parser.add_argument("--pyramid_ratio", type=float, default=4.0,
                        help="PyramidKV bottom/top ratio (default 4.0).")
    parser.add_argument("--shard_count", type=int, default=1,
                        help="Cross-server shard count (round-robin task distribution).")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="This instance's shard id ∈ [0, shard_count).")
    parser.add_argument("--shard_weights", type=str, default=None,
                        help="Comma-separated per-shard weights (len == shard_count) for "
                             "GPU-speed-proportional splitting, e.g. '1,1,1.5,1.5' for "
                             "3090,3090,4090,4090. Omit for equal round-robin. Each shard takes a "
                             "contiguous slice of the (deterministically shuffled) task list sized "
                             "by its weight fraction.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Override auto-generated run name (output dir suffix).")
    parser.add_argument("--curve", type=str, default="sigmoid",
                        choices=["sigmoid", "exp", "linear", "gauss"],
                        help="WAITS weight-curve family (rebuttal alt-function study). "
                             "Non-sigmoid curves read b as the scale (tau/W/sigma).")
    parser.add_argument("--waits_table", type=str, default=None,
                        help="Per-prompt WAITS action table JSON (result_txt/backup/waits_action_table.json). "
                             "If set, overrides --method/--window with per-prompt (a, b) from the table.")
    parser.add_argument("--min_prompt_tokens", type=int, default=0,
                        help="Skip samples whose tokenized length < this value (Phase 2: long sequences).")
    parser.add_argument("--max_prompt_tokens", type=int, default=0,
                        help="Skip samples whose tokenized length >= this value; 0=no limit (Phase 1: short sequences).")
    parser.add_argument("--sigmoid_a", type=float, default=None,
                        help="Override 'a' parameter for sigmoid compression (use with --method sigmoid).")
    parser.add_argument("--key_prior", type=str, default=None,
                        help="Key-position prior curve multiplying accumulated scores before "
                             "selection: 'band:gamma:lo:hi' or 'sink:gamma:d:c'. Composes with "
                             "any --method / --waits_table. Default off.")
    parser.add_argument("--value_weight", type=float, default=None,
                        help="Value-aware scoring exponent p: score *= ||v_k||^p. Default off.")
    parser.add_argument("--triattention_stats", type=str, default=None,
                        help="Path to calibrated TriAttention stats .pt file. "
                             "Sets compression_method=triattention automatically.")
    return parser.parse_args(args)


def load_jsonl_file(file_path: str):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def _build_tasks(longbench_dir: str, dataset2maxlen: dict) -> list:
    """
    LongBench 전체에 대해 (dataset, sample_idx, json_obj)를 나열한 태스크 리스트 생성.
    """
    tasks = []
    for dataset in dataset2maxlen.keys():
        jsonl_path = os.path.join(longbench_dir, f"{dataset}.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"Warning: {jsonl_path} not found, skipping {dataset}")
            continue
        data = load_jsonl_file(jsonl_path)
        for idx, obj in enumerate(data):
            tasks.append(
                {
                    "dataset": dataset,
                    "sample_idx": idx,
                    "json_obj": obj,
                }
            )
    return tasks


def _longbench_worker(
    worker_id: int,
    gpu_group: list,
    model_name: str,
    method: str,
    window: int,
    budget: int,
    max_length: int,
    dataset2maxlen: dict,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    aj_weight_fn: str = "aj_offset",
    aj_offset: float = 0.1,
    recent_budget: int = 16,
    n_sink: int = 0,
    ada_kv: bool = False,
    chunk_size: int = 0,
    chunk_group_size: int = 1,
    pyramid_kv: bool = False,
    pyramid_ratio: float = 4.0,
    waits_table: dict = None,
    min_prompt_tokens: int = 0,
    max_prompt_tokens: int = 0,
    sigmoid_a: float = None,
    curve: str = "sigmoid",
    triattention_stats: dict = None,
    key_prior: str = None,
    value_weight: float = None,
):
    """
    CUDA_VISIBLE_DEVICES를 gpu_group으로 설정하고 모델을 로드한 뒤,
    task_queue에서 하나씩 (dataset, sample_idx, json_obj)를 꺼내 예측을 수행.
    결과는 result_queue로 넘긴다.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_group)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_grad_enabled(False)

    print(f"[longbench][worker {worker_id}] start with CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    model, tokenizer = load_model(model_name)
    print(f"[longbench][worker {worker_id}] model device={model.device}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = CompressionConfig()
    config["compression_method"] = method
    config["observation_window"] = window
    config["total_budget"] = budget
    config["a"] = 10
    config["b"] = window
    config["aj_weight_fn"] = aj_weight_fn
    config["aj_offset"] = aj_offset
    config["recent_budget"] = recent_budget
    config["n_sink"] = n_sink
    config["ada_kv"] = ada_kv
    config["chunk_size"] = int(chunk_size)
    config["chunk_group_size"] = int(chunk_group_size)
    config["pyramid_kv"] = bool(pyramid_kv)
    config["pyramid_ratio"] = float(pyramid_ratio)
    config["key_prior"] = key_prior
    config["value_weight"] = value_weight
    if sigmoid_a is not None:
        config["a"] = float(sigmoid_a)
    config["curve"] = str(curve)
    if triattention_stats is not None:
        config["compression_method"] = "triattention"
        config["triattention_stats"] = triattention_stats

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
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                    tokenized_prompt[-half:], skip_special_tokens=True
                )

            # Per-model chat template (config/model2chat.json): llama/mistral -> [INST],
            # qwen -> native ChatML, etc. Few-shot/completion datasets skip wrapping.
            prompt = build_chat_prompt(prompt, model_name, tokenizer, dataset)

            encoded = tokenizer(prompt, truncation=False, return_tensors="pt")
            input_ids = encoded.input_ids.to(model.device)
            attention_mask = encoded.attention_mask.to(torch.bfloat16).to(model.device)
            context_length = int(input_ids.shape[-1])

            if min_prompt_tokens > 0 and context_length < min_prompt_tokens:
                result_queue.put(("__skip__", dataset, sample_idx))
                continue
            if max_prompt_tokens > 0 and context_length >= max_prompt_tokens:
                result_queue.put(("__skip__", dataset, sample_idx))
                continue

            max_gen = int(dataset2maxlen.get(dataset, 64))

            if waits_table is not None:
                ab = waits_table.get(dataset, [None] * (sample_idx + 1))[sample_idx]
                if ab is not None:
                    config["compression_method"] = "waits"
                    config["a"] = float(ab[0])
                    config["b"] = int(ab[1])

            if method == "kvzip":
                from utils_real_drop.kvzip import kvzip_generate
                stop_ids = [tokenizer.eos_token_id]
                if dataset == "samsum":
                    try:
                        stop_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])
                    except Exception:
                        pass
                gen_ids = kvzip_generate(
                    model, tokenizer, input_ids, max_new_tokens=max_gen,
                    budget=budget, recent_budget=recent_budget, n_sink=n_sink,
                    stop_token_ids=stop_ids,
                ).to(input_ids.device)
                output = torch.cat([input_ids[0], gen_ids])
            else:
                model.init_cache(config)
                with torch.inference_mode():
                    if dataset == "samsum":
                        output = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_gen,
                            num_beams=1,
                            do_sample=False,
                            min_length=context_length + 1,
                            eos_token_id=[
                                tokenizer.eos_token_id,
                                tokenizer.encode("\n", add_special_tokens=False)[-1],
                            ],
                            pad_token_id=tokenizer.eos_token_id,
                            tokenizer=tokenizer,
                            stop_strings=chat_stop_strings(model_name),
                            num_logits_to_keep=1,
                        )[0]
                    else:
                        output = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_gen,
                            num_beams=1,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            tokenizer=tokenizer,
                            stop_strings=chat_stop_strings(model_name),
                            num_logits_to_keep=1,
                        )[0]

            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            result_queue.put(
                (
                    dataset,
                    sample_idx,
                    {
                        "pred": pred,
                        "answers": json_obj.get("answers", []),
                        "all_classes": json_obj.get("all_classes", []),
                        "length": json_obj.get("length"),
                    },
                )
            )
    except Exception as exc:  # pragma: no cover
        import traceback as _tb
        result_queue.put(("__error__", f"worker {worker_id} gpus={gpu_group} ds={dataset if 'dataset' in dir() else '?'}: {exc}\n{_tb.format_exc()}"))


def _run_longbench_multi_gpu(args):
    set_seed(42)

    model_key = args.model.split("_")[0].lower()
    with open("config/model2maxlen.json", "r", encoding="utf-8") as f:
        model2maxlen = json.load(f)
    max_length = int(model2maxlen[model_key])

    with open("config/dataset2maxlen.json", "r", encoding="utf-8") as f:
        dataset2maxlen = json.load(f)

    if not os.path.exists("result_txt/pred"):
        os.makedirs("result_txt/pred")

    # Load TriAttention calibration stats if provided
    triattention_stats_data = None
    if args.triattention_stats:
        import torch as _torch
        triattention_stats_data = _torch.load(args.triattention_stats, map_location="cpu", weights_only=False)
        print(f"[longbench] TriAttention stats loaded from {args.triattention_stats}")

    # Load per-prompt WAITS action table if provided
    waits_table_data = None
    if args.waits_table:
        with open(args.waits_table) as f:
            full_table = json.load(f)
        table_key = f"{model_key}_{int(args.budget)}"
        waits_table_data = full_table.get(table_key)
        if waits_table_data is None:
            raise ValueError(f"--waits_table: key '{table_key}' not found in {args.waits_table}")
        print(f"[longbench] WAITS table loaded: key={table_key}, {len(waits_table_data)} datasets")

    chunk_suffix = f"_chunk{int(args.chunk_size)}" if int(args.chunk_size) > 1 else ""
    if int(args.chunk_size) > 1 and int(args.chunk_group_size) > 1:
        chunk_suffix += f"g{int(args.chunk_group_size)}"
    if args.pyramid_kv:
        chunk_suffix += "_pyr"
    if args.run_name:
        run_name = args.run_name
    elif args.waits_table:
        run_name = f"{args.model}_WAITS_{args.budget}"
    else:
        run_name = f"{args.model}_{args.method}_{args.window}_{args.budget}{chunk_suffix}"
    output_dir = f"result_txt/pred/{int(args.budget)}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    longbench_dir = os.path.join("datasets", "longbench")
    tasks = _build_tasks(longbench_dir, dataset2maxlen)
    if not tasks:
        print("No LongBench tasks found. Exiting.")
        return

    import random as _random
    _random.Random(42).shuffle(tasks)
    if args.shard_count > 1:
        n_total = len(tasks)
        if args.shard_weights:
            # GPU-speed-proportional split (e.g. 3090:4090 = 1:1.5). Each shard gets a
            # contiguous slice of the shuffled list sized by its weight fraction; slices
            # are disjoint and cover all tasks because every shard shares the seed-42 shuffle.
            w = [float(x) for x in args.shard_weights.split(",")]
            if len(w) != args.shard_count:
                raise ValueError(f"--shard_weights has {len(w)} entries, expected shard_count={args.shard_count}")
            cum = [0.0]
            for x in w:
                cum.append(cum[-1] + x)
            lo = int(round(n_total * cum[args.shard_id] / cum[-1]))
            hi = int(round(n_total * cum[args.shard_id + 1] / cum[-1]))
            tasks = tasks[lo:hi]
            print(f"[longbench] weighted shard {args.shard_id+1}/{args.shard_count} "
                  f"(w={w[args.shard_id]}): {len(tasks)}/{n_total} tasks")
        else:
            tasks = [t for i, t in enumerate(tasks) if i % args.shard_count == args.shard_id]
            print(f"[longbench] shard {args.shard_id+1}/{args.shard_count}: {len(tasks)}/{n_total} tasks")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-GPU LongBench evaluation.")

    visible_gpu_count = int(torch.cuda.device_count())
    gpus_per_model = int(args.gpus_per_model)
    if gpus_per_model <= 0:
        raise ValueError("--gpus_per_model must be >= 1")
    if gpus_per_model > visible_gpu_count:
        raise ValueError(
            f"--gpus_per_model ({gpus_per_model}) cannot exceed visible GPU count ({visible_gpu_count})"
        )

    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if parent_visible:
        all_gpu_ids = [int(g) for g in parent_visible.split(",") if g.strip()]
    else:
        all_gpu_ids = list(range(visible_gpu_count))
    gpu_groups = [
        all_gpu_ids[start : start + gpus_per_model]
        for start in range(0, len(all_gpu_ids), gpus_per_model)
        if all_gpu_ids[start : start + gpus_per_model]
    ]

    print(f"[longbench] visible_gpu_count={visible_gpu_count}, gpus_per_model={gpus_per_model}")
    print(f"[longbench] gpu_groups={gpu_groups}")

    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()

    for t in tasks:
        task_queue.put(t)
    for _ in range(len(gpu_groups)):
        task_queue.put(None)

    processes = []
    for worker_id, gpu_group in enumerate(gpu_groups):
        p = ctx.Process(
            target=_longbench_worker,
            args=(
                worker_id,
                gpu_group,
                model_key,
                args.method,
                int(args.window),
                int(args.budget),
                max_length,
                dataset2maxlen,
                task_queue,
                result_queue,
                args.aj_weight_fn,
                float(args.aj_offset),
                int(args.recent_budget),
                int(args.n_sink),
                bool(args.ada_kv),
                int(args.chunk_size),
                int(args.chunk_group_size),
                bool(args.pyramid_kv),
                float(args.pyramid_ratio),
                waits_table_data,
                int(args.min_prompt_tokens),
                int(args.max_prompt_tokens),
                args.sigmoid_a,
                str(args.curve),
                triattention_stats_data,
                args.key_prior,
                args.value_weight,
            ),
        )
        p.start()
        processes.append(p)

    remaining = len(tasks)
    done = 0
    started_at = time.time()
    print(f"[longbench] total tasks: {remaining}")

    while remaining > 0:
        item = result_queue.get()
        if item[0] == "__error__":
            raise RuntimeError(item[1])

        if item[0] == "__skip__":
            remaining -= 1
            continue

        dataset, sample_idx, payload = item
        out_path = os.path.join(output_dir, f"{dataset}.jsonl")
        with open(out_path, "a", encoding="utf-8") as f:
            # include sample idx so cross-server/sharded outputs can be re-aligned
            # (build_lb_index_from_all.py keys on "idx"); harmless extra field for scoring.
            json.dump({"idx": sample_idx, **payload}, f, ensure_ascii=False)
            f.write("\n")

        remaining -= 1
        done += 1
        elapsed = max(1e-6, time.time() - started_at)
        rate = done / elapsed
        pct = 100.0 * done / len(tasks)
        print(
            f"[longbench] progress: {done}/{len(tasks)} ({pct:.1f}%) | "
            f"{rate:.2f} samples/s | ETA {int((len(tasks)-done)/rate) if rate>0 else 0}s",
            flush=True,
        )

    for p in processes:
        p.join()

    print("\nLongBench evaluation completed!")


if __name__ == "__main__":
    args = parse_args()
    _run_longbench_multi_gpu(args)

