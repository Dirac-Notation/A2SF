import os
import json
import argparse
import multiprocessing as mp
import time

from tqdm import tqdm
import torch

from utils import load_model, set_seed, CompressionConfig


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="LongBench end-to-end evaluation (multi-GPU, multi-process)")
    parser.add_argument("--model", type=str, required=True, choices=["llama3-8b", "llama3-1b", "qwen2"])
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

            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                if "llama" in model_name:
                    prompt = f"[INST]{prompt}[/INST]"

            encoded = tokenizer(prompt, truncation=False, return_tensors="pt")
            input_ids = encoded.input_ids.to(model.device)
            attention_mask = encoded.attention_mask.to(torch.bfloat16).to(model.device)
            context_length = int(input_ids.shape[-1])

            max_gen = int(dataset2maxlen.get(dataset, 64))

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
                        stop_strings="[/INST]",
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
                        stop_strings="[/INST]",
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
        result_queue.put(("__error__", f"worker {worker_id} gpus={gpu_group}: {exc}"))


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

    output_dir = f"result_txt/pred/{args.model}_{args.method}_{args.window}_{args.budget}"
    os.makedirs(output_dir, exist_ok=True)

    longbench_dir = os.path.join("datasets", "longbench")
    tasks = _build_tasks(longbench_dir, dataset2maxlen)
    if not tasks:
        print("No LongBench tasks found. Exiting.")
        return

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

    all_gpu_ids = list(range(visible_gpu_count))
    gpu_groups = []
    for start in range(0, visible_gpu_count, gpus_per_model):
        group = all_gpu_ids[start : start + gpus_per_model]
        if group:
            gpu_groups.append(group)

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

        dataset, sample_idx, payload = item
        out_path = os.path.join(output_dir, f"{dataset}.jsonl")
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
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

