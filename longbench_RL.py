import os
import json
import argparse
import multiprocessing as mp
import time

from tqdm import tqdm
import torch

from utils import set_seed
from RL.a2sf_model import A2SFModel, ModelConfig
from longbench_eval import dataset2metric


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="LongBench RL evaluation (multi-GPU, multi-process)")
    parser.add_argument("--model", type=str, required=True, choices=["llama3-8b", "llama3-1b", "qwen2"])
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--rl_checkpoint", type=str, required=True, help="Path to RL model checkpoint (.pt file)")
    parser.add_argument("--gpus_per_model", type=int, default=1, help="한 모델 인스턴스가 사용할 GPU 개수 (연속된 ID 그룹).")
    return parser.parse_args(args)


def load_jsonl_file(file_path: str):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def _build_tasks(longbench_dir: str, dataset2maxlen: dict) -> list:
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


def _load_rl_model(model_name: str, checkpoint_path: str) -> A2SFModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("agent_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("policy_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint missing 'agent_state_dict' (and legacy 'policy_state_dict').")
    arch_config = checkpoint.get("arch_config")

    model_cfg = ModelConfig(model=model_name)
    model = A2SFModel(
        config=model_cfg,
        state_dict=state_dict,
        arch_config=arch_config,
    )
    return model


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
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_group)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_grad_enabled(False)

    print(f"[longbench_RL][worker {worker_id}] start with CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    model = _load_rl_model(model_name, rl_checkpoint)
    tokenizer = model.model_runner.tokenizer
    print(f"[longbench_RL][worker {worker_id}] model base device={model.model_runner.model.device}")
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
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                    tokenized_prompt[-half:], skip_special_tokens=True
                )

            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                if "llama" in model_name:
                    prompt = f"[INST]{prompt}[/INST]"

            encoded = tokenizer(prompt, truncation=False, return_tensors="pt")
            input_ids = encoded.input_ids
            context_length = int(input_ids.shape[-1])
            metric_fn = dataset2metric.get(dataset)
            metric_type = metric_fn.__name__ if metric_fn is not None else "qa_f1_score"

            max_gen = int(dataset2maxlen.get(dataset, 64))

            if dataset == "samsum":
                out = model.generate(
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
                    min_length=context_length + 1,
                    eos_token_id=[
                        tokenizer.eos_token_id,
                        tokenizer.encode("\n", add_special_tokens=False)[-1],
                    ],
                    num_logits_to_keep=1,
                )
            else:
                out = model.generate(
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

            a_val = out.info.get("a")
            b_val = out.info.get("b")
            b_write = int(round(b_val)) if isinstance(b_val, (float, int)) else b_val

            result_queue.put(
                (
                    dataset,
                    sample_idx,
                    {
                        "pred": out.pred_text,
                        "answers": json_obj.get("answers", []),
                        "all_classes": json_obj.get("all_classes", []),
                        "length": json_obj.get("length"),
                        "a": a_val,
                        "b": b_write,
                    },
                )
            )
    except Exception as exc:  # pragma: no cover
        result_queue.put(("__error__", f"worker {worker_id} gpus={gpu_group}: {exc}"))


def _run_longbench_rl_multi_gpu(args):
    set_seed(42)

    model_key = args.model.split("_")[0].lower()
    with open("config/model2maxlen.json", "r", encoding="utf-8") as f:
        model2maxlen = json.load(f)
    max_length = int(model2maxlen[model_key])

    with open("config/dataset2maxlen.json", "r", encoding="utf-8") as f:
        dataset2maxlen = json.load(f)

    if not os.path.exists("result_txt/pred"):
        os.makedirs("result_txt/pred")

    output_dir = f"result_txt/pred/{args.model}_sigmoid_{args.budget}_RL"
    os.makedirs(output_dir, exist_ok=True)

    longbench_dir = os.path.join("datasets", "longbench")
    tasks = _build_tasks(longbench_dir, dataset2maxlen)
    if not tasks:
        print("No LongBench tasks found. Exiting.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-GPU RL LongBench evaluation.")

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

    print(f"[longbench_RL] visible_gpu_count={visible_gpu_count}, gpus_per_model={gpus_per_model}")
    print(f"[longbench_RL] gpu_groups={gpu_groups}")

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
            target=_rl_worker,
            args=(
                worker_id,
                gpu_group,
                model_key,
                args.rl_checkpoint,
                int(args.budget),
                max_length,
                dataset2maxlen,
                task_queue,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    remaining = len(tasks)
    done = 0
    started_at = time.time()
    print(f"[longbench_RL] total tasks: {remaining}")

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
            f"[longbench_RL] progress: {done}/{len(tasks)} ({pct:.1f}%) | "
            f"{rate:.2f} samples/s | ETA {int((len(tasks)-done)/rate) if rate>0 else 0}s",
            flush=True,
        )

    for p in processes:
        p.join()

    print("\nRL LongBench evaluation completed!")


if __name__ == "__main__":
    args = parse_args()
    _run_longbench_rl_multi_gpu(args)

