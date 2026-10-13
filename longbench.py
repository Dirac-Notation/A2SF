"""LongBench evaluation with KV-cache compression (multi-GPU, one model instance per GPU group).

    python longbench.py --model llama3-8b --method qtrust --budget 128 --tau 1
    python longbench.py --model llama3-8b --method snap --window 16 --budget 128        # SnapKV
    python longbench_eval.py result_txt/pred/128/<run_name>                             # score

Predictions go to result_txt/pred/<budget>/<run_name>/<dataset>.jsonl. See README.md for every paper run.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import time

import torch

from utils import load_model, set_seed, CompressionConfig, build_chat_prompt, chat_stop_strings

METHODS = ["full", "qtrust", "snap", "streamingllm", "keydiff", "l2norm", "triattention", "kvzip"]


def parse_args(args=None):
    p = argparse.ArgumentParser(description="LongBench evaluation with KV-cache compression")
    p.add_argument("--model", required=True, help="shortname in config/model2path.json")
    p.add_argument("--method", default="full", choices=METHODS,
                   help="snap = accumulated attention of the last --window queries: SnapKV (16), TOVA (1), H2O (32768)")
    p.add_argument("--window", type=int, default=16, help="observation window W (snap / qtrust)")
    p.add_argument("--budget", type=int, default=128, help="KV entries kept per layer and KV head")
    p.add_argument("--recent_budget", type=int, default=16, help="last tokens always kept")
    p.add_argument("--n_sink", type=int, default=0, help="first tokens always kept")
    p.add_argument("--ada_kv", action="store_true", help="Ada-KV head-wise budget allocation on top of any scorer")
    p.add_argument("--score_pool", type=int, default=0, help="max-pool the scores over keys (SnapKV uses 7); 0 = off")
    p.add_argument("--tau", type=float, default=0.0, help="(qtrust) skip queries with weight < tau / N; 0 = dense")
    p.add_argument("--temperature", type=float, default=1.0, help="(qtrust) query weight w ∝ s^(1/T)")
    p.add_argument("--window_one", action="store_true",
                   help="(qtrust) weight 1 for each request query (paper Eq. 6); default: request queries weighted by s")
    p.add_argument("--kernel", action="store_true", help="(qtrust) Triton kernels reusing the FlashAttention-2 LSE")
    p.add_argument("--window_span", choices=["full", "last"], default=None,
                   help="(ablation) observation window = the request span of each prompt "
                        "(datasets/longbench_spans; full request or its last sentence)")
    p.add_argument("--triattention_stats", default=None, help="(triattention) calibration stats .pt file")
    p.add_argument("--gpus_per_model", type=int, default=1, help="GPUs per model instance")
    p.add_argument("--shard_count", type=int, default=1, help="split the tasks over several machines")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--run_name", default=None, help="output directory name (default: derived from the options)")
    return p.parse_args(args)


def default_run_name(a):
    name = f"{a.model}_{a.method}_{a.window}_{a.budget}"
    if a.ada_kv:
        name += "_ada"
    if a.n_sink:
        name += f"_nsink{a.n_sink}"
    if a.temperature != 1.0:
        name += f"_T{a.temperature:g}"
    if a.window_one:
        name += "_win1"
    if a.score_pool > 1:
        name += f"_pool{a.score_pool}"
    if a.window_span:
        name += f"_span{a.window_span}"
    if a.kernel:
        name += "_kern"
    if a.tau > 0:
        name += f"_tau{a.tau:g}"
    return name


def load_tasks(longbench_dir, dataset2maxlen):
    tasks = []
    for dataset in dataset2maxlen:
        path = os.path.join(longbench_dir, f"{dataset}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for idx, line in enumerate(l for l in f if l.strip()):
                tasks.append({"dataset": dataset, "sample_idx": idx, "json_obj": json.loads(line)})
    return tasks


def worker(worker_id, gpu_group, a, max_length, dataset2maxlen, task_queue, result_queue, tri_stats):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_group)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_grad_enabled(False)
    model, tokenizer = load_model(a.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = CompressionConfig(compression_method=a.method, observation_window=a.window, total_budget=a.budget,
                            recent_budget=a.recent_budget, n_sink=a.n_sink, ada_kv=a.ada_kv,
                            score_pool=a.score_pool, tau=a.tau, temperature=a.temperature,
                            window_one=a.window_one, kernel=a.kernel)
    if a.method == "triattention":
        cfg["triattention_stats"] = tri_stats
    spans = {}
    dataset = "?"
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            dataset, sample_idx, obj = task["dataset"], int(task["sample_idx"]), task["json_obj"]
            prompt = obj["input_prompt"]
            ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(ids) > max_length:                                   # truncate in the middle
                half = int(max_length / 2)
                prompt = (tokenizer.decode(ids[:half], skip_special_tokens=True)
                          + tokenizer.decode(ids[-half:], skip_special_tokens=True))
            prompt = build_chat_prompt(prompt, a.model, tokenizer, dataset)

            if a.window_span:
                # observation window = token length of this prompt's request span (fallback: --window)
                if dataset not in spans:
                    with open(os.path.join("datasets", "longbench_spans", f"{dataset}.jsonl"), encoding="utf-8") as sf:
                        spans[dataset] = [json.loads(l)[a.window_span] for l in sf]
                key = spans[dataset][sample_idx].strip()[:40]
                c0 = prompt.rfind(key) if key else -1
                if c0 >= 0:
                    offs = tokenizer(prompt, truncation=False, return_offsets_mapping=True)["offset_mapping"]
                    cfg["observation_window"] = max(1, sum(1 for s, e in offs if s >= c0 and e > s))
                else:
                    cfg["observation_window"] = a.window

            enc = tokenizer(prompt, truncation=False, return_tensors="pt")
            input_ids = enc.input_ids.to(model.device)
            attention_mask = enc.attention_mask.to(torch.bfloat16).to(model.device)
            ctx_len = int(input_ids.shape[-1])
            max_gen = int(dataset2maxlen.get(dataset, 64))

            if a.method == "kvzip":
                from kvcompress.kvzip import kvzip_generate
                stop_ids = [tokenizer.eos_token_id]
                if dataset == "samsum":
                    stop_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])
                gen = kvzip_generate(model, tokenizer, input_ids, max_new_tokens=max_gen, budget=a.budget,
                                     recent_budget=a.recent_budget, n_sink=a.n_sink,
                                     stop_token_ids=stop_ids).to(input_ids.device)
                output = torch.cat([input_ids[0], gen])
            else:
                model.init_cache(cfg)
                gen_kw = dict(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_gen,
                              num_beams=1, do_sample=False, pad_token_id=tokenizer.eos_token_id, tokenizer=tokenizer,
                              stop_strings=chat_stop_strings(a.model), num_logits_to_keep=1)
                if dataset == "samsum":                                 # LongBench's samsum setting
                    gen_kw.update(min_length=ctx_len + 1,
                                  eos_token_id=[tokenizer.eos_token_id,
                                                tokenizer.encode("\n", add_special_tokens=False)[-1]])
                with torch.inference_mode():
                    output = model.generate(**gen_kw)[0]

            pred = tokenizer.decode(output[ctx_len:], skip_special_tokens=True)
            result_queue.put((dataset, sample_idx, {"pred": pred, "answers": obj.get("answers", []),
                                                    "all_classes": obj.get("all_classes", []),
                                                    "length": obj.get("length")}))
    except Exception as exc:  # pragma: no cover
        import traceback
        result_queue.put(("__error__", f"worker {worker_id} gpus={gpu_group} ds={dataset}: {exc}\n{traceback.format_exc()}"))


def main(a):
    set_seed(42)
    max_length = int(json.load(open("config/model2maxlen.json"))[a.model])
    dataset2maxlen = json.load(open("config/dataset2maxlen.json"))
    tri_stats = torch.load(a.triattention_stats, map_location="cpu", weights_only=False) if a.triattention_stats else None
    if a.method == "triattention" and tri_stats is None:
        raise ValueError("--method triattention needs --triattention_stats")

    run_name = a.run_name or default_run_name(a)
    output_dir = f"result_txt/pred/{a.budget}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    tasks = load_tasks(os.path.join("datasets", "longbench"), dataset2maxlen)
    if not tasks:
        raise SystemExit("No LongBench data found under datasets/longbench (run datasets/make_longbench_dataset.py)")
    random.Random(42).shuffle(tasks)
    if a.shard_count > 1:
        tasks = [t for i, t in enumerate(tasks) if i % a.shard_count == a.shard_id]

    n_gpu = torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    gpu_ids = [int(g) for g in visible.split(",") if g.strip()] if visible else list(range(n_gpu))
    groups = [gpu_ids[s:s + a.gpus_per_model] for s in range(0, len(gpu_ids), a.gpus_per_model)
              if len(gpu_ids[s:s + a.gpus_per_model]) == a.gpus_per_model]
    print(f"[longbench] {run_name}: {len(tasks)} tasks on GPU groups {groups}")

    ctx = mp.get_context("spawn")
    task_queue, result_queue = ctx.Queue(), ctx.Queue()
    for t in tasks:
        task_queue.put(t)
    for _ in groups:
        task_queue.put(None)
    procs = [ctx.Process(target=worker, args=(i, g, a, max_length, dataset2maxlen, task_queue, result_queue, tri_stats))
             for i, g in enumerate(groups)]
    for p in procs:
        p.start()

    start = time.time()
    for done in range(1, len(tasks) + 1):
        item = result_queue.get()
        if item[0] == "__error__":
            raise RuntimeError(item[1])
        dataset, sample_idx, payload = item
        with open(os.path.join(output_dir, f"{dataset}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"idx": sample_idx, **payload}, ensure_ascii=False) + "\n")
        rate = done / max(1e-6, time.time() - start)
        print(f"[longbench] {done}/{len(tasks)} | {rate:.2f} samples/s | ETA {int((len(tasks) - done) / rate)}s", flush=True)
    for p in procs:
        p.join()
    print(f"done: {output_dir}")


if __name__ == "__main__":
    main(parse_args())
