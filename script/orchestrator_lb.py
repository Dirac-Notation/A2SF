"""Sample-level global-queue orchestrator for LongBench evals across servers.

Replaces static sharding (longbench.py --shard_count/--shard_weights): one local
queue of ALL samples (longest first) + one persistent worker per GPU
(script/worker_lb.py, model loaded once, ssh stdin/stdout pipe as transport,
ACK-driven pull). Fast GPUs naturally pull more, so the 3090:4090 = 1:1.5
weighting becomes unnecessary and no GPU idles until the queue drains.

Method guide: script/GLOBAL_QUEUE.md (adapted from kvpress/experiments).

Usage (from eslab17, the management host):
  python script/orchestrator_lb.py --model llama3-8b --waits_table runs/waits_tables/waits_llama3-8b_u5b.json \
      --budget 128 --run_name 8b_WAITS_gq \
      --local_gpus 0,1,2,3,4,5,6,7 \
      --remote eslab18:0,1,2,3,4,5,6,7 --remote eslab19:0,1,2,3,4,5,6,7

On completion the orchestrator rsyncs each remote's out_dir back and cat-merges
(dedupe by idx) into the local result_txt/pred/<budget>/<run_name>/ so
longbench_eval.py scores it directly.
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
REMOTE_REPO = "/home/smp9898/A2SF"
REMOTE_PY = "$HOME/miniconda3/envs/A2SF/bin/python"

WORKER_PASSTHROUGH = [
    ("--method", "method"), ("--window", "window"), ("--budget", "budget"),
    ("--sigmoid_a", "sigmoid_a"), ("--curve", "curve"),
    ("--recent_budget", "recent_budget"), ("--n_sink", "n_sink"),
    ("--chunk_size", "chunk_size"), ("--chunk_group_size", "chunk_group_size"),
    ("--pyramid_ratio", "pyramid_ratio"), ("--waits_table", "waits_table"),
    ("--key_prior", "key_prior"), ("--value_weight", "value_weight"),
]
WORKER_FLAGS = [("--ada_kv", "ada_kv"), ("--pyramid_kv", "pyramid_kv")]


def build_tasks(limit=0):
    with open("config/dataset2maxlen.json") as f:
        datasets = list(json.load(f).keys())
    tasks = []
    for ds in datasets:
        path = f"datasets/longbench/{ds}.jsonl"
        if not os.path.exists(path):
            print(f"[orch] WARNING: {path} missing, skipping", flush=True)
            continue
        with open(path) as f:
            for idx, line in enumerate(f):
                tasks.append((ds, idx, json.loads(line).get("length", 0)))
    tasks.sort(key=lambda t: -t[2])          # longest first -> tiny tail
    tasks = [(ds, idx) for ds, idx, _ in tasks]
    return tasks[:limit] if limit else tasks


def worker_args_str(args):
    parts = ["--model", args.model, "--out_dir", args.out_dir]
    for flag, attr in WORKER_PASSTHROUGH:
        v = getattr(args, attr)
        if v is not None:
            parts += [flag, str(v)]
    for flag, attr in WORKER_FLAGS:
        if getattr(args, attr):
            parts.append(flag)
    return parts


def worker_thread(name, cmd, task_q, stats, lock, log_path):
    err = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=err, text=True, bufsize=1)
    ready = False
    for line in proc.stdout:
        if line.strip() == "READY_WORKER":
            ready = True
            break
    if not ready:
        print(f"[{name}] FAILED TO START (see {log_path})", flush=True)
        return
    print(f"[{name}] ready", flush=True)
    while True:
        try:
            ds, idx = task_q.get_nowait()
        except queue.Empty:
            break
        try:
            proc.stdin.write(f"{ds}\t{idx}\n")
            proc.stdin.flush()
        except Exception:
            print(f"[{name}] DIED (pipe) on {ds}:{idx}, requeued", flush=True)
            task_q.put((ds, idx))
            return
        acked = False
        for line in proc.stdout:
            if line.startswith("ACK"):
                acked = True
                break
        if not acked:
            print(f"[{name}] DIED on {ds}:{idx}, requeued (see {log_path})", flush=True)
            task_q.put((ds, idx))
            return
        with lock:
            stats["done"] += 1
            stats[name] = stats.get(name, 0) + 1
            if stats["done"] % 100 == 0:
                el = time.time() - stats["t0"]
                print(f"PROGRESS {stats['done']}/{stats['total']} "
                      f"({el/60:.0f} min, {stats['done']/el:.2f}/s)", flush=True)
    try:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.wait(timeout=60)
    except Exception:
        proc.kill()


def preflight(args, remotes):
    """Sync code + table to remotes; verify data and model presence."""
    with open("config/model2path.json") as f:
        model_path = json.load(f)[args.model.split("_")[0].lower()]
    files = "script/worker_lb.py utils.py longbench_eval.py".split()
    for host, _ in remotes:
        subprocess.run(["rsync", "-az"] + files + [f"{host}:{REMOTE_REPO}/"], check=True)
        subprocess.run(["rsync", "-az", "utils_real_drop/", f"{host}:{REMOTE_REPO}/utils_real_drop/"], check=True)
        subprocess.run(["rsync", "-az", "config/", f"{host}:{REMOTE_REPO}/config/"], check=True)
        subprocess.run(["ssh", host, f"mkdir -p {REMOTE_REPO}/script"], check=True)
        subprocess.run(["rsync", "-az", "script/worker_lb.py", f"{host}:{REMOTE_REPO}/script/"], check=True)
        if args.waits_table:
            subprocess.run(["ssh", host, f"mkdir -p {REMOTE_REPO}/{os.path.dirname(args.waits_table)}"], check=True)
            subprocess.run(["rsync", "-az", args.waits_table,
                            f"{host}:{REMOTE_REPO}/{args.waits_table}"], check=True)
        n = subprocess.run(["ssh", host, f"ls {REMOTE_REPO}/datasets/longbench/*.jsonl 2>/dev/null | wc -l"],
                           capture_output=True, text=True).stdout.strip()
        if model_path.startswith("/"):
            probe = f"test -e {model_path}"
        else:  # HF hub id -> look in the hub cache
            cache_name = "models--" + model_path.replace("/", "--")
            probe = (f"test -d $HOME/.cache/huggingface/hub/{cache_name} "
                     f"|| test -d /data2/shared/huggingface_cache/hub/{cache_name}")
        ok = subprocess.run(["ssh", host, f"{probe} && echo OK || echo MISSING"],
                            capture_output=True, text=True).stdout.strip()
        print(f"[preflight] {host}: longbench files={n}, model={ok}", flush=True)
        if n == "0" or ok != "OK":
            raise RuntimeError(f"{host}: data or model missing — sync datasets/longbench "
                               f"and the model ({model_path}) first")


def collect(args, remotes):
    """Rsync remote out_dirs home, then merge with dedupe by idx."""
    merged = {}
    srcs = [args.out_dir]
    for host, _ in remotes:
        dst = os.path.join(args.out_dir, f"_from_{host}")
        os.makedirs(dst, exist_ok=True)
        subprocess.run(["rsync", "-az", f"{host}:{REMOTE_REPO}/{args.out_dir}/", dst + "/"], check=True)
        srcs.append(dst)
    for src in srcs:
        for fn in os.listdir(src):
            if not fn.endswith(".jsonl"):
                continue
            ds = fn[:-6]
            with open(os.path.join(src, fn)) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        merged.setdefault(ds, {})[r["idx"]] = line
    total = 0
    for ds, rows in merged.items():
        with open(os.path.join(args.out_dir, f"{ds}.jsonl"), "w") as f:
            for idx in sorted(rows):
                f.write(rows[idx])
        total += len(rows)
    print(f"[collect] merged {total} samples across {len(merged)} datasets -> {args.out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", default="waits")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--sigmoid_a", type=float, default=None)
    ap.add_argument("--curve", default="sigmoid")
    ap.add_argument("--recent_budget", type=int, default=16)
    ap.add_argument("--n_sink", type=int, default=0)
    ap.add_argument("--ada_kv", action="store_true")
    ap.add_argument("--chunk_size", type=int, default=0)
    ap.add_argument("--chunk_group_size", type=int, default=1)
    ap.add_argument("--pyramid_kv", action="store_true")
    ap.add_argument("--pyramid_ratio", type=float, default=4.0)
    ap.add_argument("--waits_table", default=None)
    ap.add_argument("--key_prior", default=None)
    ap.add_argument("--value_weight", type=float, default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--local_gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--remote", action="append", default=[],
                    help="host:gpu,gpu,...  (repeatable, e.g. --remote eslab19:0,1,2,3)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only first N samples")
    ap.add_argument("--no_preflight", action="store_true")
    ap.add_argument("--no_collect", action="store_true")
    args = ap.parse_args()

    run_name = args.run_name or (
        f"{args.model}_WAITS_{args.budget}_gq" if args.waits_table
        else f"{args.model}_{args.method}_{args.window}_{args.budget}_gq")
    args.out_dir = f"result_txt/pred/{int(args.budget)}/{run_name}"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("logs/gq", exist_ok=True)

    remotes = []
    for spec in args.remote:
        host, gpus = spec.split(":")
        remotes.append((host, gpus.split(",")))

    if not args.no_preflight and remotes:
        preflight(args, remotes)

    tasks = build_tasks(args.limit)
    task_q = queue.Queue()
    for t in tasks:
        task_q.put(t)
    stats = {"done": 0, "total": len(tasks), "t0": time.time()}
    lock = threading.Lock()
    print(f"{len(tasks)} samples queued (longest first) -> {args.out_dir}", flush=True)

    wargs = worker_args_str(args)
    threads = []
    for g in [x for x in args.local_gpus.split(",") if x != ""]:
        cmd = ["env", f"CUDA_VISIBLE_DEVICES={g}", "HF_HUB_OFFLINE=1",
               sys.executable, "-u", "script/worker_lb.py"] + wargs
        threads.append(threading.Thread(
            target=worker_thread,
            args=(f"L{g}", cmd, task_q, stats, lock, f"logs/gq/{run_name}_L{g}.log")))
    for host, gpus in remotes:
        for g in gpus:
            rcmd = (f"cd {REMOTE_REPO} && CUDA_VISIBLE_DEVICES={g} TOKENIZERS_PARALLELISM=false "
                    f"HF_HUB_OFFLINE=1 {REMOTE_PY} -u script/worker_lb.py " + " ".join(wargs))
            cmd = ["ssh", host, rcmd]
            threads.append(threading.Thread(
                target=worker_thread,
                args=(f"{host[-2:]}g{g}", cmd, task_q, stats, lock,
                      f"logs/gq/{run_name}_{host}_{g}.log")))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    el = time.time() - stats["t0"]
    print(f"ALL_SAMPLES_DONE {stats['done']}/{stats['total']} in {el/60:.1f} min", flush=True)
    print("per-worker:", {k: v for k, v in sorted(stats.items())
                          if k not in ("done", "total", "t0")}, flush=True)

    if not args.no_collect and remotes:
        collect(args, remotes)


if __name__ == "__main__":
    main()
