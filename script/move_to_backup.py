"""Move COMPLETED prediction dirs (those with result.json) from result_txt/pred/<budget>/
to result_txt/backup/<model>/<budget>/<run>/. Never overwrites an existing backup dir
(read-only rule); skips + warns instead. Run after an experiment batch finishes."""
import os, shutil, glob

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["llama3-1b", "llama3-8b", "qwen2-0.5b", "qwen2", "mistral-7b"]   # longest-match first
PRED = f"{REPO}/result_txt/pred"

moved = skipped = incomplete = 0
for bud_dir in sorted(glob.glob(f"{PRED}/*")):
    if not os.path.isdir(bud_dir):
        continue
    budget = os.path.basename(bud_dir)
    for run_dir in sorted(glob.glob(f"{bud_dir}/*")):
        if not os.path.isdir(run_dir):
            continue
        run = os.path.basename(run_dir)
        if not os.path.exists(f"{run_dir}/result.json"):
            incomplete += 1
            continue
        model = next((m for m in MODELS if m in run), None)
        if model is None:
            print(f"[backup] SKIP {run}: no known model in name")
            skipped += 1
            continue
        dest = f"{REPO}/result_txt/backup/{model}/{budget}/{run}"
        if os.path.exists(dest):
            print(f"[backup] SKIP {run}: backup dir already exists (read-only)")
            skipped += 1
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(run_dir, dest)
        print(f"[backup] MOVED {budget}/{run} -> backup/{model}/{budget}/")
        moved += 1

print(f"[backup] done: moved={moved} skipped={skipped} incomplete(no result.json)={incomplete}")
