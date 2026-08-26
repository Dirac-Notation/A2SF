"""Rename completed technique pred dirs to canonical backup names and move them to
result_txt/backup/<model>/<budget>/<canonical>. Idempotent + safe:
  - only moves dirs that HAVE result.json (complete);
  - NEVER overwrites an existing backup dir (read-only rule) -> skip+warn;
  - only moves CANONICAL/faithful runs via an explicit allowlist mapping. Non-canonical
    (recent16 keydiff/l2/triatt, window32 chunkkv/keyformer, budget128 pyramid, *_eslab19,
    *_sink, RL/experiment dirs) are IGNORED, never moved.
Run repeatedly during the dispatch campaign; the dispatcher also calls it at the end."""
import os, re, shutil, glob

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = f"{REPO}/result_txt/pred"
MODELS = ["llama3-1b", "llama3-8b", "qwen2", "mistral-7b"]
MRE = "(llama3-1b|llama3-8b|qwen2|mistral-7b)"


def canonical(run):
    """pred dir name -> (model, canonical_backup_name) or None to skip."""
    # dispatcher-produced (already well-named): KeyDiff / TriAttention vanilla
    m = re.fullmatch(rf"keydiff_vanilla_{MRE}_(\d+)", run);          # faithful vanilla
    if m: return m.group(1), f"{m.group(1)}_KeyDiff_{m.group(2)}"
    m = re.fullmatch(rf"triattention_vanilla_{MRE}_(\d+)", run)
    if m: return m.group(1), f"{m.group(1)}_TriAttention_{m.group(2)}"
    # scorer x selector crosses produced by dispatcher: ChunkKV-/AdaKV-/PyramidKV-<scorer>_<model>_<bud>
    m = re.fullmatch(rf"(ChunkKV|AdaKV|PyramidKV)-(TOVA|SnapKV|H2O|WAITS)_{MRE}_(\d+)", run)
    if m: return m.group(3), f"{m.group(3)}_{m.group(1)}-{m.group(2)}_{m.group(4)}"
    # previously-completed canonical runs
    m = re.fullmatch(rf"streamingllm_{MRE}_(\d+)", run)
    if m: return m.group(1), f"{m.group(1)}_StreamingLLM_{m.group(2)}"
    m = re.fullmatch(rf"chunkkv16_{MRE}_(\d+)", run)                  # ChunkKV + SnapKV scorer
    if m: return m.group(1), f"{m.group(1)}_ChunkKV-SnapKV_{m.group(2)}"
    m = re.fullmatch(rf"kvzip_{MRE}_(\d+)", run)
    if m: return m.group(1), f"{m.group(1)}_KVZip_{m.group(2)}"
    # Keyformer intentionally NOT mapped: excluded from the comparison table (user request).
    # 8b ChunkKV cross (old naming <model>_<scorer>_Chunk_<bud>); exclude *_eslab19 dupes
    m = re.fullmatch(rf"{MRE}_(TOVA|SnapKV|H2O|WAITS)_Chunk_(\d+)", run)
    if m: return m.group(1), f"{m.group(1)}_ChunkKV-{m.group(2)}_{m.group(3)}"
    return None  # everything else: ignore (non-canonical / non-technique)


def main():
    moved = skipped = incomplete = ignored = 0
    for bud_dir in sorted(glob.glob(f"{PRED}/*")):
        if not os.path.isdir(bud_dir):
            continue
        budget = os.path.basename(bud_dir)
        for run_dir in sorted(glob.glob(f"{bud_dir}/*")):
            if not os.path.isdir(run_dir):
                continue
            run = os.path.basename(run_dir)
            c = canonical(run)
            if c is None:
                ignored += 1
                continue
            if not os.path.exists(f"{run_dir}/result.json"):
                incomplete += 1
                continue
            model, name = c
            dest = f"{REPO}/result_txt/backup/{model}/{budget}/{name}"
            if os.path.exists(dest):
                print(f"[backup] SKIP {run}: dest {model}/{budget}/{name} exists (read-only)")
                skipped += 1
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(run_dir, dest)
            print(f"[backup] MOVED {budget}/{run} -> backup/{model}/{budget}/{name}")
            moved += 1
    print(f"[backup] done: moved={moved} skipped(exists)={skipped} "
          f"incomplete={incomplete} ignored(non-canonical)={ignored}")


if __name__ == "__main__":
    main()
