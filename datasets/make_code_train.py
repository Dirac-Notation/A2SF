"""Coherent code-completion training from codeparrot-clean (whole Python files, NOT a
LongBench source). Replaces the degenerate random-function concat (full_cache ~5, flat).
Mirrors lcc/repobench structure: real repo files as context, complete the next line.
  - context = a few whole files (repo-like) + prefix of the target file
  - answer = the next non-trivial line of the target file (natural continuation)

  python datasets/make_code_train.py --model llama3-1b --out datasets/training/synth/code_v2.jsonl \
      --lengths 2000,4000,8000,12000 --per 120
"""
import argparse, json, os, random, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from transformers import AutoTokenizer
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-1b"); ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="2000,4000,8000,12000"); ap.add_argument("--per", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); random.seed(a.seed)
    m2p = json.load(open(f"{REPO}/config/model2path.json"))
    tok = AutoTokenizer.from_pretrained(m2p[a.model])
    lengths = [int(x) for x in a.lengths.split(",")]
    print("buffering codeparrot-clean files...", flush=True)
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
    files = []
    for ex in ds:
        c = ex.get("content", "")
        nl = c.count("\n")
        if 30 <= nl <= 400 and len(c) > 400: files.append(c)
        if len(files) >= 12000: break
    print(f"files={len(files)}", flush=True)
    rows = []; sid = 0; target = a.per * len(lengths)
    for made in range(target):
        L = lengths[made % len(lengths)]; random.shuffle(files)
        # target file = first; prepend other whole files as repo context until ~L
        tgt = files[0]; lines = tgt.split("\n")
        # cut point in the LAST third of the target file -> coherent recent context
        lo = max(8, int(len(lines) * 0.5)); hi = len(lines) - 2
        if hi <= lo: continue
        j = random.randint(lo, hi); ans = lines[j].rstrip()
        if len(ans.strip()) < 5:
            ans = next((lines[k].rstrip() for k in range(j, len(lines)) if len(lines[k].strip()) >= 5), None)
            if ans is None: continue
        tgt_prefix = "\n".join(lines[:j])
        ntgt = len(tok.encode(tgt_prefix, add_special_tokens=False))
        ctx = []; n = ntgt; i = 1
        while n < L - 120 and i < len(files):
            ctx.append(files[i]); n += len(tok.encode(files[i], add_special_tokens=False)) + 4; i += 1
        body = "\n\n# ---\n\n".join(ctx + [tgt_prefix]) if ctx else tgt_prefix
        body = tok.decode(tok.encode(body, add_special_tokens=False)[-(L - 80):])
        # LB lcc/repobench completion format (NO_CHAT): model continues the code directly.
        prompt = "Please complete the code given below. \n" + body + "\nNext line of code:\n"
        rows.append({"sample_id": sid, "input_prompt": prompt, "answers": [ans], "all_classes": [],
                     "metric_type": "code_sim_score", "task_type": "Code Complete", "dataset": "codeparrot_code",
                     "length": len(tok.encode(prompt)), "generation_length": 64})
        sid += 1
    random.shuffle(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
