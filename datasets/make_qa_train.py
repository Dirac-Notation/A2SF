"""Natural QA training data from SQuAD (NOT a LongBench source). Replaces the degenerate
synthetic multihop/qa (1B can't solve artificial chains). SQuAD answers are natural spans.
  - Single-doc QA: one SQuAD context padded with WikiText filler to length L + question.
  - Multi-doc QA: K SQuAD contexts (distinct titles) concatenated as numbered documents,
    one holds the answer; ask that question. Mimics HotpotQA multi-doc structure (answer
    requires locating the right document), without using any LongBench data.
SQuAD is loaded straight from the cached arrow (datasets<4.0.0 can't parse its 'List' schema).

  python datasets/make_qa_train.py --model llama3-1b --out datasets/training/synth/qa_v1.jsonl \
      --lengths 2000,4000,8000,12000 --per 120
"""
import argparse, json, os, random, sys
import pyarrow as pa, pyarrow.ipc as ipc
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from transformers import AutoTokenizer
from datasets import load_dataset

SQUAD = ("/data2/shared/huggingface_cache/datasets/squad/plain_text/0.0.0/"
         "7b6d24c440a36b6815f21b70d25016731768db1f/squad-train.arrow")

def load_squad():
    with pa.memory_map(SQUAD) as src:
        try: t = ipc.open_stream(src).read_all()
        except Exception:
            src.seek(0); t = ipc.open_file(src).read_all()
    rows = []
    for r in t.to_pylist():
        ans = r["answers"]["text"]
        if ans and len(r["context"]) > 200: rows.append((r["title"], r["context"], r["question"], ans[0]))
    return rows

def collect_filler(tok, n=4000):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    out = []
    for ex in ds:
        t = ex.get("text", "").strip()
        if len(t) < 60 or t.startswith("="): continue
        out.append(t)
        if len(out) >= n: break
    return out

def pad_to(tok, base_text, L, filler):
    cur = len(tok.encode(base_text, add_special_tokens=False)); chunks = []
    while cur < L:
        s = random.choice(filler); chunks.append(s)
        cur += len(tok.encode(s, add_special_tokens=False)) + 1
    return " ".join(chunks)

def gen_single(tok, squad, filler, lengths, per, rows, sid0):
    sid = sid0; target = per * len(lengths)
    for made in range(target):
        L = lengths[made % len(lengths)]; title, ctx, q, ans = random.choice(squad)
        pad = pad_to(tok, ctx, L - 120, filler)
        # answer-bearing context placed mid-document, filler around
        body = pad[:len(pad)//2] + "\n\n" + ctx + "\n\n" + pad[len(pad)//2:]
        body = tok.decode(tok.encode(body, add_special_tokens=False)[:L - 80])
        prompt = ("[INST]Answer the question based on the document.\n\n<document>\n" + body +
                  f"\n</document>\n\nQuestion: {q}\nAnswer:[/INST]")
        rows.append({"sample_id": sid, "input_prompt": prompt, "answers": [ans], "all_classes": [],
                     "metric_type": "qa_f1_score", "task_type": "Single-doc QA", "dataset": "squad_single",
                     "length": len(tok.encode(prompt)), "generation_length": 32})
        sid += 1
    return sid

def gen_multi(tok, squad, filler, lengths, per, rows, sid0):
    sid = sid0; target = per * len(lengths)
    for made in range(target):
        L = lengths[made % len(lengths)]
        title, ctx, q, ans = random.choice(squad)
        # build distractor docs from other titles until ~L tokens
        docs = [ctx]; n = len(tok.encode(ctx, add_special_tokens=False))
        while n < L - 150:
            _, c2, _, _ = random.choice(squad)
            docs.append(c2); n += len(tok.encode(c2, add_special_tokens=False)) + 4
        random.shuffle(docs)
        numbered = "\n\n".join(f"Document {i}: {d}" for i, d in enumerate(docs))
        numbered = tok.decode(tok.encode(numbered, add_special_tokens=False)[:L - 80])
        prompt = ("[INST]You are given several documents. Answer the question using the relevant one.\n\n"
                  + numbered + f"\n\nQuestion: {q}\nAnswer:[/INST]")
        rows.append({"sample_id": sid, "input_prompt": prompt, "answers": [ans], "all_classes": [],
                     "metric_type": "qa_f1_score", "task_type": "Multi-doc QA", "dataset": "squad_multi",
                     "length": len(tok.encode(prompt)), "generation_length": 32})
        sid += 1
    return sid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-1b"); ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="2000,4000,8000,12000"); ap.add_argument("--per", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); random.seed(a.seed)
    m2p = json.load(open(f"{REPO}/config/model2path.json"))
    tok = AutoTokenizer.from_pretrained(m2p[a.model])
    lengths = [int(x) for x in a.lengths.split(",")]
    print("loading squad + filler...", flush=True)
    squad = load_squad(); filler = collect_filler(tok)
    print(f"squad={len(squad)} filler={len(filler)}", flush=True)
    rows = []; sid = 0
    print("single-doc QA (squad)...", flush=True); sid = gen_single(tok, squad, filler, lengths, a.per, rows, sid)
    print(f"  now {len(rows)}", flush=True)
    print("multi-doc QA (squad)...", flush=True); sid = gen_multi(tok, squad, filler, lengths, a.per, rows, sid)
    print(f"  now {len(rows)}", flush=True)
    random.shuffle(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"wrote {len(rows)} -> {a.out}; {dict(Counter(r['task_type'] for r in rows))}", flush=True)

if __name__ == "__main__":
    main()
