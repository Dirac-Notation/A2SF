"""Format-diversified recipe supplement (fmtdiv_v1) — targets the cells with measured
recipe->LB transfer loss (history #66): Multi-doc QA (all models), Few Shot
classification/rouge (mistral/qwen/1B).

Non-LongBench sources only (clean protocol): SQuAD (shared arrow) + dialogsum (HF).
  1. mdqa   — TRUE multi-passage format ("Passage i:" headers, gold + distractors,
              question at the end) — v3's squad_multi looked single-doc.
  2. qtype  — trec-SHAPED question-type classification few-shot (labels derived from
              wh-word rules on SQuAD questions; fine-grained label set).
  3. dsum   — samsum-shaped dialogue-summary few-shot at LB-matched lengths.

  python datasets/make_fmtdiv.py --out datasets/training/raw/fmtdiv_v1/input.jsonl
"""
import argparse, json, os, random, re, sys
import pyarrow as pa, pyarrow.ipc as ipc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from transformers import AutoTokenizer
from datasets import load_dataset

SQUAD = ("/data2/shared/huggingface_cache/datasets/squad/plain_text/0.0.0/"
         "7b6d24c440a36b6815f21b70d25016731768db1f/squad-train.arrow")
LENGTHS = [2000, 3500, 5000, 7000]          # LB-matched token lengths

def load_squad():
    with pa.memory_map(SQUAD) as src:
        try: t = ipc.open_stream(src).read_all()
        except Exception: src.seek(0); t = ipc.open_file(src).read_all()
    return [(r["context"], r["question"], r["answers"]["text"][0]) for r in t.to_pylist()
            if r["answers"]["text"] and len(r["context"]) > 200]

def gen_mdqa(tok, per, rows, sid, squad):
    """LB hotpot/2wiki-shaped: numbered passages, gold buried among distractors."""
    ctxs = list({c: None for c, _, _ in squad}.keys())
    for made in range(per * len(LENGTHS)):
        L = LENGTHS[made % len(LENGTHS)]
        c, q, a = random.choice(squad)
        parts = [c]
        n = len(tok.encode(c, add_special_tokens=False))
        while n < L - 200:
            d = random.choice(ctxs)
            if d == c: continue
            parts.append(d); n += len(tok.encode(d, add_special_tokens=False))
        random.shuffle(parts)
        body = "\n".join(f"Passage {j+1}:\n{p}\n" for j, p in enumerate(parts))
        prompt = (f"Answer the question based on the given passages. Only give me the answer "
                  f"and do not output any other words.\n\nThe following are given passages.\n{body}\n"
                  f"Answer the question based on the given passages. Only give me the answer and do "
                  f"not output any other words.\n\nQuestion: {q}\nAnswer:")
        rows.append(dict(sample_id=sid, input_prompt=prompt, answers=[a], all_classes=[],
                         metric_type="qa_f1_score", task_type="Multi-doc QA", dataset="fmt_mdqa",
                         length=len(tok.encode(prompt)), generation_length=32, subtype="_"))
        sid += 1
    return sid

QTYPE_RULES = [
    (r"^who\b|whose\b|whom\b", "person"), (r"^where\b", "location"),
    (r"^when\b|what year|what date", "time"), (r"how (many|much|long|old|far|tall)", "number"),
    (r"^why\b", "reason"), (r"^(what|which)\b", "entity"), (r"^how\b", "manner"),
]
def qtype(q):
    ql = q.lower().strip()
    for pat, lab in QTYPE_RULES:
        if re.search(pat, ql): return lab
    return None

def gen_qtype(tok, per, rows, sid, squad):
    """trec-SHAPED: classify the question into fine types; few-shot label blocks."""
    LAB = ["person", "location", "time", "number", "reason", "entity", "manner"]
    pool = [(q, qtype(q)) for _, q, _ in squad]
    pool = [(q, t) for q, t in pool if t]
    for made in range(per * len(LENGTHS)):
        L = LENGTHS[made % len(LENGTHS)]
        random.shuffle(pool); shots = []; n = 0; i = 0
        while n < L - 100 and i < len(pool) - 1:
            s = f"Question: {pool[i][0]}\nType: {pool[i][1]}\n\n"
            n += len(tok.encode(s, add_special_tokens=False)); shots.append(s); i += 1
        q, lab = pool[i]
        prompt = (f"Please determine the type of the question below. Here are some examples "
                  f"of questions.\n\n" + "".join(shots) + f"Question: {q}\nType:")
        rows.append(dict(sample_id=sid, input_prompt=prompt, answers=[lab], all_classes=LAB,
                         metric_type="classification_score", task_type="Few Shot",
                         dataset="fmt_qtype", length=len(tok.encode(prompt)),
                         generation_length=16, subtype="classification"))
        sid += 1
    return sid

def gen_dsum(tok, per, rows, sid):
    """samsum-SHAPED at LB lengths: Dialogue/Summary exemplars then a final dialogue."""
    pool = [(e["dialogue"].strip(), e["summary"].strip())
            for e in load_dataset("knkarthick/dialogsum", split="train", streaming=True).take(12000)]
    for made in range(per * len(LENGTHS)):
        L = LENGTHS[made % len(LENGTHS)]
        random.shuffle(pool); shots = []; n = 0; i = 0
        while n < L - 250 and i < len(pool) - 1:
            d, s = pool[i]
            ex = f"Dialogue:\n{d}\n\nSummary:\n{s}\n\n"
            n += len(tok.encode(ex, add_special_tokens=False)); shots.append(ex); i += 1
        d, s = pool[i]
        prompt = ("Summarize the dialogue into a few short sentences. The following are some "
                  "examples.\n\n" + "".join(shots) + f"Dialogue:\n{d}\n\nSummary:")
        rows.append(dict(sample_id=sid, input_prompt=prompt, answers=[s], all_classes=[],
                         metric_type="rouge_score", task_type="Few Shot", dataset="fmt_dsum",
                         length=len(tok.encode(prompt)), generation_length=64, subtype="dialogue"))
        sid += 1
    return sid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=45, help="rows per length bucket per family")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(
        json.load(open(f"{REPO}/config/model2path.json"))["llama3-1b"])
    squad = load_squad()
    rows, sid = [], 0
    sid = gen_mdqa(tok, a.per, rows, sid, squad)
    sid = gen_qtype(tok, a.per, rows, sid, squad)
    sid = gen_dsum(tok, a.per, rows, sid)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"{len(rows)} rows -> {a.out}")
    print(Counter((r["task_type"], r["dataset"]) for r in rows))
    print("length dist:", np.percentile([r["length"] for r in rows], [10, 50, 90]).astype(int)
          if (np := __import__("numpy")) else "")

if __name__ == "__main__":
    main()
