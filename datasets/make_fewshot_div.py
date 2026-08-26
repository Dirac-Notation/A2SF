"""Diversified Few-Shot recipe: 3 sub-types mirroring LongBench's heterogeneous Few-Shot
family (trec=classification, triviaqa=QA, samsum=dialogue-summary) — using NON-LongBench
sources (ag_news, SQuAD, dialogsum). Each carries an OBSERVABLE sub-type marker so a meta
policy can learn sub-type -> action (the only generalizable route past task-fixed).
All NO-CHAT completion format (LB Few-Shot is no-chat). Stores 'subtype' for metadata.

  python datasets/make_fewshot_div.py --model llama3-1b --out datasets/training/synth/fewshot_div.jsonl --per 160
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
        except Exception: src.seek(0); t = ipc.open_file(src).read_all()
    return [(r["context"], r["question"], r["answers"]["text"][0]) for r in t.to_pylist()
            if r["answers"]["text"] and len(r["context"]) > 150]

def gen_class(tok, lengths, per, rows, sid):  # like trec
    LAB = ["World", "Sports", "Business", "Technology"]
    pool = [(e["text"].strip().replace("\n", " "), LAB[e["label"]])
            for e in load_dataset("ag_news", split="train", streaming=True).take(20000)]
    for made in range(per * len(lengths)):
        L = lengths[made % len(lengths)]; random.shuffle(pool); shots = []; n = 0; i = 0
        while n < L - 120 and i < len(pool) - 1:
            s = f"Article: {pool[i][0]}\nCategory: {pool[i][1]}\n\n"; n += len(tok.encode(s, add_special_tokens=False)); shots.append(s); i += 1
        q, lab = pool[i]
        prompt = ("Classify each article into: World, Sports, Business, Technology.\n\n" + "".join(shots) + f"Article: {q}\nCategory:")
        rows.append(dict(sample_id=sid, input_prompt=prompt, answers=[lab], all_classes=LAB,
                         metric_type="classification_score", task_type="Few Shot", dataset="div_class",
                         length=len(tok.encode(prompt)), generation_length=16, subtype="classification")); sid += 1
    return sid

def gen_qa(tok, lengths, per, rows, sid, squad):  # like triviaqa few-shot QA
    for made in range(per * len(lengths)):
        L = lengths[made % len(lengths)]; shots = []; n = 0
        while n < L - 120:
            c, q, a = random.choice(squad); s = f"Passage: {c[:400]}\nQuestion: {q}\nAnswer: {a}\n\n"
            n += len(tok.encode(s, add_special_tokens=False)); shots.append(s)
        c, q, a = random.choice(squad)
        prompt = ("Answer the question after each passage.\n\n" + "".join(shots) + f"Passage: {c[:400]}\nQuestion: {q}\nAnswer:")
        rows.append(dict(sample_id=sid, input_prompt=prompt, answers=[a], all_classes=[],
                         metric_type="qa_f1_score", task_type="Few Shot", dataset="div_qa",
                         length=len(tok.encode(prompt)), generation_length=32, subtype="qa")); sid += 1
    return sid

def gen_dialog(tok, lengths, per, rows, sid):  # like samsum dialogue summary
    pool = [(e["dialogue"].strip(), e["summary"].strip())
            for e in load_dataset("knkarthick/dialogsum", split="train", streaming=True).take(12000)]
    for made in range(per * len(lengths)):
        L = lengths[made % len(lengths)]; random.shuffle(pool); shots = []; n = 0; i = 0
        while n < L - 200 and i < len(pool) - 1:
            d, s = pool[i]; ex = f"Dialogue:\n{d}\nSummary: {s}\n\n"; n += len(tok.encode(ex, add_special_tokens=False)); shots.append(ex); i += 1
        d, s = pool[i]
        prompt = ("Summarize each dialogue.\n\n" + "".join(shots) + f"Dialogue:\n{d}\nSummary:")
        rows.append(dict(sample_id=sid, input_prompt=prompt, answers=[s], all_classes=[],
                         metric_type="rouge_score", task_type="Few Shot", dataset="div_dialog",
                         length=len(tok.encode(prompt)), generation_length=64, subtype="dialogue")); sid += 1
    return sid

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="llama3-1b"); ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="2000,4000,8000"); ap.add_argument("--per", type=int, default=160); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(json.load(open(f"{REPO}/config/model2path.json"))[a.model])
    lengths = [int(x) for x in a.lengths.split(",")]; rows = []; sid = 0
    print("class...", flush=True); sid = gen_class(tok, lengths, a.per, rows, sid)
    print("qa...", flush=True); sid = gen_qa(tok, lengths, a.per, rows, sid, load_squad())
    print("dialog...", flush=True); sid = gen_dialog(tok, lengths, a.per, rows, sid)
    random.shuffle(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"wrote {len(rows)} -> {a.out}; {dict(Counter(r['subtype'] for r in rows))}", flush=True)

if __name__ == "__main__":
    main()
