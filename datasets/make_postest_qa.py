"""Stage-3 feasibility: NATURAL SQuAD QA with the answer paragraph at controlled, stored
positions (answer_pos) in WikiText filler. 1B can actually solve these (unlike the gibberish
synthetic QA), so the per-prompt best (a,b) is meaningful -> fair test of position->action.

  python datasets/make_postest_qa.py --model llama3-1b --out datasets/training/synth/postest_qa.jsonl \
      --length 4000 --positions 0.05,0.15,...,0.95 --per 40
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
            if r["answers"]["text"] and len(r["context"]) > 200]

def collect_filler(tok, n=4000):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    out = []
    for ex in ds:
        t = ex.get("text", "").strip()
        if len(t) < 60 or t.startswith("="): continue
        out.append(t)
        if len(out) >= n: break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-1b"); ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=4000)
    ap.add_argument("--positions", default="0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95")
    ap.add_argument("--per", type=int, default=40); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(json.load(open(f"{REPO}/config/model2path.json"))[a.model])
    squad = load_squad(); filler = collect_filler(tok); L = a.length
    positions = [float(x) for x in a.positions.split(",")]
    # pre-tokenize filler sentences
    fil_ids = [tok.encode(s, add_special_tokens=False) for s in filler]
    rows = []; sid = 0
    for pos in positions:
        for _ in range(a.per):
            ctx, q, ans = random.choice(squad)
            ctx_ids = tok.encode(ctx, add_special_tokens=False)
            need = L - len(ctx_ids) - 40
            pre_n = int(need * pos); post_n = need - pre_n
            def pack(n):
                out = []; tot = 0
                while tot < n:
                    f = random.choice(fil_ids); out += f; tot += len(f)
                return out[:n]
            body_ids = pack(pre_n) + ctx_ids + pack(post_n)
            body = tok.decode(body_ids)
            prompt = ("[INST]Answer the question based on the document.\n\n<document>\n" + body +
                      f"\n</document>\n\nQuestion: {q}\nAnswer:[/INST]")
            rows.append({"sample_id": sid, "input_prompt": prompt, "answers": [ans], "all_classes": [],
                         "metric_type": "qa_f1_score", "task_type": "Single-doc QA", "dataset": "squad_pos",
                         "length": len(tok.encode(prompt)), "generation_length": 32, "answer_pos": pos})
            sid += 1
    random.shuffle(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
