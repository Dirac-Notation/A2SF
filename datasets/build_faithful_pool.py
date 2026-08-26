"""Build a LongBench-length-faithful RL training pool (leakage-free), approach (A).

LB uses source val/test → the source TRAIN split is leakage-free. LB's multi-doc
QA passages are FULL wiki articles (~816 words/passage), not the short hotpot
distractor intros (~77 w). Faithful full-wiki title lookup is heavy/fragile, so we
keep the REAL gold passage (so the answer is findable → reward is informative) and
pad with long wikitext-103 chunks as distractors to hit LB's length. The policy
learns from length + multi-passage structure; distractor wording is irrelevant.

  hotpotqa/musique : real gold passage(s) + wikitext distractors → ~LB length
  narrativeqa      : real full document (cap ~18k words) = LB construction
  passage_count    : wikitext chunks, repeat each random times, shuffle (procedural)
  passage_retrieval_en : 30 wikitext chunks + pseudo-abstract(first ~40w of target)
  kept as-is (already LB-length-matched): gov_report, qmsum, multi_news, samsum,
    2wikimqa, qasper, trec, triviaqa, lcc, repobench-p
  dropped: multifieldqa_en (LB-custom human-annotated, no train split + old leak)

Then resample to LB's length histogram. Output = raw inputs (scored separately).

  python datasets/build_faithful_pool.py [--max N] [--target_total 3500]
"""
import argparse, json, os, random, statistics, sys
from collections import defaultdict

random.seed(42)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
with open(os.path.join(ROOT, "config/dataset2prompt.json")) as f: TMPL = json.load(f)
with open(os.path.join(ROOT, "config/dataset2maxlen.json")) as f: GENL = json.load(f)

TASK = {"hotpotqa":"Multi-doc QA","musique":"Multi-doc QA","2wikimqa":"Multi-doc QA",
        "narrativeqa":"Single-doc QA","qasper":"Single-doc QA","gov_report":"Summarization",
        "qmsum":"Summarization","multi_news":"Summarization","triviaqa":"Few Shot",
        "trec":"Few Shot","samsum":"Few Shot","lcc":"Code Complete","repobench-p":"Code Complete",
        "passage_count":"Passage Retrieval","passage_retrieval_en":"Passage Retrieval"}
METRIC = {"gov_report":"rouge_score","qmsum":"rouge_score","multi_news":"rouge_score",
          "samsum":"rouge_score","trec":"classification_score","passage_count":"qa_f1_score",
          "passage_retrieval_en":"retrieval_score","lcc":"code_sim_score","repobench-p":"code_sim_score"}
LB_MED = {"hotpotqa":10104,"musique":11386,"narrativeqa":17499,"passage_count":11505,
          "passage_retrieval_en":9255}
def metric_of(ds): return METRIC.get(ds, "qa_f1_score")
def wc(s): return len(s.split())

records = []; CHUNKS = []; SID = 30000

def add(prompt, answers, ds, all_classes=None):
    global SID
    records.append({"sample_id": str(SID), "input_prompt": prompt,
        "answers": answers if isinstance(answers, list) else [answers],
        "all_classes": all_classes or [], "length": wc(prompt),
        "generation_length": int(GENL.get(ds, 64)), "dataset": ds,
        "task_type": TASK[ds], "metric_type": metric_of(ds)}); SID += 1


def load_wikitext_chunks(target=12000):
    """Article paragraphs from wikitext-103 as (title, text); ~100-1200 words each."""
    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train", trust_remote_code=True)
    title = "Article"; buf = []
    for row in d:
        line = row["text"]
        s = line.strip()
        if not s:
            if buf:
                text = " ".join(buf).replace(" <unk>", "").replace("<unk>", "")
                if 100 <= len(text.split()) <= 1400:
                    CHUNKS.append((title, text))
                buf = []
            continue
        if s.startswith("= ") and s.endswith(" =") and "= =" not in s:  # article header
            title = s.strip("= ").strip()
            continue
        if s.startswith("="):  # section header — flush
            if buf:
                text = " ".join(buf).replace(" <unk>", "").replace("<unk>", "")
                if 100 <= len(text.split()) <= 1400: CHUNKS.append((title, text))
                buf = []
            continue
        buf.append(s)
        if len(CHUNKS) >= target: break
    random.shuffle(CHUNKS)
    print(f"  wikitext chunks: {len(CHUNKS)} (median {statistics.median([len(t.split()) for _,t in CHUNKS])} w)")

_CHUNK_I = 0
def take_chunk():
    global _CHUNK_I
    c = CHUNKS[_CHUNK_I % len(CHUNKS)]; _CHUNK_I += 1; return c

def pad_to_length(gold_blocks, target_words):
    """gold_blocks: list[(title,text)] real evidence. Add wikitext distractors until
    ~target_words, shuffle, return numbered 'Passage i' context string."""
    items = list(gold_blocks)
    cur = sum(len(t.split()) for _, t in items)
    while cur < target_words:
        t, txt = take_chunk(); items.append((t, txt)); cur += len(txt.split())
    random.shuffle(items)
    return "\n".join(f"Passage {i+1}:\n{t}\n{txt}" for i, (t, txt) in enumerate(items))


def build_hotpotqa(n):
    from datasets import load_dataset
    d = load_dataset("hotpot_qa", "distractor", split="train", trust_remote_code=True)
    tmpl = TMPL["hotpotqa"]; cnt = 0
    for ex in d:
        if cnt >= n: break
        ctx = ex["context"]; gold_titles = set(ex["supporting_facts"]["title"])
        gold = [(t, "".join(s) if isinstance(s, list) else str(s))
                for t, s in zip(ctx["title"], ctx["sentences"]) if t in gold_titles]
        if not gold: continue
        context = pad_to_length(gold, LB_MED["hotpotqa"])
        add(tmpl.format(context=context, input=ex["question"]), [ex["answer"]], "hotpotqa"); cnt += 1
    print(f"  hotpotqa: {cnt}")


def build_musique(n):
    from datasets import load_dataset
    d = load_dataset("dgslibisey/MuSiQue", split="train", trust_remote_code=True)
    tmpl = TMPL["musique"]; cnt = 0
    for ex in d:
        if cnt >= n: break
        gold = [(p.get("title", ""), p.get("paragraph_text", ""))
                for p in ex["paragraphs"] if p.get("is_supporting")]
        if not gold: continue
        context = pad_to_length(gold, LB_MED["musique"])
        add(tmpl.format(context=context, input=ex["question"]), [ex["answer"]], "musique"); cnt += 1
    print(f"  musique: {cnt}")


def build_narrativeqa(n):
    from datasets import load_dataset
    d = load_dataset("deepmind/narrativeqa", split="train", trust_remote_code=True)
    tmpl = TMPL["narrativeqa"]; cnt = 0
    for ex in d:
        if cnt >= n: break
        doc = ex["document"]["text"] if isinstance(ex["document"], dict) else str(ex["document"])
        words = doc.split()
        if len(words) > 18000: doc = " ".join(words[:18000])
        q = ex["question"]["text"] if isinstance(ex["question"], dict) else str(ex["question"])
        ans = [a["text"] if isinstance(a, dict) else str(a) for a in ex["answers"]]
        add(tmpl.format(context=doc, input=q), ans, "narrativeqa"); cnt += 1
    print(f"  narrativeqa: {cnt}")


def build_passage_count(n):
    tmpl = TMPL["passage_count"]
    for _ in range(n):
        k = random.randint(10, 30)
        chosen = [take_chunk() for _ in range(k)]
        seq = []
        for (_, text) in chosen: seq += [text] * random.randint(1, 3)
        random.shuffle(seq)
        ctx = "\n\n".join(f"Paragraph {i+1}: {p}" for i, p in enumerate(seq))
        add(tmpl.format(context=ctx), [str(k)], "passage_count")
    print(f"  passage_count: {n}")


def build_passage_retrieval(n):
    tmpl = TMPL["passage_retrieval_en"]
    for _ in range(n):
        chosen = [take_chunk() for _ in range(30)]
        ctx = "\n\n".join(f"Paragraph {i+1}: {text}" for i, (_, text) in enumerate(chosen))
        tgt = random.randint(0, 29)
        abstract = " ".join(chosen[tgt][1].split()[:40])
        add(tmpl.format(context=ctx, input=abstract), [f"Paragraph {tgt+1}"], "passage_retrieval_en")
    print(f"  passage_retrieval_en: {n}")


KEEP = ["gov_report","qmsum","multi_news","samsum","2wikimqa","qasper","trec","triviaqa","lcc","repobench-p"]
def keep_existing():
    src = os.path.join(ROOT, "datasets/training/scored/llama3-1b/train.jsonl")
    def to_lb(ds):
        for lb in KEEP:
            if ds.startswith(lb) or ds.startswith(lb.replace("-", "_")): return lb
        if "lcc" in ds: return "lcc"
        if "repobench" in ds: return "repobench-p"
        if "trivia" in ds: return "triviaqa"
        if ds.startswith("trec"): return "trec"
        return None
    cnt = defaultdict(int)
    for line in open(src):
        d = json.loads(line); lb = to_lb(d.get("dataset", ""))
        if lb in KEEP:
            records.append({"sample_id": d["sample_id"], "input_prompt": d["input_prompt"],
                "answers": json.loads(d["answers"]) if isinstance(d["answers"], str) else d["answers"],
                "all_classes": json.loads(d["all_classes"]) if isinstance(d.get("all_classes"), str) else (d.get("all_classes") or []),
                "length": int(d["length"]), "generation_length": int(d.get("generation_length", 64)),
                "dataset": lb, "task_type": TASK[lb], "metric_type": d.get("metric_type", metric_of(lb))})
            cnt[lb] += 1
    for lb, c in cnt.items(): print(f"  keep {lb}: {c}")


def report():
    L = [r["length"] for r in records]
    b = [(0,2000),(2000,4000),(4000,8000),(8000,16000),(16000,32000),(32000,10**9)]
    lab = ['<2k','2-4k','4-8k','8-16k','16-32k','>32k']
    c = [sum(1 for x in L if lo<=x<hi) for lo, hi in b]
    print(f"  TOTAL {len(records)} (median {int(statistics.median(L))})")
    print("  len dist:", {lab[i]: f"{100*c[i]/max(1,len(L)):.1f}%" for i in range(6)})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--out", default="datasets/training/faithful_inputs.jsonl")
    a = ap.parse_args(); N = a.max
    print("=== load wikitext chunk pool ==="); load_wikitext_chunks()
    print("=== rebuild (length-faithful) ===")
    build_hotpotqa(N); build_musique(N); build_narrativeqa(min(N, 200))
    build_passage_count(N); build_passage_retrieval(N)
    print("=== keep existing matched ==="); keep_existing()
    out = os.path.join(ROOT, a.out)
    with open(out, "w") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("=== pool ==="); report(); print("saved ->", out)

if __name__ == "__main__":
    main()
