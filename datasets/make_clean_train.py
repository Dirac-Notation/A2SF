"""Clean-source training data for the 3 task types that aren't easily synthetic:
Summarization (arXiv papers), Few-Shot (AG News), Code (github-code repo files).
NONE of these are LongBench source datasets (LongBench uses gov_report/multi_news/qmsum,
trec/triviaqa/samsum, lcc/repobench-p respectively — all AVOIDED here).
Matches LongBench STRUCTURE: long context, answer-position appropriate per task.

  python datasets/make_clean_train.py --model llama3-1b --out datasets/training/synth/clean_v1.jsonl \
      --per 200 --lengths 2000,4000,8000,12000
"""
import argparse, json, os, random, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from transformers import AutoTokenizer
from datasets import load_dataset

def truncate_tokens(tok, text, n):
    ids = tok.encode(text, add_special_tokens=False)[:n]
    return tok.decode(ids)

def gen_summ(tok, lengths, per, rows, sid0):
    ds = load_dataset("scientific_papers", "arxiv", split="train", streaming=True, trust_remote_code=True)
    it = iter(ds); sid = sid0; made = 0
    target = per * len(lengths)
    while made < target:
        try: ex = next(it)
        except StopIteration: break
        art = ex.get("article", "").strip(); summ = ex.get("abstract", "").strip()
        if len(art) < 2000 or len(summ) < 100: continue
        L = lengths[made % len(lengths)]
        body = truncate_tokens(tok, art, L - 60)
        prompt = ("[INST]You are given a scientific article. Write a concise abstract summarizing it.\n\n"
                  + body + "\n\nSummary:[/INST]")
        rows.append({"sample_id": sid, "input_prompt": prompt, "answers": [summ], "all_classes": [],
                     "metric_type": "rouge_score", "task_type": "Summarization", "dataset": "arxiv_summ",
                     "length": len(tok.encode(prompt)), "generation_length": 256})
        sid += 1; made += 1
    return sid

def gen_fewshot(tok, lengths, per, rows, sid0):
    LABELS = ["World", "Sports", "Business", "Technology"]
    ds = load_dataset("ag_news", split="train", streaming=True)
    pool = []
    for ex in ds:
        pool.append((ex["text"].strip().replace("\n", " "), LABELS[ex["label"]]))
        if len(pool) >= 20000: break
    sid = sid0; target = per * len(lengths)
    for made in range(target):
        L = lengths[made % len(lengths)]
        random.shuffle(pool)
        shots = []; n = 0; i = 0
        # pack labeled examples until ~L tokens, hold out last as query
        while n < L - 120 and i < len(pool) - 1:
            txt, lab = pool[i]
            ex_str = f"Article: {txt}\nCategory: {lab}\n\n"
            n += len(tok.encode(ex_str, add_special_tokens=False)); shots.append(ex_str); i += 1
        qtxt, qlab = pool[i]
        prompt = ("[INST]Classify each news article into one of: World, Sports, Business, Technology.\n\n"
                  + "".join(shots) + f"Article: {qtxt}\nCategory:[/INST]")
        rows.append({"sample_id": sid, "input_prompt": prompt, "answers": [qlab], "all_classes": LABELS,
                     "metric_type": "classification_score", "task_type": "Few Shot", "dataset": "agnews_fs",
                     "length": len(tok.encode(prompt)), "generation_length": 16})
        sid += 1
    return sid

def gen_code(tok, lengths, per, rows, sid0):
    # code_search_net (cached, fast). Concatenate many functions -> repo-level long context
    # (matches lcc/repobench structure), then complete the next line of the final function.
    ds = load_dataset("code_search_net", "python", split="train", streaming=True, trust_remote_code=True)
    funcs = []
    for ex in ds:
        s = ex.get("whole_func_string", "")
        if 40 <= len(tok.encode(s, add_special_tokens=False)) <= 400: funcs.append(s)
        if len(funcs) >= 30000: break
    sid = sid0; target = per * len(lengths)
    for made in range(target):
        L = lengths[made % len(lengths)]; random.shuffle(funcs)
        ctx = []; n = 0; i = 0
        while n < L - 200 and i < len(funcs) - 1:
            ctx.append(funcs[i]); n += len(tok.encode(funcs[i], add_special_tokens=False)) + 2; i += 1
        last = funcs[i]; lines = last.split("\n")
        if len(lines) < 6: continue
        j = random.randint(4, len(lines) - 2); ans = lines[j].rstrip()
        if len(ans.strip()) < 5: ans = next((lines[k].rstrip() for k in range(j, len(lines)) if len(lines[k].strip()) >= 5), "    pass")
        pref = "\n\n".join(ctx) + "\n\n" + "\n".join(lines[:j])
        prompt = "[INST]Complete the next line of this Python code file.\n\n" + pref + "\n[/INST]"
        rows.append({"sample_id": sid, "input_prompt": prompt, "answers": [ans], "all_classes": [],
                     "metric_type": "code_sim_score", "task_type": "Code Complete", "dataset": "csn_code",
                     "length": len(tok.encode(prompt)), "generation_length": 64})
        sid += 1
    return sid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-1b"); ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="2000,4000,8000,12000"); ap.add_argument("--per", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); random.seed(a.seed)
    m2p = json.load(open(f"{REPO}/config/model2path.json"))
    tok = AutoTokenizer.from_pretrained(m2p[a.model])
    lengths = [int(x) for x in a.lengths.split(",")]
    rows = []; sid = 0
    print("summ (arxiv)...", flush=True); sid = gen_summ(tok, lengths, a.per, rows, sid)
    print(f"  now {len(rows)} rows", flush=True)
    print("fewshot (ag_news)...", flush=True); sid = gen_fewshot(tok, lengths, a.per, rows, sid)
    print(f"  now {len(rows)} rows", flush=True)
    print("code (github)...", flush=True); sid = gen_code(tok, lengths, a.per, rows, sid)
    print(f"  now {len(rows)} rows", flush=True)
    random.shuffle(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"wrote {len(rows)} -> {a.out}; by task: {dict(Counter(r['task_type'] for r in rows))}", flush=True)

if __name__ == "__main__":
    main()
