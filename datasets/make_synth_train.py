"""Benchmark-agnostic SYNTHETIC training data (RULER-style) — NO LongBench source data.
Filler = WikiText-103 (clean, not a LongBench eval dataset). Synthetic facts/keys/needles.
Covers task TYPES by STRUCTURE (context length + answer position), so the per-task optimal
forgetting (a,b) matches any benchmark with that structure (incl. LongBench).

Task types produced (training jsonl format = sample_id,input_prompt,answers,all_classes,
metric_type,task_type,dataset,length,generation_length):
  - passage_retrieval (Passage Retrieval): numbered passages, find the one matching a query.
  - needle (Passage Retrieval): one 'current password' among distractors.
  - passage_count (Passage Count): count occurrences of a marker phrase.
  - multihop (Multi-doc QA): chained key->key->value tracing.
  - qa_fact (Single-doc QA): insert a fact, ask about it.

Structure knobs: --lengths (token budgets), --positions (answer depth 0..1), --per N each.
  python datasets/make_synth_train.py --model llama3-1b --out datasets/training/synth/synth_v1.jsonl \
      --lengths 1000,2000,4000,8000,12000 --positions 0.1,0.3,0.5,0.7,0.9 --per 8
"""
import argparse, json, os, random, string, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from transformers import AutoTokenizer
from datasets import load_dataset

ALNUM = string.ascii_uppercase + string.digits
WORDS = ["apple","river","stone","cloud","tiger","amber","maple","crisp","lunar","quartz",
         "ember","frost","glade","harbor","ivory","jolt","karma","lotus","mango","nimbus"]
def rid(n=7): return "".join(random.choice(ALNUM) for _ in range(n))
def rword(): return random.choice(WORDS) + str(random.randint(10, 99))

def collect_filler(tokenizer, n_target=6000):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    out = []
    for ex in ds:
        t = ex.get("text", "").strip()
        if not t or t.startswith("=") or t.startswith("@"): continue
        for p in t.split("."):
            p = p.strip()
            if len(p) < 25 or "password" in p.lower(): continue
            tl = len(tokenizer.encode(p, add_special_tokens=False))
            if 8 <= tl <= 22: out.append(p + ".")
        if len(out) >= n_target: break
    random.shuffle(out)
    return out

def fill_to(toklen, filler, tokenizer):
    """Pick filler sentences until ~toklen tokens; return list of sentences."""
    picked, n = [], 0
    while n < toklen:
        s = random.choice(filler); picked.append(s)
        n += len(tokenizer.encode(s, add_special_tokens=False)) + 1
        if len(picked) > 80000: break
    return picked

def insert_at(picked, items_by_pos):
    """items_by_pos: list of (frac, text). Insert into picked at fractional positions."""
    n = len(picked)
    ins = sorted([(max(0, min(n, int(round(n * f)))), txt) for f, txt in items_by_pos], key=lambda x: x[0])
    out, off = [], 0
    for pos, txt in ins:
        out.extend(picked[off:pos]); out.append(txt); off = pos
    out.extend(picked[off:])
    return " ".join(out)

# ---- per-task generators: return (prompt, answers[list], metric, task_type, gen_len) ----
def gen_needle(filler, tok, toklen, pos):
    pw = rid(8); needle = f"The current password is {pw}."
    distract = [f"The {l} password is {rid(8)}." for l in ["old","default","temporary","backup"]]
    picked = fill_to(toklen, filler, tok)
    items = [(pos, needle)] + [(random.random(), d) for d in distract]
    body = insert_at(picked, items)
    prompt = ("[INST]The following text contains several password sentences (old, default, "
              "temporary, backup, current). Find the CURRENT password.\n\n<text>\n" + body +
              "\n</text>\n\nWhat is the CURRENT password? Answer with only the password.[/INST]")
    return prompt, [pw], "qa_f1_score", "Passage Retrieval", 16

def gen_retrieval(filler, tok, toklen, pos):
    npar = random.randint(8, 16); tgt = random.randint(0, npar - 1)
    secret = rid(10)
    pars = []
    for i in range(npar):
        pp = fill_to(max(40, toklen // npar), filler, tok)
        tag = f" The access code for section {i} is {secret if i==tgt else rid(10)}."
        pars.append(f"Passage {i}: " + " ".join(pp) + tag)
    body = "\n\n".join(pars)
    prompt = ("[INST]Below are numbered passages. Each has an 'access code for section i'.\n\n" + body +
              f"\n\nWhich passage number has access code {secret}? Answer with only the passage number.[/INST]")
    return prompt, [str(tgt), f"Passage {tgt}"], "qa_f1_score", "Passage Retrieval", 16

def gen_count(filler, tok, toklen, pos):
    marker = rword(); k = random.randint(2, 7)
    picked = fill_to(toklen, filler, tok)
    items = [(random.random(), f"Note: the keyword {marker} appears here.") for _ in range(k)]
    body = insert_at(picked, items)
    prompt = ("[INST]Read the text and count occurrences.\n\n<text>\n" + body +
              f"\n</text>\n\nHow many times does the keyword {marker} appear? Answer with only a number.[/INST]")
    return prompt, [str(k)], "qa_f1_score", "Passage Count", 8

def gen_multihop(filler, tok, toklen, pos):
    a, b, c = rword(), rword(), rid(8)
    chain = [f"The partner of {a} is {b}.", f"The secret of {b} is {c}."]
    picked = fill_to(toklen, filler, tok)
    items = [(pos, chain[0]), (min(1.0, pos + 0.1 + random.random() * 0.2), chain[1])]
    body = insert_at(picked, items)
    prompt = ("[INST]The text contains facts of the form 'partner of X is Y' and 'secret of Y is Z'.\n\n<text>\n"
              + body + f"\n</text>\n\nWhat is the secret of the partner of {a}? Answer with only the secret.[/INST]")
    return prompt, [c], "qa_f1_score", "Multi-doc QA", 16

def gen_qa_fact(filler, tok, toklen, pos):
    ent = rword(); attr = random.choice(["capital","mascot","motto","color","founder"]); val = rword()
    fact = f"The {attr} of {ent} is {val}."
    picked = fill_to(toklen, filler, tok)
    body = insert_at(picked, [(pos, fact)])
    prompt = ("[INST]Read the document and answer the question.\n\n<document>\n" + body +
              f"\n</document>\n\nWhat is the {attr} of {ent}? Answer briefly.[/INST]")
    return prompt, [val], "qa_f1_score", "Single-doc QA", 16

GENS = [gen_needle, gen_retrieval, gen_count, gen_multihop, gen_qa_fact]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-1b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="1000,2000,4000,8000,12000")
    ap.add_argument("--positions", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--per", type=int, default=8, help="prompts per (gen,length,position)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)
    m2p = json.load(open(f"{REPO}/config/model2path.json"))
    tok = AutoTokenizer.from_pretrained(m2p[a.model])
    print("collecting WikiText filler...", flush=True)
    filler = collect_filler(tok)
    print(f"filler sentences: {len(filler)}", flush=True)
    lengths = [int(x) for x in a.lengths.split(",")]
    positions = [float(x) for x in a.positions.split(",")]
    rows = []; sid = 0
    for g in GENS:
        for L in lengths:
            for pos in positions:
                for _ in range(a.per):
                    prompt, ans, metric, task, glen = g(filler, tok, L, pos)
                    rows.append({"sample_id": sid, "input_prompt": prompt, "answers": ans,
                                 "all_classes": [], "metric_type": metric, "task_type": task,
                                 "dataset": "synth_" + g.__name__[4:], "length": len(tok.encode(prompt)),
                                 "generation_length": glen, "answer_pos": pos})  # Stage-3: known answer depth
                    sid += 1
    random.shuffle(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"wrote {len(rows)} rows -> {a.out}", flush=True)
    print("by task:", dict(Counter(r["task_type"] for r in rows)), flush=True)
    print("len dist:", sorted(Counter((r["length"]//2000)*2000 for r in rows).items()), flush=True)

if __name__ == "__main__":
    main()
