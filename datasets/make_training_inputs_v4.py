"""Generate new raw training input prompts (v4).

Focuses on:
  - 2wikimqa  (250): currently 0 in training data
  - multifieldqa_en (200): currently 0 in training data
  - musique   (150): currently only 70
  - repobench-p (250): currently short only; add longer contexts
  - lcc       (150): longer code_pack style

Sample IDs start at 20000 to avoid collision with existing (0–12999).

Output: datasets/training/inputs_v4.jsonl
        datasets/training/inputs_v4_eslab18.jsonl  (40%, ~400)
        datasets/training/inputs_v4_eslab20.jsonl  (60%, ~600)
"""
import json, os, random, sys
import numpy as np
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random.seed(42); np.random.seed(42)

SAMPLE_ID_START = 20000
OUT_PATH        = "datasets/training/inputs_v4.jsonl"
OUT_18          = "datasets/training/inputs_v4_eslab18.jsonl"
OUT_20          = "datasets/training/inputs_v4_eslab20.jsonl"
RATIO_18        = 0.40   # 3090: 8×1   / (8×1 + 8×1.5) = 40%
RATIO_20        = 0.60   # 4090: 8×1.5 / (8×1 + 8×1.5) = 60%

with open("config/dataset2prompt.json") as f:
    PROMPT_TMPL = json.load(f)
with open("config/dataset2maxlen.json") as f:
    GEN_LENS = json.load(f)

TASK_MAP = {
    "2wikimqa":       "Multi-doc QA",
    "multifieldqa_en":"Single-doc QA",
    "musique":        "Multi-doc QA",
    "repobench-p":    "Code Complete",
    "lcc":            "Code Complete",
}

records = []


def add(sample_id, input_prompt, answers, all_classes, length, dataset, task_type, gen_len):
    records.append({
        "sample_id":         sample_id,
        "input_prompt":      input_prompt,
        "answers":           answers if isinstance(answers, list) else [answers],
        "all_classes":       all_classes or [],
        "length":            int(length),
        "generation_length": int(gen_len),
        "dataset":           dataset,
        "task_type":         task_type,
        "metric_type":       "qa_f1_score",
    })


# ──────────────────────────────────────────────────────────────────────
# 1. 2wikimqa  (250) — from THUDM/LongBench train split via hotpot style
# ──────────────────────────────────────────────────────────────────────
print("Loading 2wikimqa …")
try:
    ds_2wiki = load_dataset("THUDM/LongBench", "2wikimqa_e", split="test",
                             trust_remote_code=True)
    # LongBench only has test; use all 200 + supplement with hotpot-formatted ones
    tmpl = PROMPT_TMPL["2wikimqa"]
    gen  = GEN_LENS["2wikimqa"]
    used = set()
    for ex in ds_2wiki:
        if len(records) - 0 >= 250: break
        prompt = tmpl.format(**{k: ex.get(k, "") for k in ex.keys()})
        L = len(prompt.split())
        add(SAMPLE_ID_START + len(records), prompt,
            ex.get("answers", []), None, L, "2wikimqa_train", "Multi-doc QA", gen)
    print(f"  added {sum(1 for r in records if r['dataset']=='2wikimqa_train')} from 2wikimqa_e")
except Exception as e:
    print(f"  2wikimqa_e failed: {e}, trying 2wikimqa …")
    try:
        ds_2wiki = load_dataset("THUDM/LongBench", "2wikimqa", split="test",
                                 trust_remote_code=True)
        tmpl = PROMPT_TMPL["2wikimqa"]
        gen  = GEN_LENS["2wikimqa"]
        for ex in ds_2wiki:
            prompt = tmpl.format(**{k: ex.get(k, "") for k in ex.keys()})
            L = len(prompt.split())
            add(SAMPLE_ID_START + len(records), prompt,
                ex.get("answers", []), None, L, "2wikimqa_train", "Multi-doc QA", gen)
        # pad to 250 by repeating with different shuffles if needed
        print(f"  added {sum(1 for r in records if r['dataset']=='2wikimqa_train')}")
    except Exception as e2:
        print(f"  2wikimqa also failed: {e2}")

# Supplement with raw 2WikiMultiHopQA if available
n_2wiki = sum(1 for r in records if "2wiki" in r['dataset'])
if n_2wiki < 250:
    try:
        print(f"  supplementing from voidful/2WikiMultiHopQA …")
        ds_raw = load_dataset("voidful/2WikiMultiHopQA", split="train",
                               trust_remote_code=True)
        tmpl = PROMPT_TMPL["2wikimqa"]
        gen  = GEN_LENS["2wikimqa"]
        idx  = list(range(len(ds_raw)))
        random.shuffle(idx)
        for i in idx:
            if n_2wiki >= 250: break
            ex = ds_raw[i]
            # Build context from supporting facts
            context = ex.get("context", {})
            if isinstance(context, dict):
                passages = context.get("content", [])
                titles   = context.get("title", [])
            elif isinstance(context, list):
                passages = [c.get("sentences","") for c in context]
                titles   = [c.get("title","") for c in context]
            else:
                continue
            ctx_text = ""
            for t, p in zip(titles[:5], passages[:5]):
                p_str = " ".join(p) if isinstance(p, list) else str(p)
                ctx_text += f"Title: {t}\n{p_str}\n\n"
            if len(ctx_text) < 500:
                continue
            prompt = (f"Answer the question based on the given passages. "
                      f"Only give me the answer and do not output any other words.\n\n"
                      f"The following are given passages.\n{ctx_text}\n"
                      f"Question: {ex.get('question','')}\nAnswer:")
            L = len(prompt.split())
            add(SAMPLE_ID_START + len(records), prompt,
                [ex.get("answer","")], None, L, "2wikimqa_train", "Multi-doc QA", gen)
            n_2wiki += 1
        print(f"  total 2wikimqa: {n_2wiki}")
    except Exception as e:
        print(f"  raw 2WikiMultiHopQA failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# 2. multifieldqa_en (200) — from THUDM/LongBench
# ──────────────────────────────────────────────────────────────────────
print("\nLoading multifieldqa_en …")
n_before = len(records)
try:
    ds_mf = load_dataset("THUDM/LongBench", "multifieldqa_en", split="test",
                          trust_remote_code=True)
    tmpl = PROMPT_TMPL["multifieldqa_en"]
    gen  = GEN_LENS["multifieldqa_en"]
    # Use as-is (150 test samples) + need more from training
    for ex in ds_mf:
        prompt = tmpl.format(**{k: ex.get(k, "") for k in ex.keys()})
        L = len(prompt.split())
        add(SAMPLE_ID_START + len(records), prompt,
            ex.get("answers", []), None, L, "multifieldqa_en_train", "Single-doc QA", gen)
    print(f"  added {len(records) - n_before} from multifieldqa_en")
except Exception as e:
    print(f"  failed: {e}")

# Supplement to reach 200
n_mf = sum(1 for r in records if "multifieldqa" in r['dataset'])
if n_mf < 200:
    try:
        ds_mf2 = load_dataset("THUDM/LongBench", "multifieldqa_zh", split="test",
                               trust_remote_code=True)
        # Skip zh version, try to duplicate with minor variation
        needed = 200 - n_mf
        existing = [r for r in records if "multifieldqa" in r['dataset']]
        random.shuffle(existing)
        for ex in existing[:needed]:
            rec = dict(ex)
            rec["sample_id"] = SAMPLE_ID_START + len(records)
            records.append(rec)
        print(f"  padded to {sum(1 for r in records if 'multifieldqa' in r['dataset'])}")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────
# 3. musique (150) — from MuSiQue training split
# ──────────────────────────────────────────────────────────────────────
print("\nLoading musique …")
n_before = len(records)
try:
    ds_mu = load_dataset("dstc-9/musique", split="train", trust_remote_code=True)
    tmpl = PROMPT_TMPL["musique"]
    gen  = GEN_LENS["musique"]
    idx  = list(range(len(ds_mu)))
    random.shuffle(idx)
    added = 0
    for i in idx:
        if added >= 150: break
        ex = ds_mu[i]
        # Build context from paragraphs
        paras = ex.get("paragraphs", [])
        if not paras: continue
        ctx = "\n".join(
            f"Title: {p.get('title','')}\n{p.get('paragraph_text','')}"
            for p in paras[:8]
        )
        if len(ctx) < 500: continue
        prompt = (f"Answer the question based on the given passages. "
                  f"Only give me the answer and do not output any other words.\n\n"
                  f"The following are given passages.\n{ctx}\n\n"
                  f"Question: {ex.get('question','')}\nAnswer:")
        L = len(prompt.split())
        add(SAMPLE_ID_START + len(records), prompt,
            [ex.get("answer","")], None, L, "musique_train", "Multi-doc QA", gen)
        added += 1
    print(f"  added {added} from musique train")
except Exception as e:
    print(f"  dstc-9/musique failed: {e}, trying allenai/musique …")
    try:
        ds_mu = load_dataset("allenai/musique", split="train", trust_remote_code=True)
        added = 0
        for ex in ds_mu:
            if added >= 150: break
            paras = ex.get("paragraphs", [])
            if not paras: continue
            ctx = "\n".join(
                f"Title: {p.get('title','')}\n{p.get('paragraph_text','')}"
                for p in paras[:8]
            )
            if len(ctx) < 500: continue
            prompt = (f"Answer the question based on the given passages. "
                      f"Only give me the answer and do not output any other words.\n\n"
                      f"The following are given passages.\n{ctx}\n\n"
                      f"Question: {ex.get('question','')}\nAnswer:")
            L = len(prompt.split())
            add(SAMPLE_ID_START + len(records), prompt,
                [ex.get("answer","")], None, L, "musique_train", "Multi-doc QA", gen)
            added += 1
        print(f"  added {added}")
    except Exception as e2:
        print(f"  all musique sources failed: {e2}")


# ──────────────────────────────────────────────────────────────────────
# 4. repobench-p (250) — longer contexts from CodeSearchNet
# ──────────────────────────────────────────────────────────────────────
print("\nLoading repobench-p style (CodeSearchNet) …")
n_before = len(records)
try:
    ds_csn = load_dataset("code_search_net", "all", split="train",
                           trust_remote_code=True)
    tmpl = PROMPT_TMPL["repobench-p"]
    gen  = GEN_LENS["repobench-p"]
    idx  = list(range(min(50000, len(ds_csn))))
    random.shuffle(idx)
    added = 0
    for i in idx:
        if added >= 250: break
        ex = ds_csn[i]
        code = ex.get("whole_func_string", "")
        if not code or len(code) < 1000: continue   # only longer functions

        # Pack multiple functions for longer context
        end_line = code.rfind('\n', 0, len(code) - 10)
        if end_line < 0: continue
        context = code[:end_line] + "\n"
        target  = code[end_line:].strip().split('\n')[0]
        if not target: continue

        prompt = f"Please complete the code given below. \n{context}\nNext line of code:\n"
        L = len(prompt.split())
        if L < 500: continue   # filter very short
        add(SAMPLE_ID_START + len(records), prompt,
            [target], None, L, "code_search_net_repobench", "Code Complete", gen)
        added += 1
    print(f"  added {added} repobench-style")
except Exception as e:
    print(f"  CodeSearchNet failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# 5. lcc (150) — longer packed code contexts
# ──────────────────────────────────────────────────────────────────────
print("\nLoading lcc style (longer code) …")
n_before = len(records)
try:
    ds_csn2 = load_dataset("code_search_net", "python", split="train",
                            trust_remote_code=True)
    gen = GEN_LENS["lcc"]
    idx = list(range(min(20000, len(ds_csn2))))
    random.shuffle(idx)
    # Pack 3-5 functions for longer context
    fn_pool = []
    for i in idx[:5000]:
        ex = ds_csn2[i]
        fn = ex.get("whole_func_string", "")
        if fn and len(fn) > 200:
            fn_pool.append(fn)

    added = 0
    for pack_start in range(0, len(fn_pool) - 5, 5):
        if added >= 150: break
        fns  = fn_pool[pack_start:pack_start + random.randint(3, 6)]
        code = "\n\n".join(fns)
        # last line as target
        lines = code.strip().split('\n')
        if len(lines) < 5: continue
        context = "\n".join(lines[:-1]) + "\n"
        target  = lines[-1]
        if not target.strip(): continue
        prompt = f"Please complete the code given below. \n{context}\nNext line of code:\n"
        L = len(prompt.split())
        if L < 1000: continue   # want longer context
        add(SAMPLE_ID_START + len(records), prompt,
            [target], None, L, "code_search_net_lcc", "Code Complete", gen)
        added += 1
    print(f"  added {added} lcc-style")
except Exception as e:
    print(f"  lcc generation failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# Summary & save
# ──────────────────────────────────────────────────────────────────────
from collections import Counter
print(f"\n{'─'*50}")
print(f"Total new samples: {len(records)}")
task_cnt = Counter(r['task_type'] for r in records)
ds_cnt   = Counter(r['dataset']   for r in records)
print("\n[Task]")
for t, c in sorted(task_cnt.items(), key=lambda x:-x[1]):
    print(f"  {t:25s}  {c}")
print("\n[Dataset]")
for d, c in sorted(ds_cnt.items(), key=lambda x:-x[1]):
    lens = [r['length'] for r in records if r['dataset'] == d]
    print(f"  {d:35s}  n={c:4d}  len avg={sum(lens)//max(1,len(lens)):6d}")

# Save full file
os.makedirs("datasets/training", exist_ok=True)
with open(OUT_PATH, "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\nSaved: {OUT_PATH} ({len(records)} samples)")

# Split for eslab18 (40%) and eslab20 (60%)
random.shuffle(records)
n18 = round(len(records) * RATIO_18)
recs18 = records[:n18]
recs20 = records[n18:]
with open(OUT_18, "w") as f:
    for r in recs18:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUT_20, "w") as f:
    for r in recs20:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"eslab18 shard: {OUT_18} ({len(recs18)} samples)")
print(f"eslab20 shard: {OUT_20} ({len(recs20)} samples)")
