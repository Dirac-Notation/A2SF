"""Model-agnostic per-prompt features (NO model forward). For each prompt, locate the
'question' and the 'context', then compute a TF-IDF similarity profile of the question vs
sliding context chunks. Features describe WHERE the answer-relevant content sits and how
concentrated it is -- candidate signal for the per-prompt best (a,b).

Adds 'pp_features' (list) to each row of <scored>/common.jsonl -> writes common_feat.jsonl.
  python script/extract_pp_features.py --scored datasets/training/raw/recipe_v2_1b
"""
import argparse, json, os, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def split_q_ctx(prompt):
    # question = text after the last 'Question:' or after '</...>' marker; context = the long body
    m = list(re.finditer(r"Question:\s*(.+?)\s*(?:Answer:|\[/INST\]|$)", prompt, re.S))
    if m:
        q = m[-1].group(1).strip()[:300]
        ctx = prompt[:m[-1].start()]
    else:
        # retrieval/needle: question is the trailing instruction sentence
        q = prompt.strip()[-300:]; ctx = prompt
    return q, ctx

def features(prompt, nchunks=20):
    q, ctx = split_q_ctx(prompt)
    toks = ctx.split()
    if len(toks) < nchunks * 3 or not q.strip():
        return [0.5, 0.0, 0.0, 0.0, 0.0]
    csz = max(1, len(toks) // nchunks)
    chunks = [" ".join(toks[i:i + csz]) for i in range(0, len(toks), csz)][:nchunks]
    try:
        vec = TfidfVectorizer().fit([q] + chunks)
        qv = vec.transform([q]); cv = vec.transform(chunks)
        sims = (cv @ qv.T).toarray().ravel()
    except Exception:
        return [0.5, 0.0, 0.0, 0.0, 0.0]
    if sims.max() <= 0: return [0.5, 0.0, 0.0, 0.0, 0.0]
    pos = sims.argmax() / max(1, len(sims) - 1)       # estimated answer position 0..1
    peak = sims.max() / (sims.mean() + 1e-9)          # concentration
    nhigh = float((sims > 0.5 * sims.max()).sum()) / len(sims)  # redundancy (spread)
    return [float(pos), float(sims.max()), float(sims.mean()), float(peak), float(nhigh)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scored", required=True); a = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{a.scored}/common.jsonl")]
    out = open(f"{a.scored}/common_feat.jsonl", "w")
    for i, r in enumerate(rows):
        r["pp_features"] = features(r.get("input_prompt", ""))
        out.write(json.dumps({k: r[k] for k in r if k != "input_prompt"}) + "\n")
        if i % 500 == 0: print(f"{i}/{len(rows)}", flush=True)
    out.close(); print(f"wrote {a.scored}/common_feat.jsonl  (5 features/prompt)")

if __name__ == "__main__":
    main()
