"""recipe-v6: a training corpus built from real long documents and human-written questions.

The difference from v5 is that the context is not assembled. v5 padded a gold passage with
unrelated distractors, and that assembly turned the task into retrieval: across four
constructions (proxy or gold reward, distractor or same-document padding) the (M, QA) cell
optimum came out as 10:16 every time, while LongBench wants 10:1 / 0:1.

v6 keeps the original long documents from LooGLE and QuALITY and only cuts a window of the
target length out of them, choosing the window position so that the evidence lands at a
prescribed relative depth. Documents and questions are real.

Stratification:

  context length     {4k, 8k, 16k, 24k} tokens (32k overflows full-cache generation on a 24GB
                     GPU, and LongBench itself is median 8.5k)
  evidence depth     {0.05, 0.275, 0.5, 0.725, 0.95}
  generation length  32 / 128 / 512, the last of which v5 did not cover
  instruction form   two or three templates per family, none copied from LongBench

  python datasets/build_recipe_v6.py --model llama3-8b --per_family 170 --gpu 0 \
      --out rv6_prompts.jsonl
"""
import argparse
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SRC = os.environ.get("ICLR_TRACES", "/data2/smp9898/iclr_traces") + "/rv6/src"
DEPTHS = [0.05, 0.275, 0.5, 0.725, 0.95]
CTX = [4000, 8000, 16000, 24000]   # 32k overflows a 24GB GPU during uncompressed full-cache generation

F_QA = [
    "Use the document to answer the question.\n\n<document>\n{ctx}\n</document>\n\nQuestion: {q}\nAnswer:",
    "{ctx}\n\nBased on the document above, answer the following.\nQ: {q}\nA:",
    "Read the document and respond concisely.\n\nDOCUMENT:\n{ctx}\n\nREQUEST: {q}\nRESPONSE:",
]
F_MC = [
    "{ctx}\n\nQuestion: {q}\nReply with the number of the best option only.\nAnswer:",
    "Read the document.\n\n{ctx}\n\n{q}\nGive only the option number.\nAnswer:",
]
F_PID = [
    "{ctx}\n\nThe sentence below was taken from one of the numbered passages above.\n{q}\n"
    "Which passage is it from? Answer in the form \"Paragraph N\".\nAnswer:",
    "Below are numbered passages from a document.\n\n{ctx}\n\nSentence: {q}\n"
    "Identify the passage it came from (reply \"Paragraph N\").\nAnswer:",
]
F_SUM = [
    "{ctx}\n\nWrite a summary of the document above.\nSummary:",
    "Summarize the following document.\n\n<document>\n{ctx}\n</document>\n\nSummary:",
    "DOCUMENT:\n{ctx}\n\nTASK: summarize the document.\nOUTPUT:",
]


def rd(name, limit=None):
    out = []
    for i, l in enumerate(open(f"{SRC}/{name}.jsonl")):
        if limit and i >= limit:
            break
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def as_list(ev):
    """Normalize `evidence`, which arrives as a list, a JSON string or a bare string.

    Always go through this: iterating a bare string yields characters, which silently dropped
    every row once."""
    if ev is None:
        return []
    if isinstance(ev, list):
        return [str(x) for x in ev]
    ev = str(ev).strip()
    if ev.startswith("["):
        for loader in (json.loads, __import__("ast").literal_eval):
            try:
                v = loader(ev)
                if isinstance(v, list):
                    return [str(x) for x in v]
            except Exception:
                pass
    return [ev] if ev else []


def _norm(t):
    return re.sub(r"\s+", " ", t)


def find_evidence(ctx, evidence):
    """Locate the evidence after whitespace normalization, as an approximate character index
    into the original text. Returns -1 on failure."""
    nc = _norm(ctx)
    for ev in evidence:
        e = _norm(ev).strip()
        for L in (200, 120, 60, 40):
            if len(e) < L:
                continue
            p = nc.find(e[:L])
            if p >= 0:
                return int(len(ctx) * p / max(len(nc), 1))
    return -1


def window(tok, ctx, evidence, n_tok, depth):
    """Cut an n_tok window out of the document so the evidence sits at relative depth `depth`.

    Returns None when the evidence cannot be located, in which case the row is dropped.
    The document is only cut, never assembled.
    """
    ids = tok(ctx, add_special_tokens=False).input_ids
    if len(ids) <= n_tok:
        return (tok.decode(ids, skip_special_tokens=True), 0.5) if not evidence else None
    pos = find_evidence(ctx, evidence)
    if pos < 0:
        return None
    # approximate the token position from the character position
    ev_tok = int(len(ids) * pos / max(len(ctx), 1))
    start = int(ev_tok - depth * n_tok)
    start = max(0, min(start, len(ids) - n_tok))
    got = (ev_tok - start) / n_tok
    return tok.decode(ids[start:start + n_tok], skip_special_tokens=True), got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--per_family", type=int, default=120)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(ROOT)
    import utils as U
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(json.load(open("config/model2path.json"))[a.model])
    rng = random.Random(a.seed)
    items, drop = [], 0

    def emit(fam, tt, gen_len, metric, ctx, q, gold, meta, fmts, fi, allc=None):
        body = fmts[fi % len(fmts)].format(ctx=ctx, q=q)
        d = dict(sample_id=f"rv6_{fam}_{len(items):05d}",
                 input_prompt=U.build_chat_prompt(body, a.model, tok, None),
                 dataset=fam, task_type=tt, generation_length=gen_len,
                 gold=gold, metric=metric, fmt=fi % len(fmts), **meta)
        if allc:
            d["all_classes"] = allc
        items.append(d)

    # 1) LooGLE shortdep_qa: real long documents with human-written free-form short answers
    rows = rd("loogle_shortqa"); rng.shuffle(rows)
    made = 0
    for r in rows:
        if made >= a.per_family:
            break
        ev = r.get("evidence")
        ev = as_list(r.get("evidence"))
        ct = CTX[made % len(CTX)]; dp = DEPTHS[made % len(DEPTHS)]
        w = window(tok, r["context"], ev, ct, dp)
        if w is None:
            drop += 1; continue
        emit("loogle_qa", "Single-doc QA", 128, "qa_f1_score", w[0], r["question"], [r["answer"]],
             dict(source="LooGLE/shortdep_qa", source_split="test", source_idx=r["id"],
                  ctx_tokens=ct, evidence_depth=round(w[1], 3)), F_QA, made)
        made += 1

    # 2) LooGLE longdep_qa: multiple choice, options inline in the question, answer is an index
    rows = rd("loogle_longqa"); rng.shuffle(rows)
    made = 0
    for r in rows:
        if made >= a.per_family:
            break
        ev = r.get("evidence")
        ev = as_list(r.get("evidence"))
        ct = CTX[made % len(CTX)]; dp = DEPTHS[made % len(DEPTHS)]
        w = window(tok, r["context"], ev, ct, dp)
        if w is None:
            drop += 1; continue
        opts = re.findall(r"^\s*(\d)\.", r["question"], flags=re.M)
        emit("loogle_mc", "Few Shot", 32, "classification_score", w[0], r["question"],
             [str(r["answer"])],
             dict(source="LooGLE/longdep_qa", source_split="test", source_idx=r["id"],
                  ctx_tokens=ct, evidence_depth=round(w[1], 3)), F_MC, made,
             allc=opts or ["1", "2", "3", "4"])
        made += 1

    # 3) LooGLE summarization: generation length 512, a regime v5 did not cover
    rows = rd("loogle_summ"); rng.shuffle(rows)
    for j, r in enumerate(rows[:a.per_family]):
        ct = CTX[j % len(CTX)]
        ids = tok(r["context"], add_special_tokens=False).input_ids[:ct]
        if len(ids) < ct * 0.6:
            drop += 1; continue
        emit("loogle_summ", "Summarization", 512, "rouge_score",
             tok.decode(ids, skip_special_tokens=True), "", [r["answer"]],
             dict(source="LooGLE/summarization", source_split="test", source_idx=r["id"],
                  ctx_tokens=len(ids), evidence_depth=-1.0), F_SUM, j)

    # 3b) Paragraph identification over real documents: the same task shape as LongBench's
    # passage_retrieval, on unrelated data. This cell carries by far the largest cost for a
    # wrong action (on mistral, passage_retrieval_en spans 57.70 points, from 10:32 at 74.78
    # down to 0:1 at 17.08), yet the recipe's existing retrieval families teach nothing about
    # it: synth_needle sits at P0 0.02-0.04 (no signal) and synth_retrieval at 0.89-0.92
    # (saturated). Here a real long document is split into numbered paragraphs and one of its
    # sentences is quoted back.
    rows = rd("loogle_shortqa"); rng.shuffle(rows)
    made = 0
    for r in rows:
        if made >= a.per_family:
            break
        ct = CTX[made % len(CTX)]; dp = DEPTHS[made % len(DEPTHS)]
        ids = tok(r["context"], add_special_tokens=False).input_ids
        if len(ids) < ct:
            continue
        st = rng.randint(0, max(0, len(ids) - ct))
        seg = tok.decode(ids[st:st + ct], skip_special_tokens=True)
        # LooGLE documents carry no newlines, so paragraphs are formed by grouping sentences
        # into ~800-character chunks. These are contiguous spans of a real document, not
        # synthesized text.
        sent = [x.strip() for x in re.split(r"(?<=[.!?])\s+", seg) if x.strip()]
        paras, cur = [], ""
        for sx in sent:
            cur = (cur + " " + sx).strip()
            if len(cur) >= 800:
                paras.append(cur); cur = ""
        if len(cur) > 300:
            paras.append(cur)
        if len(paras) < 6:
            continue
        k = min(int(round(dp * (len(paras) - 1))), len(paras) - 1)
        sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", paras[k]) if len(x.strip()) > 80]
        if len(sents) < 2:
            continue
        probe = sents[len(sents) // 2]
        ctx = "\n\n".join(f"Paragraph {i+1}:\n{p}" for i, p in enumerate(paras))
        emit("para_id_real", "Passage Retrieval", 32, "retrieval_score", ctx, probe,
             [f"Paragraph {k+1}"],
             dict(source="LooGLE document, paragraphs numbered", source_split="test", source_idx=r["id"],
                  ctx_tokens=ct, evidence_depth=dp), F_PID, made)
        made += 1

    # 4) QuALITY: multiple choice over Gutenberg short stories and magazine articles
    rows = rd("quality_mc"); rng.shuffle(rows)
    for j, r in enumerate(rows[:a.per_family]):
        ids = tok(r["article"], add_special_tokens=False).input_ids[:CTX[j % len(CTX)]]
        if len(ids) < 1500:
            drop += 1; continue
        o = r["options"] if isinstance(r["options"], list) else json.loads(r["options"])
        q = r["question"] + "\n" + "\n".join(f"{i+1}. {x}" for i, x in enumerate(o))
        emit("quality_mc", "Few Shot", 32, "classification_score",
             tok.decode(ids, skip_special_tokens=True), q, [str(int(r["answer"]))],
             dict(source="QuALITY", source_split="train", source_idx=str(j),
                  ctx_tokens=len(ids), evidence_depth=-1.0), F_MC, j,
             allc=[str(i + 1) for i in range(len(o))])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    import collections
    c = collections.Counter(i["dataset"] for i in items)
    print(f"[rv6] {len(items)} rows -> {a.out}  ({drop} dropped, mostly evidence not found)")
    for k, v in sorted(c.items()):
        print(f"   {k:16s} {v}")


if __name__ == "__main__":
    main()
