"""recipe-v5: a long-context training corpus that carries reference answers.

Why the rewrite. The earlier corpus had no references, so the only available reward was
P0 = Tanimoto(compressed output, full-cache output). Computing P0 directly on LongBench
prompts shows it picks the wrong action in the QA regime: for the (M, QA) cell the P0-optimal
action is 10:16 on all three models, while the score-optimal one is 10:1 (llama) or 0:1 (qwen,
mistral), costing 1.86 / 0.24 / 2.32 points. The (L, summarization) cell loses 0.00. The cause
is that the full-cache output is itself often wrong on hard QA, and maximizing similarity to a
wrong reference does not raise the score; more data does not fix it.

v5 therefore attaches a gold reference to every row and scores the reward with the task's own
metric (F1 / ROUGE / classification match / retrieval accuracy). LongBench remains a pure
evaluation benchmark and is never used for training.

Construction:

  1. Source separation   only public corpora disjoint from the 16 LongBench datasets
                         (see the data card in datasets/fetch_rv5_sources.py)
  2. Evidence position   long contexts are assembled from an evidence passage plus distractors,
                         with the evidence depth spread evenly over
                         {0.05, 0.275, 0.5, 0.725, 0.95} (the position control of Liu et al.,
                         2023). Self-generated questions are avoided here because their
                         evidence concentrates in the recent window.
  3. Context length      {4k, 8k, 16k} tokens, evenly allocated
  4. Instruction form    four templates per task family, none copied from LongBench
  5. Generation length   all of S (<=32) / M (<=128) / L (>128) are filled

Each row keeps what reproduction needs: source, source_split, source_idx, ctx_tokens,
evidence_depth, fmt, gold, metric.

  python datasets/build_recipe_v5.py --model llama3-8b --per_family 120 --gpu 0 \
      --out datasets/training/raw/rv5_prompts.jsonl

The model argument is used only for tokenizer length accounting and the chat template;
generation itself is done by iclr/gen_actions.py.
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SRC = os.environ.get("ICLR_TRACES", "/data2/smp9898/iclr_traces") + "/rv5/src"
DEPTHS = [0.05, 0.275, 0.5, 0.725, 0.95]
CTX = [4000, 8000, 16000]

# instruction templates, none copied from LongBench wording; {ctx} and {q} are filled in
F_QA = [
    "Use the material below to answer the question.\n\n<material>\n{ctx}\n</material>\n\nQuestion: {q}\nAnswer:",
    "{ctx}\n\nBased on the text above, answer the following.\nQ: {q}\nA:",
    "Here are some documents.\n\n{ctx}\n\nAnswer this briefly, using only the documents.\n{q}\n",
    "Read the material and respond.\n\nMATERIAL:\n{ctx}\n\nREQUEST: {q}\nRESPONSE:",
]
F_QA_ESC = [
    "Use the material to answer. If the material does not contain the answer, reply exactly "
    "\"unanswerable\".\n\n<material>\n{ctx}\n</material>\n\nQuestion: {q}\nAnswer:",
    "{ctx}\n\nAnswer from the text above. If the text does not say, write \"unanswerable\".\nQ: {q}\nA:",
]
F_MC = [
    "{ctx}\n\nQuestion: {q}\nRespond with the letter of the best option only.\nAnswer:",
    "Read the passage.\n\n{ctx}\n\n{q}\nGive only the letter (A, B, C or D).\nAnswer:",
]
F_SUM = [
    "{ctx}\n\nWrite a concise summary of the document above.\nSummary:",
    "Summarize the following document.\n\n<document>\n{ctx}\n</document>\n\nSummary:",
    "Below is a document. Produce a short summary of its contents.\n\n{ctx}\n\nShort summary:",
    "DOCUMENT:\n{ctx}\n\nTASK: summarize the document.\nOUTPUT:",
]
F_CLS = [
    "{ctx}\nText: {q}\nLabel:",
    "Classify the text into one of the labels shown in the examples.\n\n{ctx}\nText: {q}\nLabel:",
]
F_CODE = [
    "{ctx}\n\nComplete the next line of the code above.\nNext line:",
    "Here is a Python file.\n\n{ctx}\n\nWrite the single line that comes next.\n",
]
F_KV = [
    "{ctx}\n\nWhat is the value associated with key {q}? Give only the value.\nValue:",
    "Below are key-value records.\n\n{ctx}\n\nRetrieve the value for key {q}.\nValue:",
]
F_PID = [
    "{ctx}\n\nThe following sentence was taken from one of the passages above.\n{q}\n"
    "Which passage is it from? Answer in the form \"Paragraph N\".\nAnswer:",
    "Below are numbered passages.\n\n{ctx}\n\nSentence: {q}\nIdentify the passage it came from "
    "(reply \"Paragraph N\").\nAnswer:",
]


def rd(name, limit=None):
    p = os.path.join(SRC, f"{name}.jsonl")
    out = []
    for i, l in enumerate(open(p)):
        if limit and i >= limit:
            break
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def assemble(tok, evidence, distractors, ctx_tokens, depth):
    """Place the evidence passage among distractors at the given depth, padding to the target
    token count."""
    ev_n = len(tok(evidence, add_special_tokens=False).input_ids)
    budget = max(ctx_tokens - ev_n, 0)
    picked, used = [], 0
    for d in distractors:
        n = len(tok(d, add_special_tokens=False).input_ids)
        if used + n > budget:
            continue
        picked.append(d); used += n
        if used >= budget * 0.97:
            break
    pos = int(round(depth * len(picked)))
    parts = picked[:pos] + [evidence] + picked[pos:]
    return "\n\n".join(f"Passage {i+1}:\n{p}" for i, p in enumerate(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--per_family", type=int, default=120)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="build only one family (code / doc)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(ROOT)
    import utils as U
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(json.load(open("config/model2path.json"))[a.model])
    rng = random.Random(a.seed)
    items = []

    stats = {"emitted": 0, "dropped_no_evidence": 0}

    def emit(fam, task_type, gen_len, metric, ctx, q, gold, meta, fmts, fi, verify=False):
        # an extractive row is only valid if the answer string actually occurs in the context
        if verify and not any(g and g.lower() in ctx.lower() for g in gold):
            stats["dropped_no_evidence"] += 1
            return
        stats["emitted"] += 1
        body = fmts[fi % len(fmts)].format(ctx=ctx, q=q)
        items.append(dict(sample_id=f"rv5_{fam}_{len(items):05d}",
                          input_prompt=U.build_chat_prompt(body, a.model, tok, None),
                          dataset=fam, task_type=task_type, generation_length=gen_len,
                          gold=gold, metric=metric, fmt=fi % len(fmts), **meta))

    # ---------- 1) single-document QA from SQuAD v2 (short / medium / unanswerable) ----------
    sq = rd("squad_v2", 40000)
    ans = [r for r in sq if r.get("answers", {}).get("text")]
    una = [r for r in sq if not r.get("answers", {}).get("text")]
    pool = [r["context"] for r in sq[:8000]]
    for fam, src_rows, gen_len, fmts, is_una in [
            ("sqa_short", ans, 32, F_QA, False),
            ("sqa_mid", ans, 128, F_QA, False),
            ("sqa_unans", una, 128, F_QA_ESC, True)]:
        rng.shuffle(src_rows)
        for j in range(a.per_family):
            r = src_rows[j % len(src_rows)]
            ct = CTX[j % len(CTX)]; dp = DEPTHS[j % len(DEPTHS)]
            dis = [c for c in rng.sample(pool, 220) if c != r["context"]]
            ctx = assemble(tok, r["context"], dis, ct, dp)
            gold = ["unanswerable"] if is_una else list(dict.fromkeys(r["answers"]["text"]))
            emit(fam, "Single-doc QA", gen_len, "qa_f1_score", ctx, r["question"], gold,
                 dict(source="squad_v2", source_split="train", source_idx=r.get("id", ""),
                      ctx_tokens=ct, evidence_depth=dp), fmts, j, verify=not is_una)

    # ---------- 1b) QA within one document (coherent long-form reading) ----------
    # Measured: padding the evidence with unrelated distractors turns the task into retrieval,
    # where the mid-size window (10:16) wins. LongBench's single-document QA (qasper,
    # narrativeqa) is instead one topic throughout, which is
    # coherent single-document reading, whose optimal actions sit at the extremes (10:1 / 0:1).
    # So SQuAD paragraphs are
    # grouped by title and only other paragraphs of the same document are used as context,
    # which keeps the topic coherent.
    if not a.only or a.only == "doc":
        bytitle = {}
        for r in sq:
            bytitle.setdefault(r.get("title", ""), []).append(r)
        titles = [t for t, v in bytitle.items() if len(v) >= 25]
        rng.shuffle(titles)
        made = 0
        for t in titles:
            if made >= a.per_family:
                break
            items_t = bytitle[t]
            tgt = next((r for r in items_t if r.get("answers", {}).get("text")), None)
            if tgt is None:
                continue
            paras, seen_p = [], set()
            for r in items_t:
                c = r["context"]
                if c not in seen_p:
                    seen_p.add(c); paras.append(c)
            others = [c for c in paras if c != tgt["context"]]
            ct = CTX[made % len(CTX)]; dp = DEPTHS[made % len(DEPTHS)]
            ctx = assemble(tok, tgt["context"], others, ct, dp)
            gold = list(dict.fromkeys(tgt["answers"]["text"]))
            emit("sqa_doc", "Single-doc QA", 128, "qa_f1_score", ctx, tgt["question"], gold,
                 dict(source="squad_v2 (paragraphs sharing a title)", source_split="train",
                      source_idx=tgt.get("id", ""), ctx_tokens=ct, evidence_depth=dp),
                 F_QA, made, verify=True)
            made += 1

    # ---------- 2) reasoning QA from DROP ----------
    dr = [r for r in rd("drop", 30000) if r.get("answers_spans", {}).get("spans")]
    dpool = [r["passage"] for r in dr[:6000]]
    rng.shuffle(dr)
    for j in range(a.per_family):
        r = dr[j % len(dr)]
        ct = CTX[j % len(CTX)]; dp = DEPTHS[j % len(DEPTHS)]
        ctx = assemble(tok, r["passage"], [c for c in rng.sample(dpool, 200) if c != r["passage"]], ct, dp)
        emit("drop_qa", "Multi-doc QA", 128, "qa_f1_score", ctx, r["question"],
             list(dict.fromkeys(r["answers_spans"]["spans"])),
             dict(source="drop", source_split="train", source_idx=r.get("query_id", ""),
                  ctx_tokens=ct, evidence_depth=dp), F_QA, j, verify=True)

    # ---------- 3) four-way multiple choice from RACE ----------
    rc = rd("race", 20000)
    rpool = [r["article"] for r in rc[:5000]]
    rng.shuffle(rc)
    for j in range(a.per_family):
        r = rc[j % len(rc)]
        ct = CTX[j % len(CTX)]; dp = DEPTHS[j % len(DEPTHS)]
        ctx = assemble(tok, r["article"], [c for c in rng.sample(rpool, 200) if c != r["article"]], ct, dp)
        opts = "\n".join(f"{c}. {o}" for c, o in zip("ABCD", r["options"]))
        emit("race_mc", "Few Shot", 16, "classification_score", ctx,
             f"{r['question']}\n{opts}", [r["answer"]],
             dict(source="race", source_split="train", source_idx=str(r.get("example_id", "")),
                  ctx_tokens=ct, evidence_depth=dp, all_classes=list("ABCD")), F_MC, j)

    # ---------- 4) summarization (BillSum / PubMed / CNN-DM) ----------
    # Summarization contexts are never padded, which would change the task. Instead only
    # documents that are genuinely as long as the target are selected. CNN/DM is excluded: its
    # articles run about 800 tokens, too short for the long-context regime.
    bs = [r for r in rd("billsum", 18949) if len(r.get("text", "")) > 12000]
    for fam, rows, tf, gf, gen_len, src in [
            ("billsum_summ", bs, "text", "summary", 256, "billsum")]:
        # Documents are neither padded nor truncated here. Only documents above 3k tokens are
        # used, at their natural length (capped at 16k), so the length distribution spreads on
        # its own.
        rng.shuffle(rows)
        for j in range(a.per_family):
            cand = rows[j % len(rows)]
            ids = tok(cand[tf], add_special_tokens=False).input_ids[:16000]
            if len(ids) < 3000:
                continue
            emit(fam, "Summarization", gen_len, "rouge_score",
                 tok.decode(ids, skip_special_tokens=True), "", [cand[gf]],
                 dict(source=src, source_split="train", source_idx=str(j),
                      ctx_tokens=len(ids), evidence_depth=-1.0), F_SUM, j)
    # PubMed needs the body/abstract pair, so the source is read again
    try:
        import pyarrow.parquet as pq, io, urllib.request as ur
        t = pq.read_table(io.BytesIO(ur.urlopen(
            "https://huggingface.co/api/datasets/ccdv/pubmed-summarization/parquet/section/test/0.parquet",
            timeout=600).read())).to_pylist()
        t = [r for r in t if len(r.get("article", "")) > 12000 and len(r.get("abstract", "")) > 200]
        rng.shuffle(t)
        for j in range(a.per_family):
            r = t[j % len(t)]
            ids = tok(r["article"], add_special_tokens=False).input_ids[:16000]
            if len(ids) < 3000:
                continue
            emit("pubmed_summ", "Summarization", 256, "rouge_score",
                 tok.decode(ids, skip_special_tokens=True), "", [r["abstract"]],
                 dict(source="pubmed", source_split="test", source_idx=str(j),
                      ctx_tokens=len(ids), evidence_depth=-1.0), F_SUM, j)
    except Exception as e:
        print(f"[rv5] pubmed skipped: {type(e).__name__} {e}", flush=True)

    # ---------- 5) few-shot classification (AG News / banking77) ----------
    AG = ["World", "Sports", "Business", "Sci/Tech"]
    ag = rd("ag_news", 40000)
    bk = rd("banking77", 10003)
    bk_labels = sorted({str(r["label"]) for r in bk})
    for fam, rows, labels, src in [("agnews_cls", ag, AG, "ag_news"),
                                   ("banking_cls", bk, bk_labels, "banking77")]:
        rng.shuffle(rows)
        for j in range(a.per_family):
            tgt = rows[j % len(rows)]
            shots, used = [], 0
            ct = CTX[j % len(CTX)]
            for r in rows:
                if r is tgt:
                    continue
                lab = labels[int(r["label"])] if fam == "agnews_cls" else str(r["label"])
                s = f"Text: {r['text']}\nLabel: {lab}"
                n = len(tok(s, add_special_tokens=False).input_ids)
                if used + n > ct:
                    break
                shots.append(s); used += n
            gold = labels[int(tgt["label"])] if fam == "agnews_cls" else str(tgt["label"])
            emit(fam, "Few Shot", 16, "classification_score",
                 "\n\n".join(shots), tgt["text"], [gold],
                 dict(source=src, source_split="train", source_idx=str(j),
                      ctx_tokens=used, evidence_depth=-1.0, all_classes=labels), F_CLS, j)

    # ---------- 5b) next-line code completion (CodeSearchNet Python) ----------
    # Reproduces the task shape of LCC and RepoBench-P (predict the next line given repository
    # context) without using their data: functions from the same repository are concatenated and
    # the last one is cut mid-body.
    if not a.only or a.only == "code":
        os.environ.setdefault("HF_DATASETS_CACHE", "/data2/shared/huggingface_cache/datasets")
        from datasets import load_dataset
        cs = load_dataset("code_search_net", "python", split="train")
        byrepo = {}
        for i in range(0, 120000, 3):
            r = cs[i]
            byrepo.setdefault(r["repository_name"], []).append(r["whole_func_string"])
        repos = [k for k, v in byrepo.items() if len(v) >= 12]
        rng.shuffle(repos)
        made = 0
        for rp in repos:
            if made >= a.per_family:
                break
            fns = byrepo[rp]
            ct = CTX[made % len(CTX)]
            ctx_parts, used = [], 0
            for fn in fns:
                n = len(tok(fn, add_special_tokens=False).input_ids)
                if used + n > ct:
                    break
                ctx_parts.append(fn); used += n
            if len(ctx_parts) < 3 or used < ct * 0.4:
                continue
            tail = ctx_parts[-1].split("\n")
            cut = max(2, int(len(tail) * 0.6))
            gold_line = next((l for l in tail[cut:] if l.strip()), "")
            if not gold_line.strip():
                continue
            ctx_parts[-1] = "\n".join(tail[:cut])
            emit("code_next", "Code Complete", 64, "code_sim_score",
                 "\n\n".join(ctx_parts), "", [gold_line.strip()],
                 dict(source="code_search_net(python)", source_split="train", source_idx=rp,
                      ctx_tokens=used, evidence_depth=-1.0), F_CODE, made)
            made += 1

    # ---------- 5c) paragraph identification (passage_retrieval's shape, other data) ----------
    # the reference must read "Paragraph N" for the evaluation's retrieval_score to work
    if not a.only or a.only == "code":
        import re as _re
        for j in range(a.per_family):
            ct = CTX[j % len(CTX)]; dp = DEPTHS[j % len(DEPTHS)]
            cands = rng.sample(pool, 200)
            picked, used = [], 0
            for c in cands:
                n = len(tok(c, add_special_tokens=False).input_ids)
                if used + n > ct:
                    continue
                picked.append(c); used += n
                if used >= ct * 0.97:
                    break
            if len(picked) < 5:
                continue
            k = min(int(round(dp * (len(picked) - 1))), len(picked) - 1)
            sents = [x.strip() for x in _re.split(r"(?<=[.!?])\s+", picked[k]) if len(x.strip()) > 60]
            if len(sents) < 2:
                continue
            probe = sents[len(sents) // 2]
            ctx = "\n\n".join(f"Paragraph {i+1}:\n{p}" for i, p in enumerate(picked))
            emit("para_id", "Passage Retrieval", 32, "retrieval_score", ctx, probe,
                 [f"Paragraph {k+1}"],
                 dict(source="squad_v2 paragraphs, numbered", source_split="train", source_idx=str(j),
                      ctx_tokens=used, evidence_depth=dp), F_PID, j)

    # ---------- 6) synthetic key-value retrieval (controlled evidence position) ----------
    import uuid
    wt = [c for c in (r["context"] for r in sq[:6000]) if len(c) > 200]
    for j in range(a.per_family):
        ct = CTX[j % len(CTX)]; dp = DEPTHS[j % len(DEPTHS)]
        key = uuid.UUID(int=rng.getrandbits(128)).hex[:12]
        val = uuid.UUID(int=rng.getrandbits(128)).hex[:12]
        ev = f"Record {key}: value = {val}"
        dis = [f"Record {uuid.UUID(int=rng.getrandbits(128)).hex[:12]}: value = "
               f"{uuid.UUID(int=rng.getrandbits(128)).hex[:12]}\n{c[:400]}" for c in rng.sample(wt, 200)]
        emit("kv_retrieval", "Passage Retrieval", 16, "retrieval_score",
             assemble(tok, ev, dis, ct, dp), key, [val],
             dict(source="synthetic: uuid4 keys with squad_v2 paragraphs as distractors", source_split="-",
                  source_idx=str(j), ctx_tokens=ct, evidence_depth=dp),
             F_KV, j, verify=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    import collections
    c = collections.Counter(i["dataset"] for i in items)
    print(f"[rv5] {len(items)} rows -> {a.out}  ({stats['dropped_no_evidence']} dropped, evidence absent)")
    for k, v in sorted(c.items()):
        print(f"   {k:16s} {v}")


if __name__ == "__main__":
    main()
