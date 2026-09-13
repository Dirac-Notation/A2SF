"""Generate diverse long-document QA to fill the (medium generation, QA) cell. No LongBench.

The earlier recipe drew all of its QA from SQuAD, which only covers short extractive answers.
That misrepresents datasets like qasper, where the evidence is spread across the document, and
the (M, QA) cell is mis-trained as a result.

Two axes are varied together:

  question type    local (a fact stated in one place) / synth (requires combining places)
  instruction form four templates, none copied from LongBench wording

Questions are written by the model after reading the document (self-instruct). P0 needs no
reference answer, so these are valid training rows even without a gold answer.
"""
import argparse
import json
import os
import sys

TRACES = os.environ.get("ICLR_TRACES", "/data2/smp9898/iclr_traces")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

QGEN = {
    "local": ("Read the passage below.\n\n{doc}\n\nWrite one question whose answer is a single "
              "specific fact stated somewhere in the passage. Output only the question.\n\nQuestion:"),
    "synth": ("Read the passage below.\n\n{doc}\n\nWrite one question that can only be answered by "
              "combining information from several different parts of the passage. Output only the "
              "question.\n\nQuestion:"),
    # unans: questions inside the passage's topic that the passage does not answer.
    # Motivation: on qasper's unanswerable subset H2O's advantage is largest (+6.27 vs +2.91
    # on answerable ones).
    "unans": ("Read the passage below.\n\n{doc}\n\nWrite one question that is about the same topic "
              "as the passage but whose answer is NOT stated anywhere in it. Output only the "
              "question.\n\nQuestion:"),
}

# four instruction templates, none copied from LongBench wording
FORMATS = [
    "Use the passage to answer the question.\n\n<passage>\n{doc}\n</passage>\n\nQuestion: {q}\nAnswer:",
    "{doc}\n\nBased on the text above, answer the following.\nQ: {q}\nA:",
    "Here is a document.\n\n{doc}\n\nAnswer this about the document, briefly.\n{q}\n",
    "Read the material and respond.\n\nMATERIAL:\n{doc}\n\nREQUEST: {q}\nRESPONSE:",
]
# templates that warn an answer may be absent; same intent as qasper, different wording
FORMATS_ESC = [
    "Use the passage to answer the question. Reply with \"not stated\" if the passage does not "
    "contain the answer.\n\n<passage>\n{doc}\n</passage>\n\nQuestion: {q}\nAnswer:",
    "{doc}\n\nAnswer the question from the text above. If the text does not say, write "
    "\"not stated\".\nQ: {q}\nA:",
]


def build_docs_pubmed(tok, n, lengths, seed=0, path=TRACES + "/docqa/pubmed_docs.json"):
    """Build documents from scientific paper bodies.

    The domain does not overlap qasper (biomedical rather than NLP) while the structure does:
    sectioned, dense, heavy on terminology. Unrelated to every LongBench dataset."""
    src = json.load(open(path))
    rng = np.random.RandomState(seed + 777)
    order = rng.permutation(len(src))
    docs = []
    for j in order:
        L = int(lengths[len(docs) % len(lengths)])
        ids = tok(src[int(j)], add_special_tokens=False).input_ids
        if len(ids) < L * 0.6:
            continue
        docs.append(tok.decode(ids[:L], skip_special_tokens=True))
        if len(docs) >= n:
            break
    return docs


def build_docs(tok, n, lengths, seed=0):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.RandomState(seed)
    docs, i = [], 0
    while len(docs) < n:
        L = int(lengths[len(docs) % len(lengths)])
        start = int(rng.randint(0, len(ds) - 4000))
        buf, cur = [], 0
        j = start
        while cur < L and j < len(ds):
            t = ds[j]["text"]
            if t.strip():
                buf.append(t); cur += len(tok(t, add_special_tokens=False).input_ids)
            j += 1
        txt = "".join(buf)
        ids = tok(txt, add_special_tokens=False).input_ids[:L]
        docs.append(tok.decode(ids, skip_special_tokens=True))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--lengths", default="3000,5000,8000,11000")
    ap.add_argument("--gen_len", type=int, default=128)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default="wikitext", choices=["wikitext", "pubmed"])
    ap.add_argument("--qmodes", default="local,synth")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.chdir(ROOT)
    import utils as U

    model, tok = U.load_model(args.model)
    model.init_cache(None)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lengths = [int(x) for x in args.lengths.split(",")]
    qmodes = args.qmodes.split(",")
    docs = (build_docs_pubmed if args.source == "pubmed" else build_docs)(
        tok, args.n, lengths, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["sample_id"] for l in open(args.out)}

    with torch.no_grad():
        for i, doc in enumerate(docs):
            sid = f"docqa_{args.source[:3]}_s{args.seed}_{i:04d}"
            if sid in done:
                continue
            mode = qmodes[i % len(qmodes)]
            qp = QGEN[mode].format(doc=doc)
            ids = tok(qp, return_tensors="pt").input_ids.to(model.device)
            out = model.generate(input_ids=ids, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.eos_token_id, num_logits_to_keep=1)[0]
            q = tok.decode(out[ids.shape[-1]:], skip_special_tokens=True).strip().split("\n")[0]
            if "?" in q:
                q = q[:q.index("?") + 1]                 # keep only up to the first question mark, dropping trailing commentary
            q = q.strip().strip('"')
            if len(q) < 12 or not q.endswith("?"):
                continue
            # half of the unanswerable questions get an escape template, to create a no-answer signal
            if mode == "unans" and i % 2 == 0:
                fmt, fid = FORMATS_ESC[i % len(FORMATS_ESC)], 100 + i % len(FORMATS_ESC)
            else:
                fmt, fid = FORMATS[i % len(FORMATS)], i % len(FORMATS)
            body = fmt.format(doc=doc, q=q)
            prompt = U.build_chat_prompt(body, args.model, tok, None)
            with open(args.out, "a") as f:
                f.write(json.dumps({"sample_id": sid, "input_prompt": prompt,
                                    "dataset": f"docqa_{mode}", "task_type": "Single-doc QA",
                                    "generation_length": args.gen_len,
                                    "qmode": mode, "fmt": fid, "source": args.source},
                                   ensure_ascii=False) + "\n")
            if i % 40 == 0:
                print(f"[docqa] {i}/{len(docs)} mode={mode} q={q[:60]!r}", flush=True)
    print(f"[docqa] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
