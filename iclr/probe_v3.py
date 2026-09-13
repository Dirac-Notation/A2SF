"""Router input features from a single short probe forward pass.

The probe costs one short prefill (~300 tokens) plus one generated token, independent of the
real prompt length, and yields three feature groups:

  1. an output-format class over 8 options, which also encodes whether the evidence is local
     or spread through the document
  2. the full softmax distribution over those options, not just the argmax
  3. surface features computed from the prompt string alone (no model call)

With --dump_hidden the last hidden state of the same forward pass is stored as well, so the
PCA features used by the router come at no extra compute.

  python iclr/probe_v3.py --model llama3-8b --corpus lb --gpu 3
  python iclr/probe_v3.py --model llama3-8b --corpus recipe \
      --recipe datasets/training/raw/recipe_poolx_llama3-8b.jsonl --gpu 3
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# v1 is the validated 5-option original (90-100% consistent on LB); v4 keeps v1's wording but
# splits A into local and spread.
TMPL_V1 = """Below are the beginning and the end of a task prompt.

[BEGIN] {head} [...] {tail} [END]

What output format does this task require? Choose one:
A. a short answer of a few words
B. a long summary, report, or dialogue summary
C. source code
D. a category or type label from a fixed set
E. a number or a paragraph identifier

Answer:"""

TMPL_V4 = """Below are the beginning and the end of a task prompt.

[BEGIN] {head} [...] {tail} [END]

What output format does this task require? Choose one:
A. a short answer of a few words, found in one or two specific places
B. a short answer of a few words, gathered from across the whole text
C. a long summary, report, or dialogue summary
D. source code
E. a category or type label from a fixed set
F. a number or a paragraph identifier

Answer:"""

# v5 is a flat design: every option is a concrete task description at the same level, with the
# two-stage decision removed.
TMPL_V5 = """Below are the beginning and the end of a task prompt.

[BEGIN] {head} [...] {tail} [END]

Which of these best describes the task? Choose one:
A. answer a question using one specific fact stated somewhere in the text
B. answer a question by combining facts from several different places
C. write a long summary or report of the whole text
D. summarize a conversation or a meeting
E. continue or complete source code
F. assign a category or type label from a fixed set
G. count things in the text and report a number
H. say which paragraph or passage contains something
I. answer in the same pattern as the worked examples given

Answer:"""

VARIANTS = {"v1s": (TMPL_V1, "ABCDE"), "v4": (TMPL_V4, "ABCDEF"), "v5": (TMPL_V5, "ABCDEFGHI"),
            "v1t": (TMPL_V1, "ABCDE")}   # v1t is v1 wording with token-level slicing


def surface(p):
    """Surface features computed from the string alone, with no model call."""
    n = max(len(p), 1)
    lines = p.split("\n")
    nonalpha = sum(1 for c in p if not (c.isalnum() or c.isspace())) / n
    # few-shot signal: repeated line-prefix patterns
    heads = [l[:14] for l in lines if len(l) > 14]
    rep = 0
    if heads:
        from collections import Counter
        rep = Counter(heads).most_common(1)[0][1] / max(len(heads), 1)
    tail = p[-300:]
    return [
        float(np.log1p(len(p)) / 12.0),
        float(np.log1p(len(lines)) / 8.0),
        float(nonalpha),
        float(rep),
        float("?" in tail),
        float(len(re.findall(r"\n\s*\n", p)) / max(len(lines), 1)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True, choices=["lb", "recipe", "ruler"])
    ap.add_argument("--recipe", default=None)
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--n_per_task", type=int, default=50)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--variant", default="v1t", choices=["v1s","v4","v5","v1t"])
    ap.add_argument("--slice_tokens", type=int, default=128, help="v1t: N tokens from each end")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dump_hidden", action="store_true",
                    help="also store the last hidden state of the same forward pass, at no extra cost")
    ap.add_argument("--out_root", default=os.environ.get("ICLR_TRACES", "/data2/smp9898/iclr_traces"),
                    help="root for the npz output; used on hosts without /data2")
    args = ap.parse_args()
    TMPL, OPTS = VARIANTS[args.variant]
    if args.tag is None:
        args.tag = f"v1t{args.slice_tokens}" if args.variant=="v1t" else args.variant

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.expanduser("~/hf_datasets"))
    os.chdir(ROOT)
    import utils

    model, tok = utils.load_model(args.model)
    model.init_cache(None)
    opt_ids = [tok.encode(f" {c}", add_special_tokens=False)[-1] for c in OPTS]

    items = []                       # (key, prompt)
    if args.corpus == "lb":
        for ds in sorted(json.load(open("config/dataset2maxlen.json"))):
            path = f"datasets/longbench/{ds}.jsonl"
            if not os.path.exists(path):
                continue
            for o in (json.loads(l) for l in open(path)):
                items.append((f"{ds}|{o['idx']}", o["input_prompt"]))
    elif args.corpus == "recipe":
        for o in (json.loads(l) for l in open(args.recipe)):
            items.append((str(o["sample_id"]), o["input_prompt"]))
    else:
        from datasets import load_dataset
        ds = load_dataset("simonjegou/ruler", str(args.length), split="test")
        cnt = {}
        for r in ds:
            t = r["task"]
            if cnt.get(t, 0) >= args.n_per_task:
                continue
            cnt[t] = cnt.get(t, 0) + 1
            items.append((f"{t}|{cnt[t]-1}", r["context"] + "\n\n" + r["question"]))

    keys, probs, feats, hids = [], [], [], []
    with torch.no_grad():
        for j, (k, p) in enumerate(items):
            if args.variant == "v1t":
                pid = tok(p, add_special_tokens=False).input_ids
                n = args.slice_tokens
                head = tok.decode(pid[:n], skip_special_tokens=True)
                tail = tok.decode(pid[-n:], skip_special_tokens=True)
            else:
                head, tail = p[:400], p[-400:]
            q = TMPL.format(head=head, tail=tail)
            ids = tok(q, return_tensors="pt").input_ids.to("cuda")
            # Read the last-position hidden state from the same forward pass. There is no extra
            # computation, so deployment overhead is unchanged (still one probe forward), and it
            # preserves information that collapsing to 5 option probabilities would discard.
            o = model(ids, output_hidden_states=args.dump_hidden)
            lg = o.logits[0, -1]
            v = torch.tensor([float(lg[i]) for i in opt_ids])
            keys.append(k)
            probs.append(torch.softmax(v, 0).numpy())
            feats.append(surface(p))
            if args.dump_hidden:
                hids.append(o.hidden_states[-1][0, -1].float().cpu().numpy())
            if j % 500 == 0:
                print(f"[v3] {j}/{len(items)}", flush=True)
    out = f"{args.out_root}/probe_{args.tag}"
    os.makedirs(out, exist_ok=True)
    name = args.corpus if args.corpus != "ruler" else f"ruler{args.length}"
    payload = dict(key=np.array(keys), prob=np.stack(probs).astype(np.float32),
                   surf=np.array(feats, dtype=np.float32))
    if hids:
        payload["hid"] = np.stack(hids).astype(np.float16)
    np.savez_compressed(f"{out}/{args.model}_{name}.npz", **payload)
    print(f"[v3] saved {out}/{args.model}_{name}.npz ({len(keys)})", flush=True)


if __name__ == "__main__":
    main()
