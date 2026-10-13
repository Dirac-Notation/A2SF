"""Shared setup for the paper's analysis scripts (Section 2 observations, NIAH, cost).

The Section 2 measurements use the first `per` prompts of each of the 16 English LongBench tasks, truncated in
the middle to 8,192 tokens minus the task's generation length, and keep only prompts of at least 1,024 tokens.
With per=20 this gives the 315 prompts of the paper.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATASETS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum",
            "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
MAXLEN, MIN_LEN = 8192, 1024
ATTN_IMPL = "kvcompress"                    # the attention implementation the pipeline model runs by default


def load_rows(data_dir, per, shard="0/1", reverse=False):
    """[(dataset, idx, row)] for the first `per` rows of every task, split as shard 'i/n'."""
    rows = []
    for d in DATASETS:
        with open(os.path.join(data_dir, f"{d}.jsonl"), encoding="utf-8") as f:
            rr = [json.loads(l) for l in f if l.strip()][:per]
        rows += [(d, j, r) for j, r in enumerate(rr)]
    si, sn = map(int, shard.split("/"))
    rows = rows[si::sn]
    return rows[::-1] if reverse else rows


def encode_prompt(tok, model_name, dataset, row, maxgen, device):
    """Middle-truncate to MAXLEN - generation length, apply the chat format, return input ids [1, N]."""
    from utils import build_chat_prompt
    p = row["input_prompt"]
    ids = tok(p, truncation=False).input_ids
    cap = MAXLEN - int(maxgen[dataset])
    if len(ids) > cap:
        h = cap // 2
        p = tok.decode(ids[:h], skip_special_tokens=True) + tok.decode(ids[-h:], skip_special_tokens=True)
    return tok(build_chat_prompt(p, model_name, tok, dataset), return_tensors="pt").input_ids.to(device)


def chdir_root():
    os.chdir(ROOT)
