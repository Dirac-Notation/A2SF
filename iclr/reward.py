"""Training-corpus row -> 13-action reward vector.

Two reward definitions behind one interface:

  p0    Tanimoto(compressed output, full-cache output). Needs no ground truth, but follows a
        wrong reference wherever the full-cache answer is itself wrong. Measured: on the
        LongBench (M, QA) cell the P0-optimal action is 10:16 for all three models while the
        score-optimal one is 10:1 / 0:1, a 1.86-2.32 gap.
  gold  The task's own metric against the reference answer, using the same scoring functions
        as the evaluation harness. recipe-v5 rows carry `gold` / `metric` fields. LongBench
        itself is still never used for training.

Reusing `longbench_eval`'s scorers keeps the training reward and the eval metric identical;
multiple references are reduced with max, as in evaluation.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import longbench_eval as LE

METRIC = {
    "qa_f1_score": LE.qa_f1_score,
    "rouge_score": LE.rouge_score,
    "classification_score": LE.classification_score,
    "retrieval_score": LE.retrieval_score,
    "code_sim_score": LE.code_sim_score,
    "count_score": LE.count_score,
}


def tanimoto(a, b):
    A = set(re.findall(r"\w+", (a or "").lower()))
    B = set(re.findall(r"\w+", (b or "").lower()))
    return 1.0 if not A and not B else len(A & B) / max(len(A | B), 1)


def resolve_metric(row):
    """Check that the metric matches the reference format, falling back to a safe metric.

    Observed failure: kv_retrieval was built with metric=retrieval_score, but LongBench's
    retrieval_score only works when the reference has the form "Paragraph N" and returns 0
    otherwise. An all-zero reward vector makes the argmax meaningless.
    """
    m = row.get("metric")
    g = row.get("gold") or []
    if m == "retrieval_score" and not any(re.search(r"Paragraph \d+", str(x)) for x in g):
        return "qa_f1_score"
    return m


def has_gold(row):
    return bool(row.get("gold")) and resolve_metric(row) in METRIC


def action_rewards(row, kind="p0"):
    """Reward vector of length 13. Falls back to p0 when kind='gold' but no reference exists."""
    outs = row.get("action_outputs") or []
    if kind == "gold" and has_gold(row):
        fn = METRIC[resolve_metric(row)]
        ac = row.get("all_classes")
        return np.array([max(fn(o or "", g, all_classes=ac) for g in row["gold"]) for o in outs],
                        dtype=np.float64)
    fc = row.get("full_cache_pred") or ""
    return np.array([tanimoto(o or "", fc) for o in outs], dtype=np.float64)


def full_reward(row, kind="p0"):
    """Reward of the full-cache output itself; meaningful only for gold, used to filter
    unreliable references."""
    if kind == "gold" and has_gold(row):
        fn = METRIC[resolve_metric(row)]
        ac = row.get("all_classes")
        return max(fn(row.get("full_cache_pred") or "", g, all_classes=ac) for g in row["gold"])
    return 1.0
