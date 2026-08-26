"""Observable metadata vocab for the (task, metric) routing state.

The routing policy's Input is one-hot over [task_type | metric_type]. This module owns the
canonical orderings + index helpers (derived from config/task2dataset.json), plus the default
LongBench dataset -> (task, metric) mapping used to apply the learned routing table at eval.
"""
import json
import os
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TASK2DATASET_PATH = os.path.join(REPO_ROOT, "config", "task2dataset.json")

with open(TASK2DATASET_PATH, "r", encoding="utf-8") as f:
    TASK_TO_DATASETS = json.load(f)

TASK_TYPE_ORDER = list(TASK_TO_DATASETS.keys()) + ["unknown"]
TASK_TYPE_TO_INDEX = {name: idx for idx, name in enumerate(TASK_TYPE_ORDER)}

# LongBench dataset name -> task type (derived from config/task2dataset.json)
DATASET_TO_TASK_TYPE = {
    dataset_name.lower(): task_name
    for task_name, datasets in TASK_TO_DATASETS.items()
    for dataset_name in datasets
}


def normalize_task_type(task_type: Optional[str]) -> str:
    if task_type is None:
        return "unknown"
    key = str(task_type).strip()
    return key if key in TASK_TYPE_TO_INDEX else "unknown"


def resolve_task_type(dataset: Optional[str] = None, task_type: Optional[str] = None) -> str:
    normalized = normalize_task_type(task_type)
    if normalized != "unknown":
        return normalized
    if dataset is None:
        return "unknown"
    return DATASET_TO_TASK_TYPE.get(str(dataset).strip().lower(), "unknown")


def task_type_to_index(task_type: Optional[str] = None, dataset: Optional[str] = None) -> int:
    resolved = resolve_task_type(dataset=dataset, task_type=task_type)
    return int(TASK_TYPE_TO_INDEX.get(resolved, TASK_TYPE_TO_INDEX["unknown"]))


# Metric type ordering for one-hot encoding.
METRIC_TYPE_ORDER = [
    "qa_f1_score",
    "qa_f1_zh_score",
    "rouge_score",
    "rouge_zh_score",
    "classification_score",
    "retrieval_score",
    "retrieval_zh_score",
    "count_score",
    "code_sim_score",
    "unknown",
]
METRIC_TYPE_TO_INDEX = {name: idx for idx, name in enumerate(METRIC_TYPE_ORDER)}


def metric_type_to_index(metric_type: Optional[str]) -> int:
    if metric_type is None:
        return int(METRIC_TYPE_TO_INDEX["unknown"])
    key = str(metric_type).strip()
    return int(METRIC_TYPE_TO_INDEX.get(key, METRIC_TYPE_TO_INDEX["unknown"]))


# Default metric per (task / few-shot dataset), used to apply the routing table at eval time.
TASK_DEFAULT_METRIC = {
    "Single-doc QA": "qa_f1_score", "Multi-doc QA": "qa_f1_score",
    "Passage Retrieval": "qa_f1_score", "Code Complete": "code_sim_score",
    "Summarization": "rouge_score",
}
FEWSHOT_DATASET_METRIC = {
    "trec": "classification_score", "triviaqa": "qa_f1_score", "samsum": "rouge_score",
}


def dataset_metric(dataset: str, task_type: str) -> str:
    """Metric a LongBench dataset is scored with (few-shot dataset overrides task default)."""
    return FEWSHOT_DATASET_METRIC.get(dataset, TASK_DEFAULT_METRIC.get(task_type, "qa_f1_score"))
