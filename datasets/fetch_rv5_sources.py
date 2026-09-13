"""Fetch the recipe-v5 source corpora from the HuggingFace parquet endpoints and cache them.

Only public corpora disjoint from the 16 LongBench datasets are used (narrativeqa, qasper,
multifieldqa, hotpotqa, 2wikimqa, musique, gov_report, qmsum, multi_news, trec, triviaqa,
samsum, lcc, repobench-p, passage_*). LongBench stays a pure evaluation benchmark.

SOURCES records the origin, license, split and fields used for each corpus, so it transfers
directly into the paper's data card. Only training splits are read; test splits are avoided
because other work evaluates on them.

  python datasets/fetch_rv5_sources.py --out $ICLR_TRACES/rv5/src
"""
import argparse
import io
import json
import os
import urllib.request as urlreq

SOURCES = {
    # key: (repo, config, split, fields used, license, one-line description)
    "squad_v2": ("rajpurkar/squad_v2", "squad_v2", "train",
                 ["context", "question", "answers", "title"], "CC-BY-SA-4.0",
                 "human-written extractive QA over Wikipedia paragraphs, plus unanswerable questions"),
    "drop": ("ucinlp/drop", "default", "train",
             ["passage", "question", "answers_spans"], "CC-BY-SA-4.0",
             "discrete reasoning QA over paragraphs (numeric and span answers)"),
    "race": ("ehovy/race", "high", "train",
             ["article", "question", "options", "answer"], "Apache-2.0 (research use)",
             "four-way multiple-choice reading over English exam passages"),
    "billsum": ("FiscalNote/billsum", "default", "train",
                ["text", "summary"], "CC0-1.0",
                "US federal and state bill texts with summaries"),
    "cnn_dailymail": ("abisee/cnn_dailymail", "3.0.0", "train",
                      ["article", "highlights"], "Apache-2.0",
                      "news articles with highlight summaries"),
    "ag_news": ("fancyzhx/ag_news", "default", "train",
                ["text", "label"], "CC-BY-SA-3.0",
                "four-class news topic classification"),
    "banking77": ("legacy-datasets/banking77", "default", "train",
                  ["text", "label"], "CC-BY-4.0",
                  "77-class intent classification of banking queries"),
}


def fetch(repo, config, split, out_path, max_rows, max_files=2):
    if os.path.exists(out_path):
        return sum(1 for _ in open(out_path))
    import pyarrow.parquet as pq
    meta = json.load(urlreq.urlopen(
        f"https://huggingface.co/api/datasets/{repo}/parquet", timeout=120))
    urls = meta[config][split][:max_files]
    n = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for u in urls:
            t = pq.read_table(io.BytesIO(urlreq.urlopen(u, timeout=900).read()))
            for row in t.to_pylist():
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                n += 1
                if n >= max_rows:
                    return n
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=TRACES + "/rv5/src")
    ap.add_argument("--max_rows", type=int, default=60000)
    a = ap.parse_args()
    card = {}
    for key, (repo, cfg, split, fields, lic, desc) in SOURCES.items():
        p = os.path.join(a.out, f"{key}.jsonl")
        try:
            n = fetch(repo, cfg, split, p, a.max_rows)
            card[key] = {"repo": repo, "config": cfg, "split": split, "fields": fields,
                         "license": lic, "desc": desc, "rows_cached": n}
            print(f"OK   {key:14s} {n:7d} rows  <- {repo} [{cfg}/{split}]", flush=True)
        except Exception as e:
            print(f"FAIL {key:14s} {type(e).__name__}: {str(e)[:100]}", flush=True)
    # record sources that were already cached
    card["pubmed"] = {"repo": "ccdv/pubmed-summarization", "config": "section",
                      "split": "test", "fields": ["article", "abstract"],
                      "license": "public summarization corpus derived from paper texts",
                      "desc": "biomedical paper bodies with abstracts", "rows_cached": 600}
    card["wikitext"] = {"repo": "wikitext", "config": "wikitext-103-raw-v1", "split": "train",
                        "fields": ["text"], "license": "CC-BY-SA-3.0",
                        "desc": "raw Wikipedia documents, used as distractors and for synthetic retrieval",
                        "rows_cached": -1}
    json.dump(card, open(os.path.join(a.out, "datacard_sources.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\ndata card -> {os.path.join(a.out, 'datacard_sources.json')}")


if __name__ == "__main__":
    main()
