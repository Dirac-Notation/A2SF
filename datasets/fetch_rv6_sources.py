"""Fetch the recipe-v6 sources: public benchmarks of real long documents with human questions.

Chosen after v5's diagnosis. Synthetic assembly (a gold passage plus distractors) could not
reproduce the LongBench (M, QA) optimum under any of four constructions, and the conclusion
that "only qasper/narrativeqa themselves would do" was too broad: a long-context benchmark
that is not LongBench works just as well and keeps evaluation clean.

  LooGLE   human-written QA over post-2022 arXiv papers, movie scripts and Wikipedia articles;
           shortdep_qa (free-form short answers), longdep_qa (multiple choice), summarization.
           qasper draws on pre-2022 arXiv NLP papers via S2ORC and narrativeqa on Gutenberg
           books and scripts, so documents and questions do not overlap, though the genres
           partly do.
  QuALITY  human-written multiple choice over Gutenberg short stories and magazine articles.

  python datasets/fetch_rv6_sources.py
"""
import io, json, os, urllib.request as u
import pyarrow.parquet as pq

OUT = os.environ.get("ICLR_TRACES", "/data2/smp9898/iclr_traces") + "/rv6/src"
SRC = {
    "loogle_shortqa": ("bigai-nlco/LooGLE", "shortdep_qa"),
    "loogle_longqa": ("bigai-nlco/LooGLE", "longdep_qa"),
    "loogle_summ": ("bigai-nlco/LooGLE", "summarization"),
    "quality_mc": ("emozilla/quality", "default"),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    card = {}
    for key, (repo, cfg) in SRC.items():
        p = f"{OUT}/{key}.jsonl"
        meta = json.load(u.urlopen(f"https://huggingface.co/api/datasets/{repo}/parquet", timeout=120))
        split = "train" if "train" in meta[cfg] else list(meta[cfg])[0]
        n = 0
        if not os.path.exists(p):
            with open(p, "w") as f:
                for url in meta[cfg][split][:2]:
                    t = pq.read_table(io.BytesIO(u.urlopen(url, timeout=1800).read()))
                    for row in t.to_pylist():
                        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n"); n += 1
        else:
            n = sum(1 for _ in open(p))
        card[key] = {"repo": repo, "config": cfg, "split": split, "rows": n}
        print(f"OK {key:16s} {n:6d} rows  <- {repo}[{cfg}/{split}]", flush=True)
    json.dump(card, open(f"{OUT}/datacard_sources.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
