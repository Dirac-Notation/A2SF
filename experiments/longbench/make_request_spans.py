"""Request spans of every LongBench prompt, for the observation-window ablation (longbench.py --window_span).

  full  the whole request: the template text after {context} with the input filled in (instruction + question + answer cue)
  last  the last line of the request, together with the line before it when the last line is a short answer cue
        ("Answer:", "Summary:", "Type:", ...), e.g. the question plus "Answer:"
Rows follow datasets/longbench/<ds>.jsonl order. Output: datasets/longbench_spans/<ds>.jsonl with {idx, full, last}.
Run from the repository root.
"""
import json, os
from datasets import load_dataset

T = json.load(open("config/dataset2prompt.json"))
DS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum", "multi_news",
      "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
os.makedirs("datasets/longbench_spans", exist_ok=True)
for ds in DS:
    hf = load_dataset("THUDM/LongBench", ds, split="test", trust_remote_code=True)
    raw = [json.loads(l) for l in open(f"datasets/longbench/{ds}.jsonl")]
    suf = T[ds].split("{context}")[1]
    bad = 0
    with open(f"datasets/longbench_spans/{ds}.jsonl", "w") as f:
        for i, (h, r) in enumerate(zip(hf, raw)):
            bad += T[ds].format(**h) != r["input_prompt"]
            full = suf.format(**h)
            lines = [x for x in full.split("\n") if x.strip()]
            k = 2 if len(lines) > 1 and len(lines[-1].strip()) < 25 else 1        # keep a short answer cue with its line
            last = full[full.rindex(lines[-k]):]
            f.write(json.dumps(dict(idx=i, full=full, last=last), ensure_ascii=False) + "\n")
    print(f"{ds:22s} rows {len(raw)} mismatched {bad}")
