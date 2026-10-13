"""RULER-style single-needle NIAH prompts at controlled length and depth (Section 4.3).

The haystack is cut from long documents: every `<name>.jsonl` file in --haystack_dir whose rows carry a "context"
field longer than 2,000 characters (the paper used the source documents of narrativeqa, gov_report, qmsum,
multi_news, qasper and multifieldqa_en, not LongBench samples). Documents are concatenated and cut to the target
length with the Llama-3.1 tokenizer; the needle sentence goes to the sentence boundary nearest the requested depth,
and the question closes the prompt, so it is the request the observation window sees.

    python experiments/niah/build_niah.py --haystack_dir <dir> --out datasets/niah_single.jsonl
"""
import argparse, glob, json, os, random
from transformers import AutoTokenizer

NEEDLE = "The special magic number for {key} is: {val}."
QUESTION = "What is the special magic number for {key} mentioned in the provided text? Answer with the number only."
INSTR = "Some special magic numbers are hidden within the following text. Make sure to memorize them. I will quiz you about the numbers afterwards.\n\n"
KEYS = ["cheerful-otter", "silent-harbor", "amber-comet", "rapid-willow", "lucky-quartz", "gentle-falcon", "crimson-maple",
        "hollow-river", "brave-lantern", "frozen-orchid", "quiet-meadow", "golden-pebble", "swift-canyon", "misty-cedar"]
DOCS = ["narrativeqa", "gov_report", "qmsum", "multi_news", "qasper", "multifieldqa_en"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--haystack_dir", required=True)
    ap.add_argument("--lens", default="4096,8192,16384,32768"); ap.add_argument("--depths", type=int, default=11)
    ap.add_argument("--per", type=int, default=5); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    rng = random.Random(0)
    files = []
    for d in DOCS:                                     # paper order first, then any other file in the directory
        files += sorted(glob.glob(os.path.join(a.haystack_dir, f"{d}*.jsonl")))
    files += [f for f in sorted(glob.glob(os.path.join(a.haystack_dir, "*.jsonl"))) if f not in files]
    pool = []
    for fn in files:
        for l in open(fn):
            c = json.loads(l).get("context", "")
            if len(c) > 2000:
                pool.append(c)
    rng.shuffle(pool)
    out = open(a.out, "w")
    n = 0
    for L in map(int, a.lens.split(",")):
        for di in range(a.depths):
            depth = di / (a.depths - 1)
            for k in range(a.per):
                key = KEYS[(n + k) % len(KEYS)]; val = str(rng.randint(1_000_000, 9_999_999))
                hay, ids = [], 0                                # haystack of about L - 150 tokens
                i = rng.randrange(len(pool))
                while ids < L - 150:
                    t = pool[i % len(pool)]; hay.append(t); ids += len(tok(t, add_special_tokens=False).input_ids); i += 1
                text = tok.decode(tok("\n\n".join(hay), add_special_tokens=False).input_ids[: L - 150])
                pos = int(len(text) * depth)
                cut = text.rfind(". ", 0, pos) + 2 if depth > 0 else 0
                if depth == 1.0:
                    cut = len(text)
                ctx = text[:cut] + (" " if cut else "") + NEEDLE.format(key=key, val=val) + " " + text[cut:]
                prompt = INSTR + ctx + "\n\n" + QUESTION.format(key=key)
                out.write(json.dumps(dict(id=n, length=L, depth=round(depth, 2), key=key, answer=val, prompt=prompt)) + "\n")
                n += 1
    print("wrote", n)


if __name__ == "__main__":
    main()
