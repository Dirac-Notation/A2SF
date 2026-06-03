"""Generate harder needle-in-haystack dataset using WikiText filler.

Difficulty knobs (vs original):
  1. Distractor needles — 4 fake "password" sentences (old/default/temporary/backup)
     scattered in the haystack. Only ONE labelled "current" is the real answer.
  2. Alphanumeric 8-char passwords (mixed case + digits) instead of 5-digit numbers.
     Less likely to be confused with random WikiText numbers (years etc).
  3. System prompt + question both reference "current password" specifically.

Usage:
  python datasets/make_needle_dataset.py [--no_distractors] ...
"""
import argparse
import json
import os
import random
import string

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "[INST]You are a helpful assistant. The following <text> contains several sentences "
    "describing different passwords (old, default, temporary, backup, current). "
    "Find the CURRENT password and report it.\n\n<text>\n"
)
USER_PROMPT_SUFFIX = (
    "\n</text>\n\nWhat is the CURRENT password? "
    "Respond exactly in the format: \"The current password is (insert answer here).\"[/INST]"
)
ANSWER_PREFIX = "The current password is"

REAL_LABEL = "current"
DISTRACTOR_LABELS = ["old", "default", "temporary", "backup"]

NEEDLE_TEMPLATE = "The {label} password is {pw}."
PW_ALPHABET = string.ascii_letters + string.digits   # 62 chars


def gen_password(length=8):
    return "".join(random.choice(PW_ALPHABET) for _ in range(length))


def _is_clean_sentence(s: str) -> bool:
    s = s.strip()
    if len(s) < 20: return False
    if s.startswith("=") or s.startswith("@"): return False
    if "password" in s.lower(): return False
    return True


def _digit_density(s: str) -> float:
    """Fraction of characters that are digits (used for numeric-heavy filtering)."""
    if not s: return 0.0
    return sum(c.isdigit() for c in s) / len(s)


def collect_haystack_sentences(tokenizer, target_per_bucket=2000, min_tokens=8, max_tokens=20,
                               numeric_heavy=False, min_digit_density=0.06):
    print(f"Loading wikitext (streaming)...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    buckets = {n: [] for n in range(min_tokens, max_tokens + 1)}
    pbar = tqdm(desc="collecting sentences")
    for example in ds:
        text = example.get("text", "").strip()
        if not text: continue
        parts = [p.strip() for p in text.split(".") if p.strip()]
        for p in parts:
            sent = p + "."
            if not _is_clean_sentence(sent): continue
            if numeric_heavy and _digit_density(sent) < min_digit_density: continue
            tl = len(tokenizer.encode(sent, add_special_tokens=False))
            if min_tokens <= tl <= max_tokens and len(buckets[tl]) < target_per_bucket:
                buckets[tl].append(sent)
                pbar.update(1)
        if all(len(v) >= target_per_bucket for v in buckets.values()):
            break
    pbar.close()
    counts = ", ".join(f"{n}:{len(buckets[n])}" for n in sorted(buckets))
    print(f"sentence buckets: {counts}")
    flat = [s for v in buckets.values() for s in v]
    random.shuffle(flat)
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="llama3-8b")
    ap.add_argument("--output", type=str, default="datasets/needle/needle_eval.jsonl")
    ap.add_argument("--target_lengths", type=str,
                    default="1024,2048,4096,8192,16384,32768")
    ap.add_argument("--positions", type=str,
                    default="0.0,0.1,0.25,0.5,0.75,0.9,1.0")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_distractors", action="store_true",
                    help="Disable distractor needles (use only the real 'current' password).")
    ap.add_argument("--pw_length", type=int, default=8,
                    help="Alphanumeric password length (default 8).")
    ap.add_argument("--numeric_heavy", action="store_true",
                    help="Filter haystack sentences for high digit density (numeric-heavy noise).")
    ap.add_argument("--min_digit_density", type=float, default=0.06,
                    help="Minimum digit density when --numeric_heavy is set (default 0.06).")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    model2path = json.load(open("config/model2path.json"))
    tokenizer = AutoTokenizer.from_pretrained(model2path[args.model])
    sentences = collect_haystack_sentences(
        tokenizer,
        numeric_heavy=args.numeric_heavy,
        min_digit_density=args.min_digit_density,
    )
    if not sentences:
        raise RuntimeError("Failed to collect any haystack sentences")

    target_lengths = [int(x) for x in args.target_lengths.split(",") if x.strip()]
    positions = [float(x) for x in args.positions.split(",") if x.strip()]

    # Compute prompt-overhead estimate
    sample_real_needle = NEEDLE_TEMPLATE.format(label=REAL_LABEL, pw="X" * args.pw_length)
    overhead = (
        len(tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=True))
        + len(tokenizer.encode(USER_PROMPT_SUFFIX, add_special_tokens=False))
        + len(tokenizer.encode(sample_real_needle, add_special_tokens=False))
    )
    if not args.no_distractors:
        # Distractor needles also take tokens
        for lab in DISTRACTOR_LABELS:
            d = NEEDLE_TEMPLATE.format(label=lab, pw="X" * args.pw_length)
            overhead += len(tokenizer.encode(d, add_special_tokens=False))
    print(f"prompt overhead ≈ {overhead} tokens (distractors={'OFF' if args.no_distractors else 'ON'})")

    samples = []
    for total in tqdm(target_lengths, desc="lengths"):
        haystack_target = max(0, total - overhead)
        for pos in positions:
            for rep in range(args.repeats):
                real_pw = gen_password(args.pw_length)
                real_needle = NEEDLE_TEMPLATE.format(label=REAL_LABEL, pw=real_pw)

                distractor_needles = []
                if not args.no_distractors:
                    for lab in DISTRACTOR_LABELS:
                        dpw = gen_password(args.pw_length)
                        # Avoid accidental collision with real_pw substring
                        while dpw == real_pw:
                            dpw = gen_password(args.pw_length)
                        distractor_needles.append(NEEDLE_TEMPLATE.format(label=lab, pw=dpw))

                # Build haystack to ~haystack_target tokens
                tokens_so_far = 0
                picked = []
                while tokens_so_far < haystack_target:
                    s = random.choice(sentences)
                    picked.append(s)
                    tokens_so_far += len(tokenizer.encode(s, add_special_tokens=False)) + 1
                    if len(picked) > 100000: break

                # Insert real needle at requested position
                n = len(picked)
                ins_real = max(0, min(n, int(round(n * pos))))
                # Insert distractors at random other positions (avoid colliding with real)
                if distractor_needles:
                    others = list(range(n + 1))
                    if ins_real in others:
                        others.remove(ins_real)
                    random.shuffle(others)
                    distractor_positions = sorted(others[:len(distractor_needles)])
                else:
                    distractor_positions = []

                # Build the final sequence: track inserts in order so positions stay coherent
                inserts = sorted([(ins_real, real_needle, "real")] +
                                 [(p_, d, "distractor") for p_, d in zip(distractor_positions, distractor_needles)],
                                 key=lambda x: x[0])

                out_parts = []
                offset = 0
                for ins_pos, needle_text, kind in inserts:
                    out_parts.extend(picked[offset:ins_pos])
                    out_parts.append(needle_text)
                    offset = ins_pos
                out_parts.extend(picked[offset:])
                haystack_full = " ".join(out_parts)

                prompt = SYSTEM_PROMPT + haystack_full + USER_PROMPT_SUFFIX
                actual_tokens = len(tokenizer.encode(prompt))

                # Compute the REAL needle's char position in final haystack (for analysis)
                # Find position by reconstructing partial join
                pre_real = " ".join(picked[:ins_real]) + (" " if ins_real > 0 else "")
                # add any distractor positioned before real
                pre_real_extra = ""
                for ip, nt, k in inserts:
                    if ip < ins_real and k == "distractor":
                        # this distractor was inserted before real
                        # it's already in pre_real_extra implicitly if we went through inserts
                        pass
                needle_char = haystack_full.find(real_needle)

                samples.append({
                    "prompt": prompt,
                    "password": real_pw,
                    "needle": real_needle,
                    "distractor_needles": distractor_needles,
                    "question": "What is the CURRENT password?",
                    "answer_prefix": ANSWER_PREFIX,
                    "target_length": str(total),
                    "actual_tokens": str(actual_tokens),
                    "position_pct": str(pos),
                    "needle_char_position": str(needle_char),
                    "haystack_char_length": str(len(haystack_full)),
                    "repeat_idx": str(rep),
                    "n_distractors": str(len(distractor_needles)),
                    "pw_length": str(args.pw_length),
                })

    with open(args.output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"saved {len(samples)} samples → {args.output}")
    print(f"  difficulty: distractors={'OFF' if args.no_distractors else f'4 ({DISTRACTOR_LABELS})'}, "
          f"pw_len={args.pw_length} alphanum, "
          f"haystack={'numeric-heavy' if args.numeric_heavy else 'wikitext'}")


if __name__ == "__main__":
    main()
