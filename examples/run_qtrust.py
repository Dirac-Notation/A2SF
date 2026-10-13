"""Generate with a Q-Trust-compressed KV cache and compare against the full cache.

    python examples/run_qtrust.py --model meta-llama/Llama-3.1-8B-Instruct --budget 128

The prompt is a synthetic needle-in-a-haystack: a 7-digit number hidden in filler text, asked about at the end.
"""
import argparse, os, random, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qtrust.hf import enable_qtrust, QTrustCache


def needle_prompt(n_sentences=1500, depth=0.4, seed=0):
    rng = random.Random(seed)
    filler = ["The grass is green.", "The sky is blue.", "The sun is yellow.", "Here we go.", "There and back again."]
    body = [rng.choice(filler) for _ in range(n_sentences)]
    number = str(rng.randint(1_000_000, 9_999_999))
    body.insert(int(depth * n_sentences), f"The special magic number is: {number}.")
    question = "What is the special magic number mentioned in the text? Answer with the number only."
    return " ".join(body) + "\n\n" + question, number


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--no_kernel", action="store_true", help="use the plain PyTorch reference instead of Triton")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    text, answer = needle_prompt()
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"].to(model.device)
    gen = dict(max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
    print(f"prompt: {ids.shape[1]} tokens, answer {answer}")

    with torch.inference_mode():
        full = model.generate(ids, **gen)[0, ids.shape[1]:]
        enable_qtrust(model)
        cache = QTrustCache(model.config, budget=a.budget, window=16, recent=16, tau=a.tau, use_kernel=not a.no_kernel)
        comp = model.generate(ids, past_key_values=cache, **gen)[0, ids.shape[1]:]

    kept = cache.layers[0].keys.shape[2]
    print(f"full cache        : {tok.decode(full, skip_special_tokens=True)!r}")
    print(f"Q-Trust (KV={a.budget}) : {tok.decode(comp, skip_special_tokens=True)!r}   (cached length after decoding: {kept})")


if __name__ == "__main__":
    main()
