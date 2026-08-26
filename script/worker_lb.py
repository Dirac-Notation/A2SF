"""Persistent per-GPU worker for sample-level global-queue LongBench runs.

Protocol (stdin/stdout, line-based; launched by script/orchestrator_lb.py):
  startup  -> loads model ONCE, prints "READY_WORKER"
  stdin    <- "<dataset>\t<idx>"   (one sample assignment)
  work     -> replicates longbench.py's per-sample path exactly (truncation,
              chat template, waits_table per-sample action, samsum EOS), then
              appends {"idx", "pred", "answers", ...} to <out_dir>/<dataset>.jsonl
              on THIS server's local disk.
  stdout   -> "ACK <dataset> <idx>"        (flow control: orchestrator sends next)
  stdin    <- "QUIT" or EOF -> exit

Accepts the same compression knobs as longbench.py so any method/table runs
unchanged. Reads samples from datasets/longbench/<dataset>.jsonl by line index
(files must be pre-synced to every server; only a few bytes travel the pipe).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from utils import load_model, set_seed, CompressionConfig, build_chat_prompt, chat_stop_strings  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", default="waits")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--sigmoid_a", type=float, default=None)
    ap.add_argument("--curve", default="sigmoid")
    ap.add_argument("--recent_budget", type=int, default=16)
    ap.add_argument("--n_sink", type=int, default=0)
    ap.add_argument("--ada_kv", action="store_true")
    ap.add_argument("--chunk_size", type=int, default=0)
    ap.add_argument("--chunk_group_size", type=int, default=1)
    ap.add_argument("--pyramid_kv", action="store_true")
    ap.add_argument("--pyramid_ratio", type=float, default=4.0)
    ap.add_argument("--waits_table", default=None)
    ap.add_argument("--key_prior", default=None)
    ap.add_argument("--value_weight", type=float, default=None)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    model_key = args.model.split("_")[0].lower()
    with open("config/model2maxlen.json") as f:
        max_length = int(json.load(f)[model_key])
    with open("config/dataset2maxlen.json") as f:
        dataset2maxlen = json.load(f)

    waits_table = None
    if args.waits_table:
        with open(args.waits_table) as f:
            waits_table = json.load(f)[f"{model_key}_{int(args.budget)}"]

    model, tokenizer = load_model(model_key)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = CompressionConfig()
    config["compression_method"] = args.method
    config["observation_window"] = int(args.window)
    config["total_budget"] = int(args.budget)
    config["a"] = 10 if args.sigmoid_a is None else float(args.sigmoid_a)
    config["b"] = int(args.window)
    config["recent_budget"] = int(args.recent_budget)
    config["n_sink"] = int(args.n_sink)
    config["ada_kv"] = bool(args.ada_kv)
    config["chunk_size"] = int(args.chunk_size)
    config["chunk_group_size"] = int(args.chunk_group_size)
    config["pyramid_kv"] = bool(args.pyramid_kv)
    config["pyramid_ratio"] = float(args.pyramid_ratio)
    config["key_prior"] = args.key_prior
    config["value_weight"] = args.value_weight
    config["curve"] = str(args.curve)

    data_cache = {}

    def get_sample(dataset, idx):
        if dataset not in data_cache:
            with open(f"datasets/longbench/{dataset}.jsonl") as f:
                data_cache[dataset] = f.readlines()
        return json.loads(data_cache[dataset][idx])

    print("READY_WORKER", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line or line == "QUIT":
            break
        dataset, idx = line.split("\t")
        idx = int(idx)
        json_obj = get_sample(dataset, idx)

        prompt = json_obj["input_prompt"]
        tok = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tok) > max_length:
            half = int(max_length / 2)
            prompt = (tokenizer.decode(tok[:half], skip_special_tokens=True)
                      + tokenizer.decode(tok[-half:], skip_special_tokens=True))
        prompt = build_chat_prompt(prompt, model_key, tokenizer, dataset)

        enc = tokenizer(prompt, truncation=False, return_tensors="pt")
        input_ids = enc.input_ids.to(model.device)
        attention_mask = enc.attention_mask.to(torch.bfloat16).to(model.device)
        ctx_len = int(input_ids.shape[-1])
        max_gen = int(dataset2maxlen.get(dataset, 64))

        if waits_table is not None:
            ab = waits_table.get(dataset, [None] * (idx + 1))[idx]
            if ab is not None:
                config["compression_method"] = "waits"
                config["a"] = float(ab[0])
                config["b"] = int(ab[1])

        if args.method == "kvzip":
            from utils_real_drop.kvzip import kvzip_generate
            stop_ids = [tokenizer.eos_token_id]
            if dataset == "samsum":
                stop_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])
            gen_ids = kvzip_generate(
                model, tokenizer, input_ids, max_new_tokens=max_gen,
                budget=int(args.budget), recent_budget=int(args.recent_budget),
                n_sink=int(args.n_sink), stop_token_ids=stop_ids,
            ).to(input_ids.device)
            output = torch.cat([input_ids[0], gen_ids])
        else:
            model.init_cache(config)
            gen_kwargs = dict(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_gen, num_beams=1, do_sample=False,
                pad_token_id=tokenizer.eos_token_id, tokenizer=tokenizer,
                stop_strings=chat_stop_strings(model_key), num_logits_to_keep=1,
            )
            if dataset == "samsum":
                gen_kwargs["min_length"] = ctx_len + 1
                gen_kwargs["eos_token_id"] = [
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ]
            with torch.inference_mode():
                output = model.generate(**gen_kwargs)[0]

        pred = tokenizer.decode(output[ctx_len:], skip_special_tokens=True)
        with open(os.path.join(args.out_dir, f"{dataset}.jsonl"), "a", encoding="utf-8") as f:
            json.dump({"idx": idx, "pred": pred,
                       "answers": json_obj.get("answers", []),
                       "all_classes": json_obj.get("all_classes", []),
                       "length": json_obj.get("length")}, f, ensure_ascii=False)
            f.write("\n")
        print(f"ACK {dataset} {idx}", flush=True)


if __name__ == "__main__":
    main()
