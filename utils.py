import torch
import json
import os
import numpy as np
import random

from transformers import AutoTokenizer

class CompressionConfig(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

# ── per-model chat formatting ────────────────────────────────────────────────
# Datasets evaluated WITHOUT a chat wrapper (few-shot / completion style); matches
# the original LongBench-WAITS convention and applies to every model.
NO_CHAT_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

_MODEL2CHAT = None

def get_chat_mode(model_name):
    """Per-model chat-format mode from config/model2chat.json:
      'inst'   -> [INST]{p}[/INST]              (Llama-2/Mistral convention; preserves
                                                 existing llama/mistral baselines)
      'native' -> tokenizer.apply_chat_template (the model's own template, e.g. Qwen ChatML
                                                 with its default system prompt + <|im_end|>)
      'raw'    -> no wrapping                    (base / non-instruct models)
    Unlisted models default to 'native' (correct for any instruct tokenizer)."""
    global _MODEL2CHAT
    if _MODEL2CHAT is None:
        try:
            _MODEL2CHAT = json.load(open("config/model2chat.json", "r"))
        except Exception:
            _MODEL2CHAT = {}
    return _MODEL2CHAT.get(model_name, "native")

def build_chat_prompt(prompt, model_name, tokenizer, dataset=None):
    """Wrap a raw prompt in the model's chat format (see get_chat_mode)."""
    if dataset in NO_CHAT_DATASETS:
        return prompt
    mode = get_chat_mode(model_name)
    if mode == "inst":
        return f"[INST]{prompt}[/INST]"
    if mode == "native":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False)
    return prompt

def chat_stop_strings(model_name):
    """Generation stop string for the model's chat format ('inst' -> '[/INST]';
    'native'/'raw' rely on the model's eos token)."""
    return "[/INST]" if get_chat_mode(model_name) == "inst" else None


def load_model(model_name):
    """Load a KV-compression model + tokenizer by shortname (transformers v5).

    Uses the model-agnostic v5 plugin (utils_real_drop.compress): any HF model
    is loaded and given a `model.init_cache(cfg)` method + a `model.generate(...)`
    that auto-injects the compressed cache, so the rest of the pipeline
    (longbench.py / RL / evaluate_needle) is model-agnostic.

    Args:
        model_name (str): shortname key in config/model2path.json (e.g. 'llama3-1b').

    Returns:
        tuple: (model, tokenizer)
    """
    model2path = json.load(open("config/model2path.json", "r"))
    model_path = model2path[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    from utils_real_drop.compress import load_pipeline_model
    model = load_pipeline_model(model_path, dtype=torch.bfloat16, device_map="auto")

    return model, tokenizer


def load_compressed_lm(model_path, dtype=torch.bfloat16, device_map="auto"):
    """Load a KV-compression model by HF *path* (not shortname).

    Like `load_model` but for scripts that already have a resolved model path and
    load their own tokenizer (e.g. build_lb_index.py / longbench_oracle.py /
    benchmark_ttft.py). Returns a model exposing the `model.init_cache(cfg)` +
    `model.generate(...)` interface."""
    from utils_real_drop.compress import load_pipeline_model
    return load_pipeline_model(model_path, dtype=dtype, device_map=device_map)
