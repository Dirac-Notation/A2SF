"""Shared helpers for the experiment scripts: model loading, chat formatting, seeding."""
import json
import random

import numpy as np
import torch
from transformers import AutoTokenizer

from kvcompress.compress import CompressionConfig  # noqa: F401  (re-exported for the scripts)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


# Datasets evaluated WITHOUT a chat wrapper (few-shot / completion style), for every model.
NO_CHAT_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
_MODEL2CHAT = None


def get_chat_mode(model_name):
    """Chat format per model (config/model2chat.json):
      'inst'   -> [INST]{p}[/INST]               (Llama / Mistral)
      'native' -> tokenizer.apply_chat_template (e.g. Qwen ChatML); also the default for unlisted models
      'raw'    -> no wrapping"""
    global _MODEL2CHAT
    if _MODEL2CHAT is None:
        _MODEL2CHAT = json.load(open("config/model2chat.json", "r"))
    return _MODEL2CHAT.get(model_name, "native")


def build_chat_prompt(prompt, model_name, tokenizer, dataset=None):
    if dataset in NO_CHAT_DATASETS:
        return prompt
    mode = get_chat_mode(model_name)
    if mode == "inst":
        return f"[INST]{prompt}[/INST]"
    if mode == "native":
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             add_generation_prompt=True, tokenize=False)
    return prompt


def chat_stop_strings(model_name):
    """'inst' models stop at '[/INST]'; the others rely on their eos token."""
    return "[/INST]" if get_chat_mode(model_name) == "inst" else None


def load_model(model_name):
    """Load (model, tokenizer) by shortname from config/model2path.json, with the compression pipeline API:
    `model.init_cache(CompressionConfig(...))` then `model.generate(...)`."""
    model_path = json.load(open("config/model2path.json", "r"))[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    from kvcompress.compress import load_pipeline_model
    model = load_pipeline_model(model_path, dtype=torch.bfloat16, device_map="auto")
    return model, tokenizer
