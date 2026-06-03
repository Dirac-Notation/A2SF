"""Per-(layer, head) attention-map and weighting visualisations for the
Figure 1 (method overview) panels. Each combination is saved as a folder
of small PNGs that can be composed by hand into the final figure.

Outputs (per (layer, head) under `plots/layer_X/head_Y/`):
  full.png                     full attention map (no weighting)
  tova.png, _vector, _weight   TOVA   (last-query-only)
  snapkv.png, _vector, _weight SnapKV (fixed window of last 4 queries)
  h2o.png, _vector, _weight    H2O    (uniform over all queries)
  waits.png, _vector, _weight  WAITS  (sigmoid soft window: w_q = 1/(1+exp(a*(d_q-b))))
"""
import os
import shutil

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
from transformers import AutoTokenizer, AutoModelForCausalLM


rcParams.update({
    "font.family": "serif",
    "font.size":      22,
    "axes.labelsize": 24,
    "axes.titlesize": 26,
    "axes.linewidth":  1.4,
})


try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TEXT     = "I want to go to the park with my dog and play ball."
SNAPKV_W = 4              # SnapKV observation-window length
SIG_A, SIG_B = 0.5, 4.0   # WAITS sigmoid (a, b) — visibly S-shaped on this prompt

BASE_PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
if os.path.exists(BASE_PLOT_DIR):
    shutil.rmtree(BASE_PLOT_DIR)
os.makedirs(BASE_PLOT_DIR, exist_ok=True)


def setup_folders(layer_idx, head_idx):
    layer_dir = os.path.join(BASE_PLOT_DIR, f"layer_{layer_idx}")
    os.makedirs(layer_dir, exist_ok=True)
    head_dir = os.path.join(layer_dir, f"head_{head_idx}")
    os.makedirs(head_dir, exist_ok=True)
    return head_dir


def save_path(dir_path, filename):
    return os.path.join(dir_path, filename)


# ─────────────────────────────────────────────────────────────────
# Load model and attentions
# ─────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_ID}…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, output_attentions=True).eval()
print("Model loaded.")

inputs = tokenizer(TEXT, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

attentions = outputs.attentions
num_layers = len(attentions)
num_heads  = attentions[0].shape[1]
seq_len    = attentions[0].shape[2]


# ─────────────────────────────────────────────────────────────────
# Image-saving helpers
# ─────────────────────────────────────────────────────────────────
def save_attention_map(attn_map, dir_path, filename, box_start=None, box_height=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    vmax = attn_map.max() if attn_map.max() > 0 else 1.0
    ax.imshow(attn_map, cmap="Blues", vmin=0, vmax=vmax)
    if box_start is not None and box_height is not None:
        rect = patches.Rectangle(
            (-0.5, box_start - 0.5), seq_len, box_height,
            linewidth=2, edgecolor="green", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel("Query", fontsize=24, labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Key", fontsize=24, labelpad=8)
    plt.savefig(save_path(dir_path, filename), bbox_inches="tight")
    plt.close()


def save_vector(vec, dir_path, filename, cmap="Blues", vmin=None, vmax=None):
    is_horizontal = vec.shape[0] == 1
    figsize = (6, 0.5) if is_horizontal else (0.5, 6)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(vec, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(save_path(dir_path, filename), bbox_inches="tight")
    plt.close()


def sigmoid(d, a, b):
    return 1.0 / (1.0 + np.exp(a * (d - b)))


# ─────────────────────────────────────────────────────────────────
# Generate per-(layer, head) images
# ─────────────────────────────────────────────────────────────────
print(f"Generating plots for {num_layers} layers × {num_heads} heads "
      f"({num_layers * num_heads} total)…")

for l_idx in range(num_layers):
    for h_idx in range(num_heads):
        head_dir = setup_folders(l_idx, h_idx)
        attention_map = attentions[l_idx][0, h_idx].numpy()
        N = attention_map.shape[0]

        # Full reference
        save_attention_map(attention_map, head_dir, "full.png")

        # ── TOVA: only the last query ──
        tova_attn = np.zeros_like(attention_map)
        tova_attn[-1, :] = attention_map[-1, :]
        save_attention_map(tova_attn, head_dir, "tova.png",
                            box_start=N - 1, box_height=1)
        save_vector(tova_attn.sum(axis=0, keepdims=True),
                     head_dir, "tova_vector.png", cmap="Blues")
        tova_weight = np.zeros((N, 1)); tova_weight[-1, 0] = 1.0
        save_vector(tova_weight, head_dir, "tova_weight.png",
                     cmap="Reds", vmin=0, vmax=1)

        # ── SnapKV: last SNAPKV_W queries ──
        window = min(SNAPKV_W, N)
        snap_attn = np.zeros_like(attention_map)
        snap_attn[-window:, :] = attention_map[-window:, :]
        save_attention_map(snap_attn, head_dir, "snapkv.png",
                            box_start=N - window, box_height=window)
        save_vector(snap_attn.sum(axis=0, keepdims=True),
                     head_dir, "snapkv_vector.png", cmap="Blues")
        snap_weight = np.zeros((N, 1)); snap_weight[-window:, 0] = 1.0
        save_vector(snap_weight, head_dir, "snapkv_weight.png",
                     cmap="Reds", vmin=0, vmax=1)

        # ── H2O: all queries ──
        save_attention_map(attention_map.copy(), head_dir, "h2o.png",
                            box_start=0, box_height=N)
        save_vector(attention_map.sum(axis=0, keepdims=True),
                     head_dir, "h2o_vector.png", cmap="Blues")
        save_vector(np.ones((N, 1)), head_dir, "h2o_weight.png",
                     cmap="Reds", vmin=0, vmax=1)

        # ── WAITS: sigmoid soft window w_q = 1/(1+exp(a(d_q-b))) ──
        waits_attn = np.zeros_like(attention_map)
        waits_weight = np.zeros((N, 1))
        for i in range(N):
            d = (N - 1) - i        # distance from last query
            w = sigmoid(d, SIG_A, SIG_B)
            waits_attn[i, :] = attention_map[i, :] * w
            waits_weight[i, 0] = w
        # WAITS is a soft window — no green dashed box
        save_attention_map(waits_attn, head_dir, "waits.png")
        save_vector(waits_attn.sum(axis=0, keepdims=True),
                     head_dir, "waits_vector.png", cmap="Blues")
        save_vector(waits_weight, head_dir, "waits_weight.png",
                     cmap="Reds", vmin=0, vmax=1)

print(f"All images saved under '{BASE_PLOT_DIR}/layer_*/head_*/'.")
print(f"WAITS sigmoid: w(d) = 1 / (1 + exp({SIG_A} * (d - {SIG_B})))")
