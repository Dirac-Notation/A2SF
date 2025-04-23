import ast
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse

parser = argparse.ArgumentParser(description="Generate heatmap from a txt file.")
parser.add_argument("--file", type=str, required=True, help="Path to the input txt file")
args = parser.parse_args()

txt_filename = args.file
base_name = os.path.splitext(os.path.basename(txt_filename))[0]
png_filename = base_name + ".png"

with open(txt_filename, "r") as file:
    f = file.readlines()

tensor_group = torch.tensor([ast.literal_eval(s) for s in f]).to(torch.float32).view(-1,32,32)/10

heatmap_data = tensor_group.mean(dim=0).numpy()
row_means = heatmap_data.mean(axis=1)
nrows, ncols = heatmap_data.shape

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.05, 0.05])

ax_heatmap = fig.add_subplot(gs[0])
ax_text = fig.add_subplot(gs[1], sharey=ax_heatmap)
ax_cbar = fig.add_subplot(gs[2])

im = ax_heatmap.imshow(heatmap_data, cmap="viridis", aspect="auto", vmin=0, vmax=0.5)
ax_heatmap.set_xlabel("HEAD NUM")
ax_heatmap.set_ylabel("LAYER NUM")
ax_heatmap.set_xlim(-0.5, ncols - 0.5)

ax_text.set_xlim(0, 1)
ax_text.set_xticks([])
ax_text.set_yticks([])
ax_text.set_ylim(ax_heatmap.get_ylim())

for spine in ax_text.spines.values():
    spine.set_visible(False)

for row in range(nrows):
    ax_text.text(
        0.0, row, f"{row_means[row]:.2f}",
        va="center", ha="left", fontsize=10, color="black"
    )

fig.colorbar(im, cax=ax_cbar, orientation="vertical")

plt.tight_layout()
plt.savefig(png_filename, dpi=300)
plt.show()
