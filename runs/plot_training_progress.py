import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import rcParams

# ---------------------------------------------------------
# 1. Global Style Settings (Publication-ready)
# ---------------------------------------------------------
rcParams.update({
    "font.family": "serif",
    "figure.figsize": (14, 10),
    "figure.dpi": 150,
    "font.size": 22,
    "axes.labelsize": 26,
    "axes.titlesize": 28,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.5,
})

# ---------------------------------------------------------
# 2. Data Loading
# ---------------------------------------------------------
folder_name = "runs/a2sf_rl_all"
data_file = f"{folder_name}/training_progress.jsonl"

iterations = []
avg_rewards = []
total_losses = []

with open(data_file, 'r') as f:
    for line in f:
        if line.strip():  # Skip empty lines
            data = json.loads(line)
            iterations.append(data['iteration'])
            avg_rewards.append(data['avg_reward'])
            total_losses.append(data['total_loss'])

iterations = np.array(iterations)
avg_rewards = np.array(avg_rewards)
total_losses = np.array(total_losses)

# ---------------------------------------------------------
# 2.5. Data Smoothing (5-point moving average)
# ---------------------------------------------------------
window_size = 10

def smooth_data(data, window_size):
    """5개씩 묶어서 평균 계산"""
    smoothed = []
    smoothed_indices = []
    
    for i in range(0, len(data), window_size):
        chunk = data[i:i+window_size]
        smoothed.append(np.mean(chunk))
        # iteration은 묶음의 중간값 사용
        smoothed_indices.append(iterations[i + len(chunk) // 2] if len(chunk) > 0 else iterations[i])
    
    return np.array(smoothed_indices), np.array(smoothed)

iterations_smooth, avg_rewards_smooth = smooth_data(avg_rewards, window_size)
_, total_losses_smooth = smooth_data(total_losses, window_size)

# ---------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, figsize=(14, 10))

# Subplot 1: Average Reward
ax1.plot(
    iterations_smooth,
    avg_rewards_smooth,
    marker='o',
    linewidth=4,
    markersize=8,
    color="#4C72B0",  # Muted Blue
    label=f"Average Reward ({window_size}-point average)",
    zorder=5
)

ax1.set_title("Average Reward Over Training", pad=20)
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Average Reward")
ax1.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
ax1.legend(frameon=False, loc='best')
ax1.margins(x=0.02)

# Subplot 2: Total Loss
ax2.plot(
    iterations_smooth,
    total_losses_smooth,
    marker='o',
    linewidth=4,
    markersize=8,
    color="#C44E52",  # Muted Red
    label=f"Total Loss ({window_size}-point average)",
    zorder=5
)

ax2.set_title("Total Loss Over Training", pad=20)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Total Loss")
ax2.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
ax2.legend(frameon=False, loc='best')
ax2.margins(x=0.02)

plt.savefig(f"{folder_name}/training_progress.png", dpi=150, bbox_inches='tight')
plt.show()

