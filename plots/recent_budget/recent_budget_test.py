import matplotlib.pyplot as plt

# --- Experimental data ---
budget_sums = [100, 200, 300, 400]
ratios      = [75/25, 50/50, 25/75]  # [3.0, 1.0, 0.333...]

# ROUGE-1 scores for ff=1.00 (“h2o”) at each (sum, ratio)
rouge1_h2o = {
    ratios[0]: [0.3369, 0.3685, 0.3772, 0.4032],  # ratio=3.0
    ratios[1]: [0.3408, 0.3841, 0.3923, 0.3865],  # ratio=1.0
    ratios[2]: [0.0718, 0.3808, 0.3807, 0.4033],  # ratio≈0.333
}
# ROUGE-1 scores for ff=0.99 (“a2sf”) at each (sum, ratio)
rouge1_a2sf = {
    ratios[0]: [0.3642, 0.3899, 0.3936, 0.4014],
    ratios[1]: [0.3697, 0.3868, 0.3991, 0.4088],
    ratios[2]: [0.3622, 0.3893, 0.4010, 0.3939],
}

markers    = ['o', 's', '^']
linestyles = {'h2o': '-', 'a2sf': '--'}
colors     = {'h2o': 'tab:blue', 'a2sf': 'tab:orange'}

# --- Create subplots: one per ratio ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, ratio, marker in zip(axes, ratios, markers):
    # h2o curve
    ax.plot(
        budget_sums,
        rouge1_h2o[ratio],
        label='h2o (ff=1.00)',
        marker=marker,
        linestyle=linestyles['h2o'],
        color=colors['h2o'],
        linewidth=2,
        markersize=8
    )
    # a2sf curve
    ax.plot(
        budget_sums,
        rouge1_a2sf[ratio],
        label='a2sf (ff=0.99)',
        marker=marker,
        linestyle=linestyles['a2sf'],
        color=colors['a2sf'],
        linewidth=2,
        markersize=8
    )
    # formatting each subplot
    ax.set_title(f'Recent/Select = {ratio:.2f}', fontsize=14, pad=8)
    ax.set_xlabel('Total Budget (select + recent)', fontsize=12)
    ax.set_xticks(budget_sums)
    ax.grid(True, linestyle='--', alpha=0.6)

# only the first subplot gets a y-label
axes[0].set_ylabel('ROUGE-1 Score', fontsize=12)

# overall figure title and legend
fig.suptitle('ROUGE-1 vs Total Budget by Recent/Select Ratio', fontsize=16, y=1.03)
fig.legend(
    handles=axes[0].get_lines()[0:2],
    labels=['h2o (ff=1.00)', 'a2sf (ff=0.99)'],
    loc='upper center',
    ncol=2,
    fontsize=12,
    frameon=False,
    bbox_to_anchor=(0.5, -0.05)
)

plt.tight_layout()
plt.savefig('plots/recent_budget/recent_budget_test.png', dpi=300)
