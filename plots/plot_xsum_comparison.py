import matplotlib.pyplot as plt
import numpy as np

# Data from experiments
# Format: [select, recent, random, streaming, ff, rm, rouge1, rouge2, rougeL, throughput]
a2sf_data = [
    [25, 25, 0, 0, 0.99, 'att', 0.3118, 0.1074, 0.2421, 29.63],
    [50, 50, 0, 0, 0.99, 'att', 0.3003, 0.0967, 0.2265, 29.56],
    [75, 75, 0, 0, 0.99, 'att', 0.2866, 0.0854, 0.2197, 29.37],
    [100, 100, 0, 0, 0.99, 'att', 0.2886, 0.0849, 0.2186, 29.15],
    [125, 125, 0, 0, 0.99, 'att', 0.2974, 0.0869, 0.2256, 28.80],
    [150, 150, 0, 0, 0.99, 'att', 0.2977, 0.0906, 0.2270, 28.07]
]

h2o_data = [
    [25, 25, 0, 0, 1.0, 'att', 0.1549, 0.0431, 0.1137, 29.37],
    [50, 50, 0, 0, 1.0, 'att', 0.3033, 0.0982, 0.2291, 29.67],
    [75, 75, 0, 0, 1.0, 'att', 0.3143, 0.1050, 0.2386, 29.66],
    [100, 100, 0, 0, 1.0, 'att', 0.3184, 0.0990, 0.2400, 29.40],
    [125, 125, 0, 0, 1.0, 'att', 0.3192, 0.1045, 0.2404, 29.12],
    [150, 150, 0, 0, 1.0, 'att', 0.3061, 0.0988, 0.2307, 28.25]
]

# Full Llama-2 model results
llama2_rouge1 = 0.3012

# Calculate total budget (select + recent)
a2sf_budget = [d[0] + d[1] for d in a2sf_data]
h2o_budget = [d[0] + d[1] for d in h2o_data]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot A2SF results
plt.plot(a2sf_budget, [d[6] for d in a2sf_data], 'o-', label='A2SF (ff=0.99, rm=att)', color='blue', linewidth=2, markersize=8)

# Plot H2O results
plt.plot(h2o_budget, [d[6] for d in h2o_data], 'o-', label='H2O', color='red', linewidth=2, markersize=8)

# Plot full Llama-2 model results as horizontal line
max_budget = max(max(a2sf_budget), max(h2o_budget))
plt.axhline(y=llama2_rouge1, color='black', linestyle='--', label='Llama-2', linewidth=2)

# Customize the plot
plt.xlabel('Total Budget (select + recent)', fontsize=12)
plt.ylabel('ROUGE-1 Score', fontsize=12)
plt.title('XSUM: ROUGE-1 Score vs Total Budget', fontsize=14)
plt.grid(True, alpha=0.3)

# Set y-axis limits with some padding
plt.ylim(0.15, 0.35)

# Add legend
plt.legend(loc='lower right', fontsize=10)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('plots/xsum_comparison_rouge1.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'xsum_comparison_rouge1.png'") 