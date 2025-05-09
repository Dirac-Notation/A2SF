import matplotlib.pyplot as plt
import numpy as np

# Data from experiments
# Format: [select, recent, random, streaming, ff, rm, rouge1, rouge2, rougeL, throughput]
a2sf_data = [
    [50, 0.3092],
    [100, 0.3116],
    [150, 0.3133],
]

h2o_data = [
    [50, 0.1549],
    [100, 0.3033],
    [150, 0.3143],
]

# Full Llama-2 model results
llama2_rouge1 = 0.3012

# Calculate total budget (select + recent)
a2sf_budget = [d[0] for d in a2sf_data]
h2o_budget = [d[0] for d in h2o_data]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot A2SF results
plt.plot(a2sf_budget, [d[1] for d in a2sf_data], 'o-', label='A2SF(Layer-wise Ratio/Factor)', color='blue', linewidth=2, markersize=8)

# Plot H2O results
plt.plot(h2o_budget, [d[1] for d in h2o_data], 'o-', label='H2O(50%/50%)', color='red', linewidth=2, markersize=8)

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
plt.savefig('plots/rouge-1/xsum_comparison_rouge1.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'xsum_comparison_rouge1.png'") 