import matplotlib.pyplot as plt
import numpy as np

# Data from experiments
# Format: [select, recent, random, streaming, ff, rm, rouge1, rouge2, rougeL, throughput]
a2sf_data = [
    [100, 0.4000],
    [200, 0.4075],
    [300, 0.4110],
    [400, 0.4103],
    [500, 0.4077]
]

h2o_data = [
    [100, 0.3408],
    [200, 0.3841],
    [300, 0.3923],
    [400, 0.3865],
    [500, 0.4015]
]

# A2SF with random=5 and rm=random
# greed_search = [
#     [100, 0.3891],
#     [200, 0.4043]
# ]

# Full Llama-2 model results
llama2_rouge1 = 0.4204

# Calculate total budget (select + recent)
a2sf_budget = [d[0] for d in a2sf_data]
h2o_budget = [d[0] for d in h2o_data]
# greed_budget = [d[0] for d in greed_search]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot A2SF results
plt.plot(a2sf_budget, [d[1] for d in a2sf_data], 'o-', label='A2SF(Layer-wise Ratio/Factor)', color='blue', linewidth=2, markersize=8)

# # Plot A2SF with random results
# plt.plot(greed_budget, [d[1] for d in greed_search], 's-', label='A2SF(50%/50%, Fixed Factor)', color='green', linewidth=2, markersize=8)

# Plot H2O results
plt.plot(h2o_budget, [d[1] for d in h2o_data], 'o-', label='H2O(50%/50%)', color='red', linewidth=2, markersize=8)

# Plot full Llama-2 model results as horizontal line
plt.axhline(y=llama2_rouge1, color='black', linestyle='--', label='Llama-2', linewidth=2)

# Customize the plot
plt.xlabel('Total Budget (select + recent)', fontsize=12)
plt.ylabel('ROUGE-1 Score', fontsize=12)
plt.title('CNN DailyMail: ROUGE-1 Score vs Total Budget', fontsize=14)
plt.grid(True, alpha=0.3)

# Set y-axis limits with some padding
plt.ylim(0.33, 0.43)

# Add legend
plt.legend(loc='lower right', fontsize=10)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('plots/rouge-1/cnndm_comparison_rouge1.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'cnndm_comparison_rouge1.png'") 