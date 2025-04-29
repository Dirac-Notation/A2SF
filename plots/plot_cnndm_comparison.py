import matplotlib.pyplot as plt
import numpy as np

# Data from experiments
# Format: [select, recent, random, streaming, ff, rm, rouge1, rouge2, rougeL, throughput]
a2sf_data = [
    [50, 50, 0, 0, 0.99, 'att', 0.3697, 0.1266, 0.2424, 27.03],
    [100, 100, 0, 0, 0.99, 'att', 0.3868, 0.1401, 0.2536, 27.24],
    [150, 150, 0, 0, 0.99, 'att', 0.3991, 0.1529, 0.2686, 26.47],
    [200, 200, 0, 0, 0.99, 'att', 0.4088, 0.1630, 0.2785, 26.11],
    [250, 250, 0, 0, 0.99, 'att', 0.4007, 0.1545, 0.2723, 25.69],
    [300, 300, 0, 0, 0.99, 'att', 0.4078, 0.1653, 0.2815, 25.37]
]

h2o_data = [
    [50, 50, 0, 0, 1.0, 'att', 0.3408, 0.1162, 0.2290, 26.95],
    [100, 100, 0, 0, 1.0, 'att', 0.3841, 0.1376, 0.2518, 26.61],
    [150, 150, 0, 0, 1.0, 'att', 0.3923, 0.1448, 0.2628, 25.78],
    [200, 200, 0, 0, 1.0, 'att', 0.3865, 0.1440, 0.2588, 25.82],
    [250, 250, 0, 0, 1.0, 'att', 0.4015, 0.1538, 0.2658, 25.64],
    [300, 300, 0, 0, 1.0, 'att', 0.4035, 0.1476, 0.2673, 25.25]
]

# A2SF with random=5 and rm=random
a2sf_random_data = [
    [45, 50, 5, 0, 0.99, 'random', 0.3752, 0.1285, 0.2457, 21.83],
    [95, 100, 5, 0, 0.99, 'random', 0.3869, 0.1398, 0.2550, 22.21],
    [145, 150, 5, 0, 0.99, 'random', 0.3965, 0.1458, 0.2567, 21.74],
    [195, 200, 5, 0, 0.99, 'random', 0.4052, 0.1579, 0.2762, 21.66],
    [245, 250, 5, 0, 0.99, 'random', 0.4018, 0.1599, 0.2755, 21.54],
    [295, 300, 5, 0, 0.99, 'random', 0.4111, 0.1678, 0.2852, 21.44]
]

# Full Llama-2 model results
llama2_rouge1 = 0.4204

# Calculate total budget (select + recent)
a2sf_budget = [d[0] + d[1] for d in a2sf_data]
h2o_budget = [d[0] + d[1] for d in h2o_data]
a2sf_random_budget = [d[0] + d[1] + d[2] for d in a2sf_random_data]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot A2SF results
plt.plot(a2sf_budget, [d[6] for d in a2sf_data], 'o-', label='A2SF (ff=0.99, rm=att)', color='blue', linewidth=2, markersize=8)

# Plot A2SF with random results
plt.plot(a2sf_random_budget, [d[6] for d in a2sf_random_data], 's-', label='A2SF (ff=0.99, rm=random)', color='green', linewidth=2, markersize=8)

# Plot H2O results
plt.plot(h2o_budget, [d[6] for d in h2o_data], 'o-', label='H2O', color='red', linewidth=2, markersize=8)

# Plot full Llama-2 model results as horizontal line
max_budget = max(max(a2sf_budget), max(h2o_budget), max(a2sf_random_budget))
plt.axhline(y=llama2_rouge1, color='black', linestyle='--', label='Llama-2', linewidth=2)

# Customize the plot
plt.xlabel('Total Budget (select + recent)', fontsize=12)
plt.ylabel('ROUGE-1 Score', fontsize=12)
plt.title('ROUGE-1 Score vs Total Budget', fontsize=14)
plt.grid(True, alpha=0.3)

# Set y-axis limits with some padding
plt.ylim(0.33, 0.43)

# Add legend
plt.legend(loc='lower right', fontsize=10)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('plots/cnndm_comparison_rouge1.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'cnndm_comparison_rouge1.png'") 