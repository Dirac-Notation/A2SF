import numpy as np
import matplotlib.pyplot as plt

# Define x-axis range
x = np.linspace(0, 8192, 1000)

# Define the sigmoid function with exponential term -a(x-4096)
def sigmoid(x, a, b):
    """
    Sigmoid function: 1 / (1 + exp(-a(x-4096)))
    where a is the scaling parameter and 4096 is the zero point
    """
    return 1 / (1 + np.exp(-a * (x - b)))

# Values of a to plot
a_values = [0.0, 0.0001, 0.0005, 0.001, 0.01, 0.1]

# Create the plot
plt.figure(figsize=(6, 4))

# Plot sigmoid for each value of a
for a in a_values:
    y = sigmoid(x, a, 4096)
    plt.plot(x, y, label=f'a = {a}')

# Add vertical line at x = 4096 (zero point)
plt.axvline(x=4096, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Customize the plot
plt.xlabel('x', fontsize=12)
plt.ylabel('Sigmoid(x)', fontsize=12)
plt.title('Sigmoid Function', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='best')
plt.xlim(0, 8192)
plt.ylim(0, 1)

# Show the plot
plt.tight_layout()
plt.savefig('plots/sigmoid/graph.png', dpi=300, bbox_inches='tight')
plt.close()