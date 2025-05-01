import matplotlib.pyplot as plt
import numpy as np
import re
# Load data from factorsearch_cnndm3.txt
with open('factorsearch_cnndm3.txt', 'r') as file:
    lines = file.readlines()

x = [f"{i/1000:.3f}" for i in range(900, 1001)]
length = len(x)

lines = lines[2:]
lines = [lines[4*i:4*i+4] for i in range(0, 3*length)]

budget_100 = [lines[3*i][1] for i in range(0, length)]
budget_200 = [lines[3*i+1][1] for i in range(0, length)]
budget_300 = [lines[3*i+2][1] for i in range(0, length)]

rouge_100 = [float(re.search(r'ROUGE-1:\s*([0-9.]+)', budget_100[i]).group(1)) for i in range(0, length)]
rouge_200 = [float(re.search(r'ROUGE-1:\s*([0-9.]+)', budget_200[i]).group(1)) for i in range(0, length)]
rouge_300 = [float(re.search(r'ROUGE-1:\s*([0-9.]+)', budget_300[i]).group(1)) for i in range(0, length)]

# Create figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot for budget 100
max_idx_100 = np.argmax(rouge_100)
ax1.plot(x, rouge_100, 'b-', linewidth=2)
ax1.axvline(x=x[max_idx_100], color='r', linestyle='--', linewidth=1)
ax1.text(x[max_idx_100], ax1.get_ylim()[0], x[max_idx_100], 
         horizontalalignment='center', verticalalignment='bottom')
ax1.set_title('Budget 100')
ax1.set_xlabel('Forgetting Factor')
ax1.set_ylabel('ROUGE-1 Score')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xticks([x[0], x[-1]])  # Show only min and max x values

# Plot for budget 200
max_idx_200 = np.argmax(rouge_200)
ax2.plot(x, rouge_200, 'g-', linewidth=2)
ax2.axvline(x=x[max_idx_200], color='r', linestyle='--', linewidth=1)
ax2.text(x[max_idx_200], ax2.get_ylim()[0], x[max_idx_200], 
         horizontalalignment='center', verticalalignment='bottom')
ax2.set_title('Budget 200')
ax2.set_xlabel('Forgetting Factor')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xticks([x[0], x[-1]])  # Show only min and max x values

# Plot for budget 300
max_idx_300 = np.argmax(rouge_300)
ax3.plot(x, rouge_300, 'purple', linewidth=2)
ax3.axvline(x=x[max_idx_300], color='r', linestyle='--', linewidth=1)
ax3.text(x[max_idx_300], ax3.get_ylim()[0], x[max_idx_300], 
         horizontalalignment='center', verticalalignment='bottom')
ax3.set_title('Budget 300')
ax3.set_xlabel('Forgetting Factor')
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.set_xticks([x[0], x[-1]])  # Show only min and max x values

# Adjust layout and save
plt.tight_layout()
plt.savefig('plots/factorsearch_results.png', dpi=300, bbox_inches='tight')
plt.close()

