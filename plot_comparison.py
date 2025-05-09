import matplotlib.pyplot as plt
import numpy as np

# ROUGE-1 scores from h2o_window.txt
rouge_scores = [0.0384, 0.0419, 0.1417, 0.2939, 0.3408, 0.3411, 0.3476, 0.3282, 0.3087]

# Cosine similarity values
cosine_sim = [0.9404, 0.9512, 0.9561, 0.9595, 0.9604, 0.9609, 0.9604, 0.9570, 0.9521]

# X-axis values (ratios)
x_values = [f"{i}/10" for i in range(1, 10)]

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot ROUGE-1 scores
ax1.plot(x_values, rouge_scores, 'b-o', label='ROUGE-1 Scores', linewidth=2)
ax1.set_title('ROUGE-1 Scores by Local Window Ratio', fontsize=14)
ax1.set_xlabel('Local Window Ratio', fontsize=12)
ax1.set_ylabel('ROUGE-1 Score', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=12)

# Add value labels for ROUGE-1 scores
for i, score in enumerate(rouge_scores):
    ax1.text(i, score, f'{score:.4f}', ha='center', va='bottom', fontsize=8)

# Plot Cosine Similarity
ax2.plot(x_values, cosine_sim, 'r-o', label='Cosine Similarity', linewidth=2)
ax2.set_title('Cosine Similarity by Local Window Ratio', fontsize=14)
ax2.set_xlabel('Local Window Ratio', fontsize=12)
ax2.set_ylabel('Cosine Similarity', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(fontsize=12)

# Add value labels for Cosine Similarity
for i, sim in enumerate(cosine_sim):
    ax2.text(i, sim, f'{sim:.4f}', ha='center', va='bottom', fontsize=8)

# Adjust layout and save
plt.tight_layout()
plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
plt.close() 