import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Male Professions (M)',
              'Male Professions (F)',
              'Female Professions (M)',
              'Female Professions (F)']

probabilities = [0.1725, 0.003494, 0.004226, 0.2238]
colors = ['blue', 'orange', 'blue', 'orange']

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(categories, probabilities, color=colors, alpha=0.8)

# Add legend
legend_labels = ['Male Person Words', 'Female Person Words']
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color='blue', alpha=0.8),
                   plt.Rectangle((0, 0), 1, 1, color='orange', alpha=0.8)],
          labels=legend_labels, loc='upper left', fontsize=10)

# Customize axes
ax.set_ylim(0, 0.5)
ax.set_yticks(np.arange(0, 0.51, 0.1))
ax.set_ylabel('Relative Probabilities', fontsize=12)
ax.set_xlabel('Categories', fontsize=12)
ax.set_title(
    'Relative Probabilities of Person Words by Profession Type', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=30, ha='right', fontsize=10)

# Add value annotations on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.4f}',
            ha='center', fontsize=10)

# Show plot and save
plt.tight_layout()
plt.savefig("../data/plots/graph_gender_marking.png")
plt.show()
