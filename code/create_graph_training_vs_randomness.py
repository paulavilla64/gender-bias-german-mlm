import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Load the CSV files
original_data = pd.read_csv("../data/output_csv_files/english/results_EN.csv", sep='\t')
random_seed_116 = pd.read_csv("../data/random_seeds/replicated_randomized1.csv", sep='\t')
random_seed_387 = pd.read_csv("../data/random_seeds/replicated_randomized2.csv", sep='\t')
random_seed_1980 = pd.read_csv("../data/random_seeds/replicated_randomized3.csv", sep='\t')

# Gender categories
gender_categories = ['female', 'male', 'balanced']

# Create a mapping from category names to positions
category_positions = {category: i for i, category in enumerate(gender_categories)}

# Calculate the mean post-association scores for each random seed
random_seeds_post = {
    '42': {category: original_data[original_data['Prof_Gender'] == category]['Post_Assoc'].mean() for category in gender_categories},
    '116': {category: random_seed_116[random_seed_116['Prof_Gender'] == category]['Post_Assoc'].mean() for category in gender_categories},
    '387': {category: random_seed_387[random_seed_387['Prof_Gender'] == category]['Post_Assoc'].mean() for category in gender_categories},
    '1980': {category: random_seed_1980[random_seed_1980['Prof_Gender'] == category]['Post_Assoc'].mean() for category in gender_categories}
}

# Calculate pre-association scores (these should be the same across seeds)
pre_assoc_means = {
    category: original_data[original_data['Prof_Gender'] == category]['Pre_Assoc'].mean() for category in gender_categories
}

# Calculate average and standard deviation of post-associations across all seeds
avg_post_assoc = {}
std_post_assoc = {}

for category in gender_categories:
    values = [random_seeds_post[seed][category] for seed in random_seeds_post.keys()]
    avg_post_assoc[category] = np.mean(values)
    std_post_assoc[category] = np.std(values)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot individual post-association scores for each random seed
colors = {'42': 'blue', '116': 'red', '387': 'purple', '1980': 'brown'}
for seed, color in colors.items():
    y_values = [random_seeds_post[seed][category] for category in gender_categories]
    ax.scatter(gender_categories, y_values, s=200, color=color,
               label=f'Random seed {seed}', alpha=0.7)

# Plot pre-association scores
pre_y_values = [pre_assoc_means[category] for category in gender_categories]
ax.scatter(gender_categories, pre_y_values, s=200, color='green',
           marker='^', label='Their score (Pre)', alpha=0.7)

# Plot "their score" (original paper)
their_score_y_values = [0.11, 0.17, 0.135]
ax.scatter(gender_categories, their_score_y_values, s=200, color='green',
           marker='x', label='Their score (Post)', alpha=0.7)

# Plot average post-association line
avg_values = [avg_post_assoc[category] for category in gender_categories]
std_values = [std_post_assoc[category] for category in gender_categories]

# Plot the average line
ax.plot(gender_categories, avg_values, 'k-', linewidth=1.5, label='Average across seeds')

# Set the x-tick positions and labels
ax.set_xticks(range(len(gender_categories)))
ax.set_xticklabels(gender_categories)

# Add custom SD range with T-shaped whiskers - MUCH MORE PROMINENT
for i, category in enumerate(gender_categories):
    # Vertical line - much thicker and fully opaque
    ax.vlines(x=i, ymin=avg_values[i] - std_values[i], ymax=avg_values[i] + std_values[i], 
              colors='grey', linestyles='-', linewidth=1.0, alpha=1.0)
    
    # Top horizontal whisker - much thicker and fully opaque
    ax.hlines(y=avg_values[i] + std_values[i], xmin=i - 0.15, xmax=i + 0.15, 
              colors='grey', linestyles='-', linewidth=1.0, alpha=1.0)
    
    # Bottom horizontal whisker - much thicker and fully opaque
    ax.hlines(y=avg_values[i] - std_values[i], xmin=i - 0.15, xmax=i + 0.15, 
              colors='grey', linestyles='-', linewidth=1.0, alpha=1.0)

# Define text positions customized for each SD annotation to avoid overlap
text_offsets = {
    'female': (5, 25),        # Move first SD much more up
    'male': (5, 10),          # Default position
    'balanced': (15, -20)     # Move third SD more down AND to the right
}

# Add SD text annotations with customized positions
for i, category in enumerate(gender_categories):
    offset_x, offset_y = text_offsets[category]
    ax.annotate(f'SD: {std_values[i]:.3f}', 
                xy=(category, avg_values[i]), 
                xytext=(offset_x, offset_y), 
                textcoords='offset points',
                fontsize=8,
                color='dimgrey')

# Add labels and title
ax.set_xlabel('Gender Categories', fontsize=12)
ax.set_ylabel('Mean Association Score', fontsize=12)
ax.set_title('Mean Post Association Scores by Gender Categories', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)

# Define the desired order of legend items
desired_order = [
    'Their score (Pre)',
    'Their score (Post)',
    'Random seed 42',
    'Random seed 116',
    'Random seed 387',
    'Random seed 1980',
    'Average across seeds',
    'Standard deviation'
]

# Collect all the handles and labels from the existing legend
handles, labels = ax.get_legend_handles_labels()

# Add a custom legend entry for standard deviation - MUCH MORE PROMINENT
sd_handle = Line2D([0], [0], color='grey', linewidth=1.0, linestyle='-', alpha=0.85)
handles.append(sd_handle)
labels.append('Standard deviation')

# Create a mapping of labels to handles
label_to_handle = dict(zip(labels, handles))

# Reorder the handles and labels based on the desired order
ordered_handles = [label_to_handle[label] for label in desired_order if label in label_to_handle]
ordered_labels = [label for label in desired_order if label in label_to_handle]

# Set the legend with the reordered handles and labels
ax.legend(
    ordered_handles, 
    ordered_labels, 
    loc='upper left', 
    bbox_to_anchor=(1, 1),  # Position legend outside the plot area
    fontsize=10
)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the right boundary

# Save plot
plt.savefig("../data/plots/english/graph_training_vs_randomness_with_avg.png",
            bbox_inches='tight', dpi=300)

plt.show()