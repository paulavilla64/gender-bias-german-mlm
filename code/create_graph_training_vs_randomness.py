import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
original_data = pd.read_csv("../data/results_EN.csv", sep='\t')

# Calculate the mean association score for each gender category
gender_categories = ['female', 'male', 'balanced']

gender_means = {
    category: original_data[original_data['Prof_Gender'] == category]['Post_Assoc'].mean() for category in gender_categories
}


# Load the second CSV file
random_seed_116 = pd.read_csv("../data/replicated_randomized1.csv", sep='\t')

# Calculate the mean association score for each gender category in the second dataset
random_seed_116_means = {
    category: random_seed_116[random_seed_116['Prof_Gender'] == category]['Post_Assoc'].mean() for category in gender_categories
}


# Load the third CSV file
random_seed_387 = pd.read_csv("../data/replicated_randomized2.csv", sep='\t')

# Calculate the mean association score for each gender category in the second dataset
random_seed_387_means = {
    category: random_seed_387[random_seed_387['Prof_Gender'] == category]['Post_Assoc'].mean() for category in gender_categories
}


# Load the fourth CSV file
random_seed_1980 = pd.read_csv("../data/replicated_randomized3.csv", sep='\t')

# Calculate the mean association score for each gender category in the second dataset
random_seed_1980_means = {
    category: random_seed_1980[random_seed_1980['Prof_Gender'] == category]['Post_Assoc'].mean() for category in gender_categories
}



gender_means_pre = {
    category: original_data[original_data['Prof_Gender'] == category]['Pre_Assoc'].mean() for category in gender_categories
}


# Calculate the mean association score for each gender category in the second dataset
random_seed_116_means_pre = {
    category: random_seed_116[random_seed_116['Prof_Gender'] == category]['Pre_Assoc'].mean() for category in gender_categories
}


# Calculate the mean association score for each gender category in the second dataset
random_seed_387_means_pre = {
    category: random_seed_387[random_seed_387['Prof_Gender'] == category]['Pre_Assoc'].mean() for category in gender_categories
}

# Calculate the mean association score for each gender category in the second dataset
random_seed_1980_means_pre = {
    category: random_seed_1980[random_seed_1980['Prof_Gender'] == category]['Pre_Assoc'].mean() for category in gender_categories
}


# Create the plot
fig, ax = plt.subplots()

# Plot the mean association scores for random seed = 42
x_labels = list(gender_means.keys())
y_values = list(gender_means.values())
ax.scatter(x_labels, y_values, s=200, color='blue',
           label='Random seed 1', edgecolors='black', alpha=0.7)

# Plot the mean association scores for the second dataset with red crosses
random_seed_116_y_values = [random_seed_116_means[category]
                            for category in gender_categories]
ax.scatter(x_labels, random_seed_116_y_values, s=200, color='red',
            label='Random seed 2', alpha=0.7)

# Plot the mean association scores for the third dataset with purple crosses
random_seed_387_y_values = [random_seed_387_means[category]
                            for category in gender_categories]
ax.scatter(x_labels, random_seed_387_y_values, s=200, color='purple',
         label='Random seed 3', alpha=0.7)

# Plot the mean association scores for the fourth dataset with brown crosses
random_seed_1980_y_values = [random_seed_1980_means[category]
                             for category in gender_categories]
ax.scatter(x_labels, random_seed_1980_y_values, s=200, color='brown',
            label='Random seed 4', alpha=0.7)

# PRE

# Plot the mean association scores for the second dataset with red crosses
random_seed_116_y_values_pre = [random_seed_116_means_pre[category]
                                for category in gender_categories]
ax.scatter(x_labels, random_seed_116_y_values_pre, s=200, color='green',
               marker='^', label='Random seed 1 (Pre)', alpha=0.7)

# # Plot the mean association scores for the third dataset with purple crosses
# random_seed_387_y_values_pre = [random_seed_387_means_pre[category]
#                                 for category in gender_categories]
# ax.scatter(x_labels, random_seed_387_y_values_pre, s=200, color='purple',
#                marker='^', label='Random seed = 387', alpha=0.7)

# # Plot the mean association scores for the fourth dataset with green crosses
# random_seed_1980_y_values_pre = [random_seed_1980_means_pre[category]
#                              for category in gender_categories]
# ax.scatter(x_labels, random_seed_1980_y_values_pre, s=200, color='green',
#            marker='x', label='Random seed = 1980', alpha=0.7)



# Add labels and title
ax.set_xlabel('Gender Categories', fontsize=12)
ax.set_ylabel('Mean Association Score', fontsize=12)
ax.set_title('Mean Post Association Scores by Gender Categories', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)

# Add green circles at specified positions
green_circle_y_values = [0.11, 0.17, 0.135]
ax.scatter(
    x_labels,  # x-axis positions are the gender categories
    green_circle_y_values, 
    s=200, 
    color='green', 
    marker='x', 
    label='Random seed 1 (Their score)', 
    alpha=0.7
)

# Collect all the handles and labels from the existing legend
handles, labels = ax.get_legend_handles_labels()

# Define the desired order of legend items
desired_order = [
    'Random seed 1 (Pre)',
    'Random seed 1 (Their score)',
    'Random seed 1',
    'Random seed 2',
    'Random seed 3',
    'Random seed 4'
]

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

# Display the plot
plt.tight_layout()

# Save plot
plt.savefig("../data/plots/graph_training_vs_randomness.png",
            bbox_inches='tight')

plt.show()




