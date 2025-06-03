"""
Epoch Progression Analysis for English BERT Gender Bias Across Profession Categories

This script analyzes how gender bias evolves during fine-tuning across different epochs 
(0-3) for three profession categories: male-dominated, female-dominated, and gender-balanced.
It tracks association score changes for male and female person words throughout training.

Features:
- Loads association data from multiple training epochs (1, 2, 3)
- Calculates mean association scores for each epoch and gender combination
- Creates side-by-side comparison across all profession categories
- Visualizes training progression with line plots showing bias evolution
- Generates single merged visualization for comprehensive analysis

Output: Combined plot showing epoch-by-epoch bias changes across profession types
Usage: Run to assess optimal stopping points and training dynamics in bias mitigation
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
epoch_3_data = pd.read_csv("../../results/epochs/replicated_results_epoch3.csv", sep='\t')
epoch_2_data = pd.read_csv("../../results/epochs/replicated_results_epoch2.csv", sep='\t')
epoch_1_data = pd.read_csv("../../results/epochs/replicated_results_epoch1.csv", sep='\t')

# Filter for the "balanced" profession group
balanced_data_3 = epoch_3_data[epoch_3_data['Prof_Gender'] == 'balanced']
balanced_data_2 = epoch_2_data[epoch_2_data['Prof_Gender'] == 'balanced']
balanced_data_1 = epoch_1_data[epoch_1_data['Prof_Gender'] == 'balanced']

# Separate male and female person words for each epoch
balanced_male_data_3 = balanced_data_3[balanced_data_3['Gender'] == 'male']
balanced_female_data_3 = balanced_data_3[balanced_data_3['Gender'] == 'female']

balanced_male_data_2 = balanced_data_2[balanced_data_2['Gender'] == 'male']
balanced_female_data_2 = balanced_data_2[balanced_data_2['Gender'] == 'female']

balanced_male_data_1 = balanced_data_1[balanced_data_1['Gender'] == 'male']
balanced_female_data_1 = balanced_data_1[balanced_data_1['Gender'] == 'female']

# Calculate mean association scores for each epoch
balanced_male_means = [
    balanced_male_data_3['Pre_Assoc'].mean(),  # Epoch 0
    balanced_male_data_1['Post_Assoc'].mean(),  # Epoch 1
    balanced_male_data_2['Post_Assoc'].mean(),  # Epoch 2
    balanced_male_data_3['Post_Assoc'].mean()   # Epoch 3
]

balanced_female_means = [
    balanced_female_data_3['Pre_Assoc'].mean(),  # Epoch 0
    balanced_female_data_1['Post_Assoc'].mean(),  # Epoch 1
    balanced_female_data_2['Post_Assoc'].mean(),  # Epoch 2
    balanced_female_data_3['Post_Assoc'].mean()   # Epoch 3
]

# Filter for the "female" profession group
female_data_3 = epoch_3_data[epoch_3_data['Prof_Gender'] == 'female']
female_data_2 = epoch_2_data[epoch_2_data['Prof_Gender'] == 'female']
female_data_1 = epoch_1_data[epoch_1_data['Prof_Gender'] == 'female']

# Separate male and female person words for each epoch
female_male_data_3 = female_data_3[female_data_3['Gender'] == 'male']
female_female_data_3 = female_data_3[female_data_3['Gender'] == 'female']

female_male_data_2 = female_data_2[female_data_2['Gender'] == 'male']
female_female_data_2 = female_data_2[female_data_2['Gender'] == 'female']

female_male_data_1 = female_data_1[female_data_1['Gender'] == 'male']
female_female_data_1 = female_data_1[female_data_1['Gender'] == 'female']

# Calculate mean association scores for each epoch
female_male_means = [
    female_male_data_3['Pre_Assoc'].mean(),  # Epoch 0
    female_male_data_1['Post_Assoc'].mean(),  # Epoch 1
    female_male_data_2['Post_Assoc'].mean(),  # Epoch 2
    female_male_data_3['Post_Assoc'].mean()   # Epoch 3
]

female_female_means = [
    female_female_data_3['Pre_Assoc'].mean(),  # Epoch 0
    female_female_data_1['Post_Assoc'].mean(),  # Epoch 1
    female_female_data_2['Post_Assoc'].mean(),  # Epoch 2
    female_female_data_3['Post_Assoc'].mean()   # Epoch 3
]

# Filter for the "male" profession group
male_data_3 = epoch_3_data[epoch_3_data['Prof_Gender'] == 'male']
male_data_2 = epoch_2_data[epoch_2_data['Prof_Gender'] == 'male']
male_data_1 = epoch_1_data[epoch_1_data['Prof_Gender'] == 'male']

# Separate male and female person words for each epoch
male_male_data_3 = male_data_3[male_data_3['Gender'] == 'male']
male_female_data_3 = male_data_3[male_data_3['Gender'] == 'female']

male_male_data_2 = male_data_2[male_data_2['Gender'] == 'male']
male_female_data_2 = male_data_2[male_data_2['Gender'] == 'female']

male_male_data_1 = male_data_1[male_data_1['Gender'] == 'male']
male_female_data_1 = male_data_1[male_data_1['Gender'] == 'female']

# Calculate mean association scores for each epoch
male_male_means = [
    male_male_data_3['Pre_Assoc'].mean(),  # Epoch 0
    male_male_data_1['Post_Assoc'].mean(),  # Epoch 1
    male_male_data_2['Post_Assoc'].mean(),  # Epoch 2
    male_male_data_3['Post_Assoc'].mean()   # Epoch 3
]

male_female_means = [
    male_female_data_3['Pre_Assoc'].mean(),  # Epoch 0
    male_female_data_1['Post_Assoc'].mean(),  # Epoch 1
    male_female_data_2['Post_Assoc'].mean(),  # Epoch 2
    male_female_data_3['Post_Assoc'].mean()   # Epoch 3
]

# Create a figure with 3 subplots side by side (1 row, 3 columns)
fig, (ax_male, ax_female, ax_balanced) = plt.subplots(1, 3, figsize=(15, 5))

epochs = [0, 1, 2, 3]

# Plot male-dominated professions (first subplot)
ax_male.plot(epochs, male_male_means, marker='o', color='blue', label='male')
ax_male.plot(epochs, male_female_means, marker='o', color='orange', label='female')
ax_male.set_xticks([0, 1, 2, 3])
ax_male.set_xticklabels([0, 1, 2, 3], fontsize=10)
ax_male.set_xlabel('Epochs', fontsize=12)
ax_male.set_ylabel('Mean Post Association Score', fontsize=12)
ax_male.set_title('Male-dominated Professions (M)', fontsize=14)
ax_male.grid(True, linestyle='--', alpha=0.6)
ax_male.legend()

# Plot female-dominated professions (second subplot)
ax_female.plot(epochs, female_male_means, marker='o', color='blue', label='male')
ax_female.plot(epochs, female_female_means, marker='o', color='orange', label='female')
ax_female.set_xticks([0, 1, 2, 3])
ax_female.set_xticklabels([0, 1, 2, 3], fontsize=10)
ax_female.set_xlabel('Epochs', fontsize=12)
ax_female.set_ylabel('Mean Post Association Score', fontsize=12)
ax_female.set_title('Female-dominated Professions (F)', fontsize=14)
ax_female.grid(True, linestyle='--', alpha=0.6)
ax_female.legend()

# Plot gender-balanced professions (third subplot)
ax_balanced.plot(epochs, balanced_male_means, marker='o', color='blue', label='male')
ax_balanced.plot(epochs, balanced_female_means, marker='o', color='orange', label='female')
ax_balanced.set_xticks([0, 1, 2, 3])
ax_balanced.set_xticklabels([0, 1, 2, 3], fontsize=10)
ax_balanced.set_xlabel('Epochs', fontsize=12)
ax_balanced.set_ylabel('Mean Post Association Score', fontsize=12)
ax_balanced.set_title('Gender-balanced Professions (B)', fontsize=14)
ax_balanced.grid(True, linestyle='--', alpha=0.6)
ax_balanced.legend()

# Adjust layout
plt.tight_layout()

# Save the merged plot
plt.savefig("../../results/plots/english/training_vs_debiasing.png",
            bbox_inches='tight', dpi=300)

plt.show()