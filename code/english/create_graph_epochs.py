import pandas as pd
import matplotlib.pyplot as plt


# Load the datasets
epoch_3_data = pd.read_csv("../../data/epochs/replicated_results_epoch3.csv", sep='\t')
epoch_2_data = pd.read_csv("../../data/epochs/replicated_results_epoch2.csv", sep='\t')
epoch_1_data = pd.read_csv("../../data/epochs/replicated_results_epoch1.csv", sep='\t')


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

# Create the plot for balanced
fig_bal, ax_bal = plt.subplots()

# Plot the scores for B_male and B_female
epochs = [0, 1, 2, 3]
ax_bal.plot(epochs, balanced_male_means, marker='o',
            color='blue', label='male')
ax_bal.plot(epochs, balanced_female_means, marker='o',
            color='orange', label='female')

# Set custom x-axis ticks
ax_bal.set_xticks([0, 1, 2, 3])  # Epochs as integers
ax_bal.set_xticklabels([0, 1, 2, 3], fontsize=10)  # Set labels as full numbers

# Add labels and title
ax_bal.set_xlabel('Epochs', fontsize=12)
ax_bal.set_ylabel('Mean Post Association Score', fontsize=12)
ax_bal.set_title(
    'Gender-balanced Professions (B)', fontsize=14)
ax_bal.grid(True, linestyle='--', alpha=0.6)

# Show the legend
ax_bal.legend()

# Display the plot
plt.tight_layout()

# Save the plot
plt.savefig("../../data/plots/epochs/graph_training_vs_debiasing_balanced.png",
            bbox_inches='tight')

plt.show()


# Create the plot for female
fig_fem, ax_fem = plt.subplots()

# Plot the scores for F_male and F_female
epochs = [0, 1, 2, 3]
ax_fem.plot(epochs, female_male_means, marker='o',
            color='blue', label='male')
ax_fem.plot(epochs, female_female_means, marker='o',
            color='orange', label='female')

# Set custom x-axis ticks
ax_fem.set_xticks([0, 1, 2, 3])  # Epochs as integers
ax_fem.set_xticklabels([0, 1, 2, 3], fontsize=10)  # Set labels as full numbers

# Add labels and title
ax_fem.set_xlabel('Epochs', fontsize=12)
ax_fem.set_ylabel('Mean Post Association Score', fontsize=12)
ax_fem.set_title(
    'Female-dominated Professions (F)', fontsize=14)
ax_fem.grid(True, linestyle='--', alpha=0.6)

# Show the legend
ax_fem.legend()

# Display the plot
plt.tight_layout()

# Save the plot
plt.savefig("../../data/plots/epochs/graph_training_vs_debiasing_female.png",
            bbox_inches='tight')

plt.show()

# Create the plot for male
fig_mal, ax_mal = plt.subplots()

# Plot the scores for M_male and M_female
epochs = [0, 1, 2, 3]
ax_mal.plot(epochs, male_male_means, marker='o', color='blue', label='male')
ax_mal.plot(epochs, male_female_means, marker='o',
            color='orange', label='female')

# Set custom x-axis ticks
ax_mal.set_xticks([0, 1, 2, 3])  # Epochs as integers
ax_mal.set_xticklabels([0, 1, 2, 3], fontsize=10)  # Set labels as full numbers

# Add labels and title
ax_mal.set_xlabel('Epochs', fontsize=12)
ax_mal.set_ylabel('Mean Post Association Score', fontsize=12)
ax_mal.set_title(
    'Male-dominated Professions (M)', fontsize=14)
ax_mal.grid(True, linestyle='--', alpha=0.6)

# Show the legend
ax_mal.legend()

# Display the plot
plt.tight_layout()

# Save the plot
plt.savefig("../../data/plots/epochs/graph_training_vs_debiasing_male.png",
            bbox_inches='tight')

plt.show()
