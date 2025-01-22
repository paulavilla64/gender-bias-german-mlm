import pandas as pd
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv("../data/epochs/replicated_results_epoch3.csv", delimiter='\t')

# Inspect the data
print(df.head())
print(df.info())

# Filter rows for statistically male professions
male_professions = df[df['Prof_Gender'] == 'male']

# Select necessary columns
columns_needed = ['Profession', 'Gender', 'Pre_Assoc', 'Post_Assoc']
male_professions_filtered = male_professions[columns_needed]

# Group and calculate mean
grouped = male_professions_filtered.groupby(
    ['Profession', 'Gender']).mean().reset_index()

# Pivot for plotting
pre_data_male = grouped.pivot(
    index='Profession', columns='Gender', values='Pre_Assoc')
post_data_male = grouped.pivot(
    index='Profession', columns='Gender', values='Post_Assoc')

# Define the custom order for professions
profession_order_male = [
    'carpenter', 'bus mechanic', 'conductor', 'heating mechanic', 'taper', 'firefighter',
    'mining machine operator', 'repairer', 'operating engineer', 'electrician', 'security system installer',
    'mason', 'mobile equipment mechanic', 'floor installer', 'electrical installer', 'roofer',
    'plumber', 'logging worker', 'steel worker', 'service technician'
]

# Reorder the DataFrame according to the custom profession order
pre_data_male = pre_data_male.loc[profession_order_male]
post_data_male = post_data_male.loc[profession_order_male]

# Define fixed y-limits
y_min_male, y_max_male = -1.6, 0.8

# Plot the data
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Set title for the first plot
axes[0].set_title("Statistically Male Professions Epoch 3", fontsize=16)

# Pre-association plot
pre_bars_male = pre_data_male.plot(kind='bar', ax=axes[0], color=[
    'orange', 'blue'], legend=False, zorder=3)
axes[0].set_ylabel('Before fine-tuning', fontsize=12, labelpad=15)
axes[0].set_facecolor('#f0f0f0')  # Grey-ish background for the plot
axes[0].set_xticklabels(pre_data_male.index, rotation=45, ha='right')
# Set the same y-limits for both plots
axes[0].set_ylim([y_min_male, y_max_male])

# Add white grid lines (y and x axes)
axes[0].grid(axis='y', color='white', linewidth=1.0, zorder=1)
axes[0].grid(axis='x', color='white', linewidth=1.0, zorder=1)

# Post-association plot
post_bars_male = post_data_male.plot(kind='bar', ax=axes[1], color=[
    'orange', 'blue'], legend=False, zorder=3)
axes[1].set_ylabel('After fine-tuning', fontsize=12, labelpad=15)
axes[1].set_facecolor('#f0f0f0')  # Grey-ish background for the plot
axes[1].set_xticklabels(post_data_male.index, rotation=45, ha='right')
# Set the same y-limits for both plots
axes[1].set_ylim([y_min_male, y_max_male])

# Make sure the x-scale includes "-0.5" on the second plot
axes[1].set_xlim(left=-0.5, right=len(post_data_male)-0.5)

# Add white grid lines (y and x axes)
axes[1].grid(axis='y', color='white', linewidth=1.0, zorder=1)
axes[1].grid(axis='x', color='white', linewidth=1.0, zorder=1)

# Add the legend below the plots
fig.legend(['Female', 'Male'], loc='lower left',
           bbox_to_anchor=(0.5, 0.05), ncol=2)

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig("../data/plots/epoch3_statistically_male_professions.png",
            bbox_inches='tight')
plt.show()


# Filter rows for statistically female professions
female_professions = df[df['Prof_Gender'] == 'female']

# Select necessary columns
columns_needed = ['Profession', 'Gender', 'Pre_Assoc', 'Post_Assoc']
female_professions_filtered = female_professions[columns_needed]

# Group and calculate mean
grouped = female_professions_filtered.groupby(
    ['Profession', 'Gender']).mean().reset_index()

# Pivot for plotting
pre_data_female = grouped.pivot(
    index='Profession', columns='Gender', values='Pre_Assoc')
post_data_female = grouped.pivot(
    index='Profession', columns='Gender', values='Post_Assoc')

# Define the custom order for professions
profession_order_female = [
    'housekeeper', 'registered nurse', 'receptionist', 'vocational nurse', 'secretary', 'childcare worker',
    'kindergarten teacher', 'teacher assistant', 'dietitian', 'bookkeeper', 'hairdresser',
    'medical records technician', 'phlebotomist', 'health aide', 'dental assistant', 'medical assistant',
    'paralegal', 'speech-language pathologist', 'billing clerk', 'dental hygienist'
]

# Reorder the DataFrame according to the custom profession order
pre_data_female = pre_data_female.loc[profession_order_female]
post_data_female = post_data_female.loc[profession_order_female]

# Define fixed y-limits
y_min_female, y_max_female = -3.5, 1.5

# Plot the data
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Set title for the first plot
axes[0].set_title("Epoch3 Statistically Female Professions", fontsize=16)

# Pre-association plot
pre_bars_female = pre_data_female.plot(kind='bar', ax=axes[0], color=[
    'orange', 'blue'], legend=False, zorder=3)
axes[0].set_ylabel('Before fine-tuning', fontsize=12, labelpad=15)
axes[0].set_facecolor('#f0f0f0')  # Grey-ish background for the plot
axes[0].set_xticklabels(pre_data_female.index, rotation=45, ha='right')
# Set the same y-limits for both plots
axes[0].set_ylim([y_min_female, y_max_female])

# Add white grid lines (y and x axes)
axes[0].grid(axis='y', color='white', linewidth=1.0, zorder=1)
axes[0].grid(axis='x', color='white', linewidth=1.0, zorder=1)

# Post-association plot
post_bars_female = post_data_female.plot(kind='bar', ax=axes[1], color=[
    'orange', 'blue'], legend=False, zorder=3)
axes[1].set_ylabel('After fine-tuning', fontsize=12, labelpad=15)
axes[1].set_facecolor('#f0f0f0')  # Grey-ish background for the plot
axes[1].set_xticklabels(post_data_female.index, rotation=45, ha='right')
# Set the same y-limits for both plots
axes[1].set_ylim([y_min_female, y_max_female])

# Make sure the x-scale includes "-0.5" on the second plot
axes[1].set_xlim(left=-0.5, right=len(post_data_female)-0.5)

# Add white grid lines (y and x axes)
axes[1].grid(axis='y', color='white', linewidth=1.0, zorder=1)
axes[1].grid(axis='x', color='white', linewidth=1.0, zorder=1)

# Add the legend below the plots
fig.legend(['Female', 'Male'], loc='lower left',
           bbox_to_anchor=(0.5, 0.05), ncol=2)

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig("../data/plots/epoch3_statistically_female_professions.png",
            bbox_inches='tight')
plt.show()


# Filter rows for balanced professions
balanced_professions = df[df['Prof_Gender'] == 'balanced']

# Select necessary columns
columns_needed = ['Profession', 'Gender', 'Pre_Assoc', 'Post_Assoc']
balanced_professions_filtered = balanced_professions[columns_needed]

# Group and calculate mean
grouped = balanced_professions_filtered.groupby(
    ['Profession', 'Gender']).mean().reset_index()

# Pivot for plotting
pre_data_balanced = grouped.pivot(
    index='Profession', columns='Gender', values='Pre_Assoc')
post_data_balanced = grouped.pivot(
    index='Profession', columns='Gender', values='Post_Assoc')

# Define the custom order for professions
profession_order_balanced = [
    'electrical assembler', 'judge', 'insurance sales agent', 'crossing guard', 'statistician', 'photographer',
    'mail sorter', 'dispatcher', 'director of religious activities', 'medical scientist', 'insurance underwriter',
    'bartender', 'training specialist', 'lifeguard', 'lodging manager', 'healthcare practitioner',
    'sales agent', 'mail clerk', 'salesperson', 'order clerk'
]

# Reorder the DataFrame according to the custom profession order
pre_data_balanced = pre_data_balanced.loc[profession_order_balanced]
post_data_balanced = post_data_balanced.loc[profession_order_balanced]

# Define fixed y-limits
y_min_balanced, y_max_balanced = -1.6, 0.9

# Plot the data
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Set title for the first plot
axes[0].set_title("Epoch3 Statistically Balanced Professions", fontsize=16)

# Pre-association plot
pre_bars_balanced = pre_data_balanced.plot(kind='bar', ax=axes[0], color=[
    'orange', 'blue'], legend=False, zorder=3)
axes[0].set_ylabel('Before fine-tuning', fontsize=12, labelpad=15)
axes[0].set_facecolor('#f0f0f0')  # Grey-ish background for the plot
axes[0].set_xticklabels(pre_data_balanced.index, rotation=45, ha='right')
# Set the same y-limits for both plots
axes[0].set_ylim([y_min_balanced, y_max_balanced])

# Add white grid lines (y and x axes)
axes[0].grid(axis='y', color='white', linewidth=1.0, zorder=1)
axes[0].grid(axis='x', color='white', linewidth=1.0, zorder=1)

# Post-association plot
post_bars_balanced = post_data_balanced.plot(kind='bar', ax=axes[1], color=[
    'orange', 'blue'], legend=False, zorder=3)
axes[1].set_ylabel('After fine-tuning', fontsize=12, labelpad=15)
axes[1].set_facecolor('#f0f0f0')  # Grey-ish background for the plot
axes[1].set_xticklabels(post_data_balanced.index, rotation=45, ha='right')
# Set the same y-limits for both plots
axes[1].set_ylim([y_min_balanced, y_max_balanced])

# Make sure the x-scale includes "-0.5" on the second plot
axes[1].set_xlim(left=-0.5, right=len(post_data_balanced)-0.5)

# Add white grid lines (y and x axes)
axes[1].grid(axis='y', color='white', linewidth=1.0, zorder=1)
axes[1].grid(axis='x', color='white', linewidth=1.0, zorder=1)

# Add the legend below the plots
fig.legend(['Female', 'Male'], loc='lower left',
           bbox_to_anchor=(0.5, 0.05), ncol=2)

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig("../data/plots/epoch3_statistically_balanced_professions.png",
            bbox_inches='tight')
plt.show()
