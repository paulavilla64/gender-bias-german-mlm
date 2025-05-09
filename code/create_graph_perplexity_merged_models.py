import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

## FOR MULTIPLE MODELS ##

print("\nCreating perplexity comparison bar graph for three models...")

# Load the CSV files for all models
dbmdz_neutral_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_gender_neutral/dbmdz/all_seeds_bias_gender_neutral_dbmdz_results_one_MASK.csv')
google_neutral_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_gender_neutral/google_bert/all_seeds_bias_gender_neutral_google_bert_results_one_MASK.csv')
deepset_neutral_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_gender_neutral/deepset_bert/all_seeds_bias_gender_neutral_deepset_bert_results.csv')
distilbert_neutral_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_gender_neutral/distilbert/all_seeds_bias_gender_neutral_distilbert_results_one_MASK.csv')

# Calculate average perplexity for each model
dbmdz_neutral_data = dbmdz_neutral_results.groupby(['model']).agg({
    'male_perplexity': 'mean',
    'female_perplexity': 'mean'
}).reset_index()

google_neutral_data = google_neutral_results.groupby(['model']).agg({
    'male_perplexity': 'mean',
    'female_perplexity': 'mean'
}).reset_index()

deepset_neutral_data = deepset_neutral_results.groupby(['model']).agg({
    'male_perplexity': 'mean',
    'female_perplexity': 'mean'
}).reset_index()

distilbert_neutral_data = distilbert_neutral_results.groupby(['model']).agg({
    'male_perplexity': 'mean',
    'female_perplexity': 'mean'
}).reset_index()

# Create the figure
fig, ax = plt.subplots(figsize=(15, 6))

# Define positions for the bars - now with 4 model groups
positions = [0, 1, 3, 4, 6, 7, 9, 10]  # Positions with gaps between model groups
bar_width = 0.45

# Get data for each model
# dbmdz with gender-neutral professions
dbmdz_neutral_baseline_male = dbmdz_neutral_data[dbmdz_neutral_data['model'] == 'baseline']['male_perplexity'].values[0]
dbmdz_neutral_baseline_female = dbmdz_neutral_data[dbmdz_neutral_data['model'] == 'baseline']['female_perplexity'].values[0]
dbmdz_neutral_finetuned_male = dbmdz_neutral_data[dbmdz_neutral_data['model'] == 'finetuned']['male_perplexity'].values[0]
dbmdz_neutral_finetuned_female = dbmdz_neutral_data[dbmdz_neutral_data['model'] == 'finetuned']['female_perplexity'].values[0]

# google-bert with gender-neutral professions
google_neutral_baseline_male = google_neutral_data[google_neutral_data['model'] == 'baseline']['male_perplexity'].values[0]
google_neutral_baseline_female = google_neutral_data[google_neutral_data['model'] == 'baseline']['female_perplexity'].values[0]
google_neutral_finetuned_male = google_neutral_data[google_neutral_data['model'] == 'finetuned']['male_perplexity'].values[0]
google_neutral_finetuned_female = google_neutral_data[google_neutral_data['model'] == 'finetuned']['female_perplexity'].values[0]

# deepset-bert with gender-neutral professions
deepset_neutral_baseline_male = deepset_neutral_data[deepset_neutral_data['model'] == 'baseline']['male_perplexity'].values[0]
deepset_neutral_baseline_female = deepset_neutral_data[deepset_neutral_data['model'] == 'baseline']['female_perplexity'].values[0]
deepset_neutral_finetuned_male = deepset_neutral_data[deepset_neutral_data['model'] == 'finetuned']['male_perplexity'].values[0]
deepset_neutral_finetuned_female = deepset_neutral_data[deepset_neutral_data['model'] == 'finetuned']['female_perplexity'].values[0]

# distilbert with gender-neutral professions
distilbert_neutral_baseline_male = distilbert_neutral_data[distilbert_neutral_data['model'] == 'baseline']['male_perplexity'].values[0]
distilbert_neutral_baseline_female = distilbert_neutral_data[distilbert_neutral_data['model'] == 'baseline']['female_perplexity'].values[0]
distilbert_neutral_finetuned_male = distilbert_neutral_data[distilbert_neutral_data['model'] == 'finetuned']['male_perplexity'].values[0]
distilbert_neutral_finetuned_female = distilbert_neutral_data[distilbert_neutral_data['model'] == 'finetuned']['female_perplexity'].values[0]

# Create bars for each model and gender
bars = [
    # dbmdz with gender-neutral
    ax.bar(positions[0] - bar_width/2, dbmdz_neutral_baseline_male, bar_width, color='blue', label='Male'),
    ax.bar(positions[0] + bar_width/2, dbmdz_neutral_baseline_female, bar_width, color='orange', label='Female'),
    ax.bar(positions[1] - bar_width/2, dbmdz_neutral_finetuned_male, bar_width, color='blue'),
    ax.bar(positions[1] + bar_width/2, dbmdz_neutral_finetuned_female, bar_width, color='orange'),
    
    # google-bert with gender-neutral
    ax.bar(positions[2] - bar_width/2, google_neutral_baseline_male, bar_width, color='blue'),
    ax.bar(positions[2] + bar_width/2, google_neutral_baseline_female, bar_width, color='orange'),
    ax.bar(positions[3] - bar_width/2, google_neutral_finetuned_male, bar_width, color='blue'),
    ax.bar(positions[3] + bar_width/2, google_neutral_finetuned_female, bar_width, color='orange'),

    # deepset-bert with gender-neutral
    ax.bar(positions[4] - bar_width/2, deepset_neutral_baseline_male, bar_width, color='blue'),
    ax.bar(positions[4] + bar_width/2, deepset_neutral_baseline_female, bar_width, color='orange'),
    ax.bar(positions[5] - bar_width/2, deepset_neutral_finetuned_male, bar_width, color='blue'),
    ax.bar(positions[5] + bar_width/2, deepset_neutral_finetuned_female, bar_width, color='orange'),
    
    # distilbert with gender-neutral
    ax.bar(positions[6] - bar_width/2, distilbert_neutral_baseline_male, bar_width, color='blue'),
    ax.bar(positions[6] + bar_width/2, distilbert_neutral_baseline_female, bar_width, color='orange'),
    ax.bar(positions[7] - bar_width/2, distilbert_neutral_finetuned_male, bar_width, color='blue'),
    ax.bar(positions[7] + bar_width/2, distilbert_neutral_finetuned_female, bar_width, color='orange')
]

# Set custom x-ticks
ax.set_xticks(positions)
ax.set_xticklabels(['Baseline', 'Fine-tuned', 'Baseline', 'Fine-tuned', 'Baseline', 'Fine-tuned', 'Baseline', 'Fine-tuned'])

# Adjust bottom margin to be smaller
plt.subplots_adjust(bottom=0.15)

# Use a different approach to add model labels
trans = ax.get_xaxis_transform()
ax.text((positions[0] + positions[1])/2, -0.15, 'dbmdz-bert\ngender-inclusive', 
        transform=trans, ha='center', fontsize=11, fontweight='bold')
ax.text((positions[2] + positions[3])/2, -0.15, 'google-bert\ngender-inclusive', 
        transform=trans, ha='center', fontsize=11, fontweight='bold')
ax.text((positions[4] + positions[5])/2, -0.15, 'deepset-bert\ngender-inclusive', 
        transform=trans, ha='center', fontsize=11, fontweight='bold')
ax.text((positions[6] + positions[7])/2, -0.15, 'distilbert\ngender-inclusive', 
        transform=trans, ha='center', fontsize=11, fontweight='bold')

# Add title and axis labels
plt.title('Comparison of Male vs Female Perplexity Across German Models and Gender Types', fontsize=14)
plt.xlabel('', fontsize=12, labelpad=30)
plt.ylabel('Average Perplexity', fontsize=12)

# Function to add value labels
def add_value_label(bar, value):
    if value < 10:
        # For small bars, add a connector line and place label higher
        label_y = 20
        height = bar.get_height()
        x_position = bar.get_x() + bar.get_width()/2
        
        # Draw connector line
        ax.plot([x_position, x_position], [height, label_y*0.8], 
                color='black', linestyle='-', linewidth=0.5)
        
        # Add text
        ax.text(x_position, label_y, f'{value:.2f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        # For taller bars, place label just above the bar
        height = value
        x_position = bar.get_x() + bar.get_width()/2
        ax.text(x_position, height + 5, f'{value:.2f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add value labels to each bar
perplexity_values = [
    dbmdz_neutral_baseline_male, dbmdz_neutral_baseline_female,
    dbmdz_neutral_finetuned_male, dbmdz_neutral_finetuned_female,
    google_neutral_baseline_male, google_neutral_baseline_female,
    google_neutral_finetuned_male, google_neutral_finetuned_female,
    deepset_neutral_baseline_male, deepset_neutral_baseline_female,
    deepset_neutral_finetuned_male, deepset_neutral_finetuned_female,
    distilbert_neutral_baseline_male, distilbert_neutral_baseline_female,
    distilbert_neutral_finetuned_male, distilbert_neutral_finetuned_female
]

for bar, value in zip(bars, perplexity_values):
    add_value_label(bar[0], value)

# Add dividing lines between model groups
ax.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=8, color='gray', linestyle='--', alpha=0.5)

# Add a legend
ax.legend(loc='upper right')

# Remove 0.0 on x-axis
ax.tick_params(axis='x', which='both', length=0)
ax.tick_params(axis='y', which='both', left=True)

# Ensure directory exists
os.makedirs('../data/perplexity_measure/', exist_ok=True)

plt.tight_layout()
plt.savefig(f'../data/perplexity_measure/german_models_comparison_perplexity.png', dpi=300)
print("Saved perplexity comparison to ../data/perplexity_measure/german_models_comparison_perplexity.png")


print("\nCreating bias score distribution comparison for four models...")

# Create output directory if it doesn't exist
os.makedirs('../data/perplexity_measure/', exist_ok=True)

# Load the bias results files
try:
    # Load the averaged bias results for each model/data combination
    dbmdz_neutral_avg_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_gender_neutral/dbmdz/averaged_bias_results_gender_neutral_dbmdz_one_MASK.csv')
    google_neutral_avg_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_gender_neutral/google_bert/averaged_bias_results_gender_neutral_google_bert_one_MASK.csv')
    deepset_neutral_avg_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_gender_neutral/deepset_bert/averaged_bias_results_gender_neutral_deepset_bert.csv')
    distilbert_neutral_avg_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_gender_neutral/distilbert/averaged_bias_results_gender_neutral_distilbert_one_MASK.csv')
    
    print(f"Successfully loaded DbmdzBERT (gender-neutral) data with {len(dbmdz_neutral_avg_results)} rows")
    print(f"Successfully loaded Google BERT (gender-neutral) data with {len(google_neutral_avg_results)} rows")
    print(f"Successfully loaded DeepSet BERT (gender-neutral) data with {len(deepset_neutral_avg_results)} rows")
    print(f"Successfully loaded DistilBERT (gender-neutral) data with {len(distilbert_neutral_avg_results)} rows")
except Exception as e:
    print(f"Error loading CSV files: {e}")

# Create figure with four subplots in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
# Flatten the axes array for easier indexing
axes = axes.flatten()

# Function to create KDE plot for a given model's data on a given axis
def create_bias_kde(data, ax, model_name, profession_type):
    # Calculate unfiltered mean bias scores (including outliers)
    baseline_mean = data[data['model'] == 'baseline']['avg_bias_score'].mean()
    finetuned_mean = data[data['model'] == 'finetuned']['avg_bias_score'].mean()
    
    # Filter out extreme outliers for visualization only
    filtered_data = data[
        (data['avg_bias_score'] > -200) & 
        (data['avg_bias_score'] < 150)
    ]
    
    # Get data for each model
    baseline_data = filtered_data[filtered_data['model'] == 'baseline']
    finetuned_data = filtered_data[filtered_data['model'] == 'finetuned']
    
    # Create KDE plots
    try:
        sns.kdeplot(data=baseline_data, x='avg_bias_score', 
                    fill=True, alpha=0.7, linewidth=2, color='#000080', 
                    label='baseline', ax=ax, warn_singular=False)
    except Exception as e:
        print(f"Warning: Could not create KDE for baseline {model_name}: {e}")
        # Fallback to histogram if KDE fails
        ax.hist(baseline_data['avg_bias_score'], bins=20, alpha=0.7, 
                color='#000080', label='baseline', density=True)
    
    try:
        sns.kdeplot(data=finetuned_data, x='avg_bias_score',
                    fill=True, alpha=0.7, linewidth=2, color='#008080', 
                    label='finetuned', ax=ax, warn_singular=False)
    except Exception as e:
        print(f"Warning: Could not create KDE for finetuned {model_name}: {e}")
        # Fallback to histogram if KDE fails
        ax.hist(finetuned_data['avg_bias_score'], bins=20, alpha=0.7, 
                color='#008080', label='finetuned', density=True)
    
    # Add a vertical line at x=0 (no bias reference)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='No bias')
    
    # Add vertical dotted lines for the means (using ALL data, not just filtered)
    ax.axvline(x=baseline_mean, color='#000080', linestyle=':', linewidth=2, 
                label=f'Baseline avg: {baseline_mean:.2f}')
    ax.axvline(x=finetuned_mean, color='#008080', linestyle=':', linewidth=2, 
                label=f'Finetuned avg: {finetuned_mean:.2f}')
    

    # Set title as two lines for better readability with model name in bold
    ax.set_title(f'$\\mathbf{{{model_name}}}$\n$\\mathbf{{{profession_type}}}$', fontsize=14)
    ax.set_xlabel('Bias Score (Female-Male Perplexity Difference)', fontsize=12)
    ax.set_xlim(-150, 150)  # Set x-axis limits consistently across all plots
    ax.set_ylim(0.0, 0.2)    # Set y-axis limits consistently across all plots
    
    # Add legend with explicit labels - make it a bit smaller to fit
    ax.legend(title='Model', frameon=True, fontsize=9, loc='upper right')

# Create plots for each model and profession type
create_bias_kde(dbmdz_neutral_avg_results, axes[0], 'dbmdz-bert', 'gender-inclusive')
create_bias_kde(google_neutral_avg_results, axes[1], 'google-bert', 'gender-inclusive')
create_bias_kde(deepset_neutral_avg_results, axes[2], 'deepset-bert', 'gender-inclusive')
create_bias_kde(distilbert_neutral_avg_results, axes[3], 'distilbert', 'gender-inclusive')

# Add common y-label
fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=12)

# Add overall title
fig.suptitle('Comparison of Bias Score Distributions Across Models', fontsize=16)

plt.tight_layout()
# Adjust spacing
plt.subplots_adjust(top=0.9, wspace=0.1, hspace=0.3)
plt.savefig(f'../data/perplexity_measure/german_models_bias_distribution_comparison.png', dpi=300)

print("Saved bias distribution comparison to ../data/perplexity_measure/german_models_bias_distribution_comparison.png")



## FOR SINGLE MODEL ##

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import os

# print("\nCreating combined figure for bert model analysis...")

# # Ensure directory exists
# os.makedirs('../data/perplexity_measure/', exist_ok=True)

# typ = "english"

# model_name = "bert"

# # Load the CSV files for DBMDZ model only
# try:
#     # Load perplexity data
#     bert_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_{typ}/all_seeds_bias_{model_name}_results_one_MASK.csv')
    
#     # Load averaged bias results
#     bert_avg_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_{typ}/averaged_bias_results_{model_name}_one_MASK.csv')
    
#     print(f"Successfully loaded bert perplexity data with {len(bert_results)} rows")
#     print(f"Successfully loaded bert bias data with {len(bert_avg_results)} rows")
# except Exception as e:
#     print(f"Error loading CSV files: {e}")
#     exit(1)

# # Calculate average perplexity
# bert_data = bert_results.groupby(['model']).agg({
#     'male_perplexity': 'mean',
#     'female_perplexity': 'mean'
# }).reset_index()

# # Get data for perplexity comparison
# bert_baseline_male = bert_data[bert_data['model'] == 'baseline']['male_perplexity'].values[0]
# bert_baseline_female = bert_data[bert_data['model'] == 'baseline']['female_perplexity'].values[0]
# bert_finetuned_male = bert_data[bert_data['model'] == 'finetuned']['male_perplexity'].values[0]
# bert_finetuned_female = bert_data[bert_data['model'] == 'finetuned']['female_perplexity'].values[0]

# # Create a figure with two subplots side by side
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# # Left subplot - Perplexity Comparison
# # CHANGED: Reducing the distance between baseline and fine-tuned positions even more
# positions = [0, 0.25]  # Much closer spacing (was [0, 0.5])
# bar_width = 0.1  # Narrow bars

# # Create bars for perplexity comparison
# bars = [
#     # dbmdz with gender-neutral
#     ax1.bar(positions[0] - bar_width/2, bert_baseline_male, bar_width, color='blue', label='Male'),
#     ax1.bar(positions[0] + bar_width/2, bert_baseline_female, bar_width, color='orange', label='Female'),
#     ax1.bar(positions[1] - bar_width/2, bert_finetuned_male, bar_width, color='blue'),
#     ax1.bar(positions[1] + bar_width/2, bert_finetuned_female, bar_width, color='orange'),
# ]

# # Set custom x-ticks for perplexity plot
# ax1.set_xticks(positions)
# ax1.set_xticklabels(['Baseline', 'Fine-tuned'])

# # Adjust x-axis limits to center the bars better
# ax1.set_xlim(-0.2, 0.45)

# # Set y-axis limit to 500
# ax1.set_ylim(0, 600)

# # Add title and axis labels for perplexity plot
# ax1.set_title('Male vs Female Perplexity for BERT', fontsize=14, fontweight='bold', pad=20)
# ax1.set_xlabel('Model', fontsize=12)
# ax1.set_ylabel('Average Perplexity', fontsize=12)

# # Function to add value labels - modified to place them closer to bars
# def add_value_label(ax, bar, value):
#     height = bar.get_height()
#     x_position = bar.get_x() + bar.get_width()/2
    
#     # Place labels just above the bars (reduced offset)
#     y_offset = height * 0.03  # 3% of the height
#     if y_offset < 2:  # Minimum offset
#         y_offset = 2
    
#     ax.text(x_position, height + y_offset, f'{value:.2f}', 
#             ha='center', va='bottom', fontsize=10, fontweight='bold')

# # Add value labels to each bar
# perplexity_values = [
#     bert_baseline_male, bert_baseline_female,
#     bert_finetuned_male, bert_finetuned_female
# ]

# for bar, value in zip(bars, perplexity_values):
#     add_value_label(ax1, bar[0], value)

# # Add a legend for perplexity plot
# ax1.legend(loc='upper right')

# # Right subplot - Bias Score Distribution
# # Function to create KDE plot
# def create_bias_kde(data, ax, model_name):
#     # Calculate unfiltered mean bias scores (including outliers)
#     baseline_mean = data[data['model'] == 'baseline']['avg_bias_score'].mean()
#     finetuned_mean = data[data['model'] == 'finetuned']['avg_bias_score'].mean()
    
#     # Filter out extreme outliers for visualization only
#     filtered_data = data[
#         (data['avg_bias_score'] > -200) & 
#         (data['avg_bias_score'] < 150)
#     ]
    
#     # Get data for each model
#     baseline_data = filtered_data[filtered_data['model'] == 'baseline']
#     finetuned_data = filtered_data[filtered_data['model'] == 'finetuned']
    
#     # Create KDE plots
#     try:
#         sns.kdeplot(data=baseline_data, x='avg_bias_score', 
#                     fill=True, alpha=0.7, linewidth=2, color='#000080', 
#                     label='Baseline', ax=ax, warn_singular=False)
#     except Exception as e:
#         print(f"Warning: Could not create KDE for baseline {model_name}: {e}")
#         # Fallback to histogram if KDE fails
#         ax.hist(baseline_data['avg_bias_score'], bins=20, alpha=0.7, 
#                 color='#000080', label='Baseline', density=True)
    
#     try:
#         sns.kdeplot(data=finetuned_data, x='avg_bias_score',
#                     fill=True, alpha=0.7, linewidth=2, color='#008080', 
#                     label='Fine-tuned', ax=ax, warn_singular=False)
#     except Exception as e:
#         print(f"Warning: Could not create KDE for finetuned {model_name}: {e}")
#         # Fallback to histogram if KDE fails
#         ax.hist(finetuned_data['avg_bias_score'], bins=20, alpha=0.7, 
#                 color='#008080', label='Fine-tuned', density=True)
    
#     # Add a vertical line at x=0 (no bias reference)
#     ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='No bias')
    
#     # Add vertical dotted lines for the means (using ALL data, not just filtered)
#     ax.axvline(x=baseline_mean, color='#000080', linestyle=':', linewidth=2, 
#                 label=f'Baseline avg: {baseline_mean:.2f}')
#     ax.axvline(x=finetuned_mean, color='#008080', linestyle=':', linewidth=2, 
#                 label=f'Fine-tuned avg: {finetuned_mean:.2f}')
    
#     # Set title for bias plot with increased padding
#     ax.set_title(f'Bias Score Distribution for BERT', fontsize=14, fontweight='bold', pad=20)
#     ax.set_xlabel('Bias Score (Female-Male Perplexity Difference)', fontsize=12)
#     ax.set_ylabel('Density', fontsize=12)
#     ax.set_xlim(-150, 150)  # Set x-axis limits
#     ax.set_ylim(0.0, 0.2)  # Set x-axis limits
    
#     # Add legend
#     ax.legend(frameon=True, fontsize=10, loc='upper right')

# # Create bias distribution plot
# create_bias_kde(bert_avg_results, ax2, 'bert')

# # Add global title for the entire figure with increased spacing
# fig.suptitle('Perplexity-based Gender Bias Measurement', fontsize=16, y=0.98)

# # Adjust layout with increased spacing between main title and subplot titles
# plt.tight_layout()
# plt.subplots_adjust(top=0.85)  # Reduced from 0.88 to create more space

# # Save the figure
# output_path = f'../data/perplexity_measure/perplexity_measure_{typ}/bert_model_bias_perplexity.png'
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# print(f"Saved combined analysis to {output_path}")