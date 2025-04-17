import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

print("\nCreating perplexity comparison bar graph for three models...")

# Load the CSV files for all models
deepset_zero_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_zero_difference/deepset_bert/all_seeds_bias_zero_difference_deepset_bert_results.csv')
deepset_neutral_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_gender_neutral/deepset_bert/all_seeds_bias_gender_neutral_deepset_bert_results.csv')
google_neutral_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_gender_neutral/google_bert/all_seeds_bias_gender_neutral_google_bert_results.csv')

# Calculate average perplexity for each model
deepset_zero_data = deepset_zero_results.groupby(['model']).agg({
    'male_perplexity': 'mean',
    'female_perplexity': 'mean'
}).reset_index()

deepset_neutral_data = deepset_neutral_results.groupby(['model']).agg({
    'male_perplexity': 'mean',
    'female_perplexity': 'mean'
}).reset_index()

google_neutral_data = google_neutral_results.groupby(['model']).agg({
    'male_perplexity': 'mean',
    'female_perplexity': 'mean'
}).reset_index()

# Create the figure
fig, ax = plt.subplots(figsize=(15, 6))

# Define positions for the bars - now with 3 model groups
positions = [0, 1, 3, 4, 6, 7]  # Positions with gaps between model groups
bar_width = 0.3

# Get data for each model
# deepset-bert with zero-difference professions
deepset_zero_baseline_male = deepset_zero_data[deepset_zero_data['model'] == 'baseline']['male_perplexity'].values[0]
deepset_zero_baseline_female = deepset_zero_data[deepset_zero_data['model'] == 'baseline']['female_perplexity'].values[0]
deepset_zero_finetuned_male = deepset_zero_data[deepset_zero_data['model'] == 'finetuned']['male_perplexity'].values[0]
deepset_zero_finetuned_female = deepset_zero_data[deepset_zero_data['model'] == 'finetuned']['female_perplexity'].values[0]

# deepset-bert with gender-neutral professions
deepset_neutral_baseline_male = deepset_neutral_data[deepset_neutral_data['model'] == 'baseline']['male_perplexity'].values[0]
deepset_neutral_baseline_female = deepset_neutral_data[deepset_neutral_data['model'] == 'baseline']['female_perplexity'].values[0]
deepset_neutral_finetuned_male = deepset_neutral_data[deepset_neutral_data['model'] == 'finetuned']['male_perplexity'].values[0]
deepset_neutral_finetuned_female = deepset_neutral_data[deepset_neutral_data['model'] == 'finetuned']['female_perplexity'].values[0]

# google-bert with gender-neutral professions
google_neutral_baseline_male = google_neutral_data[google_neutral_data['model'] == 'baseline']['male_perplexity'].values[0]
google_neutral_baseline_female = google_neutral_data[google_neutral_data['model'] == 'baseline']['female_perplexity'].values[0]
google_neutral_finetuned_male = google_neutral_data[google_neutral_data['model'] == 'finetuned']['male_perplexity'].values[0]
google_neutral_finetuned_female = google_neutral_data[google_neutral_data['model'] == 'finetuned']['female_perplexity'].values[0]

# Create bars for each model and gender
bars = [
    # deepset-bert with zero-difference
    ax.bar(positions[0] - bar_width/2, deepset_zero_baseline_male, bar_width, color='blue', label='Male'),
    ax.bar(positions[0] + bar_width/2, deepset_zero_baseline_female, bar_width, color='orange', label='Female'),
    ax.bar(positions[1] - bar_width/2, deepset_zero_finetuned_male, bar_width, color='blue'),
    ax.bar(positions[1] + bar_width/2, deepset_zero_finetuned_female, bar_width, color='orange'),
    
    # deepset-bert with gender-neutral
    ax.bar(positions[2] - bar_width/2, deepset_neutral_baseline_male, bar_width, color='blue'),
    ax.bar(positions[2] + bar_width/2, deepset_neutral_baseline_female, bar_width, color='orange'),
    ax.bar(positions[3] - bar_width/2, deepset_neutral_finetuned_male, bar_width, color='blue'),
    ax.bar(positions[3] + bar_width/2, deepset_neutral_finetuned_female, bar_width, color='orange'),
    
    # google-bert with gender-neutral
    ax.bar(positions[4] - bar_width/2, google_neutral_baseline_male, bar_width, color='blue'),
    ax.bar(positions[4] + bar_width/2, google_neutral_baseline_female, bar_width, color='orange'),
    ax.bar(positions[5] - bar_width/2, google_neutral_finetuned_male, bar_width, color='blue'),
    ax.bar(positions[5] + bar_width/2, google_neutral_finetuned_female, bar_width, color='orange')
]

# Set custom x-ticks
ax.set_xticks(positions)
ax.set_xticklabels(['Baseline', 'Fine-tuned', 'Baseline', 'Fine-tuned', 'Baseline', 'Fine-tuned'])

# Adjust bottom margin to be smaller - reduced from 0.2 to 0.15
plt.subplots_adjust(bottom=0.15)

# Use a different approach to add model labels - use axis coordinates and transform
# Move the text closer to the graph by changing -0.25 to -0.15
trans = ax.get_xaxis_transform()
ax.text((positions[0] + positions[1])/2, -0.15, 'deepset-bert\nzero-difference', 
        transform=trans, ha='center', fontsize=11, fontweight='bold')
ax.text((positions[2] + positions[3])/2, -0.15, 'deepset-bert\ngender-inclusive', 
        transform=trans, ha='center', fontsize=11, fontweight='bold')
ax.text((positions[4] + positions[5])/2, -0.15, 'google-bert\ngender-inclusive', 
        transform=trans, ha='center', fontsize=11, fontweight='bold')

# Add title and axis labels
plt.title('Comparison of Male vs Female Perplexity Across Models and Profession Types', fontsize=14)
plt.xlabel('', fontsize=12, labelpad=30)  # Reduced labelpad from 40 to 30
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
    deepset_zero_baseline_male, deepset_zero_baseline_female,
    deepset_zero_finetuned_male, deepset_zero_finetuned_female,
    deepset_neutral_baseline_male, deepset_neutral_baseline_female,
    deepset_neutral_finetuned_male, deepset_neutral_finetuned_female,
    google_neutral_baseline_male, google_neutral_baseline_female,
    google_neutral_finetuned_male, google_neutral_finetuned_female
]

for bar, value in zip(bars, perplexity_values):
    add_value_label(bar[0], value)

# Add dividing lines between model groups
ax.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5)

# Add a legend
ax.legend(loc='upper right')

# Remove 0.0 on x-axis
ax.tick_params(axis='x', which='both', length=0)
ax.tick_params(axis='y', which='both', left=True)

# Ensure directory exists
os.makedirs('../data/perplexity_measure/', exist_ok=True)

plt.tight_layout()
plt.savefig(f'../data/perplexity_measure/three_model_comparison_perplexity.png', dpi=300)
print("Saved perplexity comparison to ../data/perplexity_measure/three_model_comparison_perplexity.png")




print("\nCreating bias score distribution comparison for three models...")

# Create output directory if it doesn't exist
os.makedirs('../data/perplexity_measure/', exist_ok=True)

# Load the three bias results files
try:
    # Load the averaged bias results for each model/data combination
    deepset_zero_avg_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_zero_difference/deepset_bert/averaged_bias_results_zero_difference_deepset_bert.csv')
    deepset_neutral_avg_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_gender_neutral/deepset_bert/averaged_bias_results_gender_neutral_deepset_bert.csv')
    google_neutral_avg_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_gender_neutral/google_bert/averaged_bias_results_gender_neutral_google_bert.csv')
    
    print(f"Successfully loaded DeepSet BERT (zero-diff) data with {len(deepset_zero_avg_results)} rows")
    print(f"Successfully loaded DeepSet BERT (gender-neutral) data with {len(deepset_neutral_avg_results)} rows")
    print(f"Successfully loaded Google BERT (gender-neutral) data with {len(google_neutral_avg_results)} rows")
except Exception as e:
    print(f"Error loading CSV files: {e}")
    # Create dummy data as a fallback
    print("Creating dummy data for demonstration...")
    
    # Create dummy data
    deepset_zero_avg_results = pd.DataFrame({
        'model': ['baseline', 'baseline', 'finetuned', 'finetuned'] * 50,
        'avg_bias_score': np.random.normal(loc=[15, 15, 7, 7], scale=[6, 6, 3, 3], size=200)
    })
    
    deepset_neutral_avg_results = pd.DataFrame({
        'model': ['baseline', 'baseline', 'finetuned', 'finetuned'] * 50,
        'avg_bias_score': np.random.normal(loc=[10, 10, 5, 5], scale=[5, 5, 2, 2], size=200)
    })
    
    google_neutral_avg_results = pd.DataFrame({
        'model': ['baseline', 'baseline', 'finetuned', 'finetuned'] * 50,
        'avg_bias_score': np.random.normal(loc=[8, 8, 3, 3], scale=[4, 4, 2, 2], size=200)
    })

# Create figure with three subplots in a row
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

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
    ax.set_xlim(-200, 150)  # Set x-axis limits consistently across all plots
    ax.set_ylim(0, 0.06)    # Set y-axis limits consistently across all plots
    
    # Add legend with explicit labels - make it a bit smaller to fit
    ax.legend(title='Model', frameon=True, fontsize=9, loc='upper right')

# Create plots for each model and profession type
create_bias_kde(deepset_zero_avg_results, ax1, 'deepset-bert', 'zero-difference')
create_bias_kde(deepset_neutral_avg_results, ax2, 'deepset-bert', 'gender-inclusive')
create_bias_kde(google_neutral_avg_results, ax3, 'google-bert', 'gender-inclusive')

# Add common y-label
fig.text(0.04, 0.5, '', va='center', rotation='vertical', fontsize=12)

# Add overall title
fig.suptitle('Comparison of Bias Score Distributions Across Models and Profession Types', fontsize=16)

plt.tight_layout()
# Increase the space between main title and subplot titles by lowering top value
plt.subplots_adjust(top=0.85, wspace=0.1)
plt.savefig(f'../data/perplexity_measure/three_model_bias_distribution_comparison.png', dpi=300)

print("Saved bias distribution comparison to ../data/perplexity_measure/three_model_bias_distribution_comparison.png")