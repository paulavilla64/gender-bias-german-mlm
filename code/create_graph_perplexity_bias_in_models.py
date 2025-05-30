import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os



################### FOR ENGLISH BERT ###############


print("\nCreating figures by profession gender group for bert model analysis...")

# Ensure directory exists
os.makedirs('../data/perplexity_measure/', exist_ok=True)

typ = "english"
model_name = "bert"

# Load the CSV files
try:
    # Load perplexity data

    bert_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_{typ}/{model_name}/all_seeds_bias_{model_name}_{typ}_results_one_MASK_groups.csv')
    # Load averaged bias results
    bert_avg_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_{typ}/{model_name}/averaged_bias_results_{model_name}_{typ}_one_MASK_groups.csv')
    
    # Check if prof_gender column exists
    if 'prof_gender' not in bert_results.columns:
        print("Error: 'prof_gender' column not found in the dataset!")
        exit(1)
        
    print(f"Successfully loaded bert perplexity data with {len(bert_results)} rows")
    print(f"Successfully loaded bert bias data with {len(bert_avg_results)} rows")
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit(1)

# Define the profession gender groups in the desired order
prof_gender_groups = ['female', 'male', 'balanced']

# Verify groups exist in the data
existing_groups = bert_results['prof_gender'].unique()
print(f"Profession gender groups found in data: {existing_groups}")

# Filter for available groups
prof_gender_groups = [group for group in prof_gender_groups if group in existing_groups]

# Prepare a custom figure for each profession gender group
for prof_gender in prof_gender_groups:
    print(f"\nProcessing profession gender group: {prof_gender}")
    
    # Filter the data for this profession gender group
    current_results = bert_results[bert_results['prof_gender'] == prof_gender]
    current_avg_results = bert_avg_results[bert_avg_results['prof_gender'] == prof_gender]
    
    # Calculate average perplexity for this profession gender group
    bert_data = current_results.groupby(['model']).agg({
        'male_perplexity': 'mean',
        'female_perplexity': 'mean'
    }).reset_index()
    
    # Check if we have both baseline and finetuned models in the filtered data
    if len(bert_data) < 2:
        print(f"Warning: Not enough data for profession gender group '{prof_gender}'. Skipping...")
        continue
    
    # Get data for perplexity comparison
    bert_baseline_male = bert_data[bert_data['model'] == 'baseline']['male_perplexity'].values[0]
    bert_baseline_female = bert_data[bert_data['model'] == 'baseline']['female_perplexity'].values[0]
    bert_finetuned_male = bert_data[bert_data['model'] == 'finetuned']['male_perplexity'].values[0]
    bert_finetuned_female = bert_data[bert_data['model'] == 'finetuned']['female_perplexity'].values[0]
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left subplot - Perplexity Comparison
    positions = [0, 0.25]  # Close spacing
    bar_width = 0.1  # Narrow bars
    
    # Create bars for perplexity comparison
    bars = [
        ax1.bar(positions[0] - bar_width/2, bert_baseline_male, bar_width, color='blue', label='Male'),
        ax1.bar(positions[0] + bar_width/2, bert_baseline_female, bar_width, color='orange', label='Female'),
        ax1.bar(positions[1] - bar_width/2, bert_finetuned_male, bar_width, color='blue'),
        ax1.bar(positions[1] + bar_width/2, bert_finetuned_female, bar_width, color='orange'),
    ]
    
    # Set custom x-ticks for perplexity plot
    ax1.set_xticks(positions)
    ax1.set_xticklabels(['Baseline', 'Fine-tuned'])
    
    # Adjust x-axis limits to center the bars better
    ax1.set_xlim(-0.2, 0.45)
    
    # Set y-axis limit based on data
    y_max = max(bert_baseline_male, bert_baseline_female, 
                bert_finetuned_male, bert_finetuned_female) * 1.15  # 15% headroom
    ax1.set_ylim(0, 100)  # Cap at 600 for consistency
    
    # Format profession gender group for title
    prof_group_title = {
        'male': 'Male-dominated Professions',
        'female': 'Female-dominated Professions', 
        'balanced': 'Gender-balanced Professions'
    }.get(prof_gender, prof_gender.capitalize())
    
    # Add title and axis labels for perplexity plot
    ax1.set_title(f'Male vs Female Perplexity: {prof_group_title}', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Average Perplexity', fontsize=12)
    
    # Function to add value labels
    def add_value_label(ax, bar, value):
        height = bar.get_height()
        x_position = bar.get_x() + bar.get_width()/2
        
        # Place labels just above the bars (reduced offset)
        y_offset = height * 0.03  # 3% of the height
        if y_offset < 2:  # Minimum offset
            y_offset = 2
        
        ax.text(x_position, height + y_offset, f'{value:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add value labels to each bar
    perplexity_values = [
        bert_baseline_male, bert_baseline_female,
        bert_finetuned_male, bert_finetuned_female
    ]
    
    for bar, value in zip(bars, perplexity_values):
        add_value_label(ax1, bar[0], value)
    
    # Add a legend for perplexity plot
    ax1.legend(loc='upper right')
    
    # Right subplot - Bias Score Distribution
    # Function to create KDE plot
    def create_bias_kde(data, ax, model_name, group_name):
        # Check if we have enough data
        if len(data) < 5:
            ax.text(0.5, 0.5, f"Insufficient data for KDE plot\n({len(data)} data points)",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
            
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
        
        # Create KDE plots or histograms based on data size
        if len(baseline_data) >= 5:
            try:
                sns.kdeplot(data=baseline_data, x='avg_bias_score', 
                            fill=True, alpha=0.7, linewidth=2, color='#000080', 
                            label='Baseline', ax=ax, warn_singular=False)
            except Exception as e:
                print(f"Warning: Could not create KDE for baseline {group_name}: {e}")
                # Fallback to histogram
                ax.hist(baseline_data['avg_bias_score'], bins=10, alpha=0.7, 
                        color='#000080', label='Baseline', density=True)
        else:
            # Use scatter plot with jitter for very small datasets
            y_jitter = np.random.normal(0, 0.01, size=len(baseline_data))
            ax.scatter(baseline_data['avg_bias_score'], y_jitter + 0.05, 
                       alpha=0.8, s=50, color='#000080', label='Baseline')
        
        if len(finetuned_data) >= 5:
            try:
                sns.kdeplot(data=finetuned_data, x='avg_bias_score',
                            fill=True, alpha=0.7, linewidth=2, color='#008080', 
                            label='Fine-tuned', ax=ax, warn_singular=False)
            except Exception as e:
                print(f"Warning: Could not create KDE for finetuned {group_name}: {e}")
                # Fallback to histogram
                ax.hist(finetuned_data['avg_bias_score'], bins=10, alpha=0.7, 
                        color='#008080', label='Fine-tuned', density=True)
        else:
            # Use scatter plot with jitter for very small datasets
            y_jitter = np.random.normal(0, 0.01, size=len(finetuned_data))
            ax.scatter(finetuned_data['avg_bias_score'], y_jitter + 0.1, 
                       alpha=0.8, s=50, color='#008080', label='Fine-tuned')
        
        # Add a vertical line at x=0 (no bias reference)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='No bias')
        
        # Add vertical dotted lines for the means (using ALL data, not just filtered)
        ax.axvline(x=baseline_mean, color='#000080', linestyle=':', linewidth=2, 
                    label=f'Baseline avg: {baseline_mean:.2f}')
        ax.axvline(x=finetuned_mean, color='#008080', linestyle=':', linewidth=2, 
                    label=f'Fine-tuned avg: {finetuned_mean:.2f}')
        
        # Set title for bias plot with increased padding
        ax.set_title(f'Bias Score Distribution: {group_name}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Bias Score (Female-Male Perplexity Difference)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_xlim(-150, 150)  # Set x-axis limits
        
        # Only set ylim if we have actual density plots
        if len(baseline_data) >= 5 or len(finetuned_data) >= 5:
            ax.set_ylim(0.0, 0.2)  # Set y-axis limits
        
        # Add legend
        ax.legend(frameon=True, fontsize=10, loc='upper right')
    
    # Create bias distribution plot
    create_bias_kde(current_avg_results, ax2, {model_name}, prof_group_title)
    
    # Add global title for the entire figure with increased spacing
    fig.suptitle(f'Perplexity-based Gender Bias Measurement for BERT - {prof_group_title}', fontsize=16, y=0.98)
    
    # Adjust layout with increased spacing between main title and subplot titles
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the figure
    output_path = f'../data/perplexity_measure/perplexity_measure_{typ}/{model_name}/{model_name}_{typ}_model_bias_perplexity_{prof_gender}_group.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved analysis for {prof_gender} profession group to {output_path}")
    plt.close()

# Now create a combined figure with all profession groups for comparison
print("\nCreating combined comparison figure for all profession gender groups...")

# Create a figure with multiple subplots (3x2 grid - one row per profession group)
fig, axes = plt.subplots(3, 2, figsize=(18, 16))  # Slightly taller figure for better spacing

# Process each profession gender group
for i, prof_gender in enumerate(prof_gender_groups):
    print(f"Adding {prof_gender} group to combined figure...")
    
    # Filter the data for this profession gender group
    current_results = bert_results[bert_results['prof_gender'] == prof_gender]
    current_avg_results = bert_avg_results[bert_avg_results['prof_gender'] == prof_gender]
    
    # Calculate average perplexity for this profession gender group
    bert_data = current_results.groupby(['model']).agg({
        'male_perplexity': 'mean',
        'female_perplexity': 'mean'
    }).reset_index()
    
    # Check if we have both baseline and finetuned models in the filtered data
    if len(bert_data) < 2:
        axes[i, 0].text(0.5, 0.5, f"Insufficient data for {prof_gender} group",
                      ha='center', va='center', transform=axes[i, 0].transAxes, fontsize=12)
        axes[i, 1].text(0.5, 0.5, f"Insufficient data for {prof_gender} group",
                      ha='center', va='center', transform=axes[i, 1].transAxes, fontsize=12)
        continue
    
    # Get data for perplexity comparison
    bert_baseline_male = bert_data[bert_data['model'] == 'baseline']['male_perplexity'].values[0]
    bert_baseline_female = bert_data[bert_data['model'] == 'baseline']['female_perplexity'].values[0]
    bert_finetuned_male = bert_data[bert_data['model'] == 'finetuned']['male_perplexity'].values[0]
    bert_finetuned_female = bert_data[bert_data['model'] == 'finetuned']['female_perplexity'].values[0]
    
    # Format profession gender group for title
    prof_group_title = {
        'male': 'Male-dominated Professions',
        'female': 'Female-dominated Professions', 
        'balanced': 'Gender-balanced Professions'
    }.get(prof_gender, prof_gender.capitalize())
    
    # Left subplot - Perplexity Comparison
    positions = [0, 0.25]  # Close spacing
    bar_width = 0.1  # Narrow bars
    
    # Create bars for perplexity comparison
    bars = [
        axes[i, 0].bar(positions[0] - bar_width/2, bert_baseline_male, bar_width, color='blue', label='Male'),
        axes[i, 0].bar(positions[0] + bar_width/2, bert_baseline_female, bar_width, color='orange', label='Female'),
        axes[i, 0].bar(positions[1] - bar_width/2, bert_finetuned_male, bar_width, color='blue'),
        axes[i, 0].bar(positions[1] + bar_width/2, bert_finetuned_female, bar_width, color='orange'),
    ]
    
    # Set custom x-ticks for perplexity plot
    axes[i, 0].set_xticks(positions)
    axes[i, 0].set_xticklabels(['Baseline', 'Fine-tuned'])
    
    # Adjust x-axis limits to center the bars better
    axes[i, 0].set_xlim(-0.2, 0.45)
    
    # Set y-axis limit based on data
    y_max = max(bert_baseline_male, bert_baseline_female, 
                bert_finetuned_male, bert_finetuned_female) * 1.15  # 15% headroom
    axes[i, 0].set_ylim(0, 100)  # Cap at 600 for consistency
    
    # Add title and axis labels for perplexity plot
    axes[i, 0].set_title(f'Male vs Female Perplexity: {prof_group_title}', fontsize=14, fontweight='bold', pad=20)
    axes[i, 0].set_xlabel('Model', fontsize=12)
    axes[i, 0].set_ylabel('Average Perplexity', fontsize=12)
    
    # Function to add value labels
    def add_value_label(ax, bar, value):
        height = bar.get_height()
        x_position = bar.get_x() + bar.get_width()/2
        
        # Place labels just above the bars (reduced offset)
        y_offset = height * 0.03  # 3% of the height
        if y_offset < 2:  # Minimum offset
            y_offset = 2
        
        ax.text(x_position, height + y_offset, f'{value:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add value labels to each bar
    perplexity_values = [
        bert_baseline_male, bert_baseline_female,
        bert_finetuned_male, bert_finetuned_female
    ]
    
    for bar, value in zip(bars, perplexity_values):
        add_value_label(axes[i, 0], bar[0], value)
    
    # Add a legend for perplexity plot
    axes[i, 0].legend(loc='upper right')
    
    # Right subplot - Bias Score Distribution
    # Create bias KDE plot
    baseline_mean = current_avg_results[current_avg_results['model'] == 'baseline']['avg_bias_score'].mean()
    finetuned_mean = current_avg_results[current_avg_results['model'] == 'finetuned']['avg_bias_score'].mean()
    
    # Filter out extreme outliers for visualization only
    filtered_data = current_avg_results[
        (current_avg_results['avg_bias_score'] > -200) & 
        (current_avg_results['avg_bias_score'] < 150)
    ]
    
    # Get data for each model
    baseline_data = filtered_data[filtered_data['model'] == 'baseline']
    finetuned_data = filtered_data[filtered_data['model'] == 'finetuned']
    
    if len(baseline_data) >= 5 and len(finetuned_data) >= 5:
        try:
            sns.kdeplot(data=baseline_data, x='avg_bias_score', 
                        fill=True, alpha=0.7, linewidth=2, color='#000080', 
                        label='Baseline', ax=axes[i, 1], warn_singular=False)
        except Exception as e:
            print(f"Warning: Could not create KDE for baseline {prof_gender}: {e}")
            # Fallback to histogram
            axes[i, 1].hist(baseline_data['avg_bias_score'], bins=10, alpha=0.7, 
                    color='#000080', label='Baseline', density=True)
            
        try:
            sns.kdeplot(data=finetuned_data, x='avg_bias_score',
                        fill=True, alpha=0.7, linewidth=2, color='#008080', 
                        label='Fine-tuned', ax=axes[i, 1], warn_singular=False)
        except Exception as e:
            print(f"Warning: Could not create KDE for finetuned {prof_gender}: {e}")
            # Fallback to histogram
            axes[i, 1].hist(finetuned_data['avg_bias_score'], bins=10, alpha=0.7, 
                    color='#008080', label='Fine-tuned', density=True)
    else:
        # Use scatter plot with jitter for very small datasets
        if len(baseline_data) > 0:
            y_jitter = np.random.normal(0, 0.01, size=len(baseline_data))
            axes[i, 1].scatter(baseline_data['avg_bias_score'], y_jitter + 0.05, 
                       alpha=0.8, s=50, color='#000080', label='Baseline')
                       
        if len(finetuned_data) > 0:
            y_jitter = np.random.normal(0, 0.01, size=len(finetuned_data))
            axes[i, 1].scatter(finetuned_data['avg_bias_score'], y_jitter + 0.1, 
                       alpha=0.8, s=50, color='#008080', label='Fine-tuned')
    
    # Add a vertical line at x=0 (no bias reference)
    axes[i, 1].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='No bias')
    
    # Add vertical dotted lines for the means
    if not np.isnan(baseline_mean):
        axes[i, 1].axvline(x=baseline_mean, color='#000080', linestyle=':', linewidth=2, 
                        label=f'Baseline avg: {baseline_mean:.2f}')
    if not np.isnan(finetuned_mean):
        axes[i, 1].axvline(x=finetuned_mean, color='#008080', linestyle=':', linewidth=2, 
                        label=f'Fine-tuned avg: {finetuned_mean:.2f}')
    
    # Set title for bias plot with increased padding
    axes[i, 1].set_title(f'Bias Score Distribution: {prof_group_title}', fontsize=14, fontweight='bold', pad=20)
    axes[i, 1].set_xlabel('Bias Score (Female-Male Perplexity Difference)', fontsize=12)
    axes[i, 1].set_ylabel('Density', fontsize=12)
    axes[i, 1].set_xlim(-150, 150)  # Set x-axis limits
    
    # Only set ylim if we have actual density plots
    if len(baseline_data) >= 5 or len(finetuned_data) >= 5:
        axes[i, 1].set_ylim(0.0, 0.2)  # Set y-axis limits
    
    # Add legend
    axes[i, 1].legend(frameon=True, fontsize=10, loc='upper right')

# Add global title for the entire figure
fig.suptitle('Perplexity-based Gender Bias Measurement for BERT by Profession Category', fontsize=18, y=0.99)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.93, hspace=0.5)  # Increased top margin and vertical spacing

# Save the combined figure
output_path = f'../data/perplexity_measure/perplexity_measure_{typ}/{model_name}/{model_name}_{typ}_model_bias_perplexity_all_groups_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved combined comparison figure to {output_path}")

print("\nAnalysis complete!")





#################### BASELINE MEASUREMENT #######################

#################### SINGLE MODEL ###############################

## FOR PROFESSION GENDER GROUPS ##


print("\nCreating figures by profession gender group for baseline DBMDZ BERT model analysis...")

# Ensure directory exists
os.makedirs('../data/perplexity_measure/', exist_ok=True)

typ = "normal"
model_name = "dbmdz"

# Load the CSV files
try:
    # Load perplexity data
    bert_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_{typ}/{model_name}/all_seeds_bias_{model_name}_{typ}_results_one_MASK_groups.csv')
    # Load averaged bias results
    bert_avg_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_{typ}/{model_name}/averaged_bias_results_{model_name}_{typ}_one_MASK_groups.csv')
    
    # Check if prof_gender column exists
    if 'prof_gender' not in bert_results.columns:
        print("Error: 'prof_gender' column not found in the dataset!")
        exit(1)
        
    # Filter to only include baseline model data
    bert_results = bert_results[bert_results['model'] == 'baseline']
    bert_avg_results = bert_avg_results[bert_avg_results['model'] == 'baseline']
        
    print(f"Successfully loaded bert baseline perplexity data with {len(bert_results)} rows")
    print(f"Successfully loaded bert baseline bias data with {len(bert_avg_results)} rows")
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit(1)

# Define the profession gender groups in the desired order
prof_gender_groups = ['female', 'male', 'balanced']

# Verify groups exist in the data
existing_groups = bert_results['prof_gender'].unique()
print(f"Profession gender groups found in data: {existing_groups}")

# Filter for available groups
prof_gender_groups = [group for group in prof_gender_groups if group in existing_groups]

# Create a figure with subplots in a single row - more compact layout
fig, axes = plt.subplots(1, 3, figsize=(12, 5))  # Wider than tall, 3 columns

# Process each profession gender group
for i, prof_gender in enumerate(prof_gender_groups):
    print(f"Adding {prof_gender} group to figure...")
    
    # Filter the data for this profession gender group
    current_results = bert_results[bert_results['prof_gender'] == prof_gender]
    
    # Calculate average perplexity for this profession gender group
    bert_data = current_results.groupby(['model']).agg({
        'male_perplexity': 'mean',
        'female_perplexity': 'mean'
    }).reset_index()
    
    # Check if we have data for baseline model
    if len(bert_data) == 0:
        axes[i].text(0.5, 0.5, f"Insufficient data for {prof_gender} group",
                      ha='center', va='center', transform=axes[i].transAxes, fontsize=10)
        continue
    
    # Get data for perplexity comparison
    bert_baseline_male = bert_data['male_perplexity'].values[0]
    bert_baseline_female = bert_data['female_perplexity'].values[0]
    
    # Format profession gender group for title
    prof_group_title = {
        'male': 'Male-dominated',
        'female': 'Female-dominated', 
        'balanced': 'Gender-balanced'
    }.get(prof_gender, prof_gender.capitalize())
    
    # Bar chart for baseline perplexity comparison
    x_position = 0  # Single position for "Baseline"
    bar_width = 0.35  # Width for grouped bars
    
    # Create bars for perplexity comparison - store the actual bar objects
    # Use gender-specific colors with labels for the legend
    male_bar = axes[i].bar(x_position - bar_width/2, bert_baseline_male, bar_width, 
                          color='blue', label='Male' if i == 0 else "")
    female_bar = axes[i].bar(x_position + bar_width/2, bert_baseline_female, bar_width, 
                            color='orange', label='Female' if i == 0 else "")
    
    # Set custom x-ticks 
    axes[i].set_xticks([x_position])
    axes[i].set_xticklabels(['Baseline'])
    
    # Adjust x-axis limits
    axes[i].set_xlim(-0.6, 0.6)
    
    # Set y-axis limit to 100
    axes[i].set_ylim(0, 500)
    
    # Calculate bias score
    bias_score = bert_baseline_female - bert_baseline_male
    bias_status = "against women" if bias_score > 0 else "against men" if bias_score < 0 else "neutral"
    
    # Set title and axis labels with smaller font
    axes[i].set_title(f'{prof_group_title} Professions', fontsize=12, pad=10, fontweight='bold')
    
    # Only add ylabel for the first subplot to save space
    if i == 0:
        axes[i].set_ylabel('Average Perplexity', fontsize=11)
    
    # Add a text annotation for the bias score positioned above the bar values
    # Calculate the position based on the higher bar value
    max_bar_value = max(bert_baseline_male, bert_baseline_female)
    text_y_position = (max_bar_value + 15) / 100  # Convert to axes coordinates (0-1 scale), moved up more
    axes[i].text(0.7, text_y_position, f'Bias Score: {bias_score:.2f}\n({bias_status})', 
             ha='right', va='bottom', transform=axes[i].transAxes, 
             fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    # Add value labels directly
    # For male bar
    axes[i].text(x_position - bar_width/2, bert_baseline_male + 1, f'{bert_baseline_male:.2f}', 
            ha='center', va='bottom', fontsize=9)
    
    # For female bar
    axes[i].text(x_position + bar_width/2, bert_baseline_female + 1, f'{bert_baseline_female:.2f}', 
            ha='center', va='bottom', fontsize=9)

# Add global title for the entire figure
fig.suptitle('Perplexity-based Gender Bias Measurement for Baseline DBMDZ BERT Regular', fontsize=14, y=1.0)

# Add legend only to the first subplot, positioned lower right
axes[0].legend(loc='upper right')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(wspace=0.25)  # Add some space between subplots

# Save the figure
output_path = f'../data/perplexity_measure/perplexity_measure_{typ}/{model_name}/{model_name}_{typ}_baseline_perplexity_all_groups_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved baseline comparison figure to {output_path}")

print("\nAnalysis complete!")




############## BASELINE ################################


############## MULTIPLE MODELS #########################


print("\nCreating figures for baseline gender-inclusive perplexity across multiple models...")

# Ensure directory exists
os.makedirs('../data/perplexity_measure/combined', exist_ok=True)

typ = "gender_neutral"
model_names = ["dbmdz", "google-bert", "deepset-bert", "distilbert"]
display_names = ["DBMDZ BERT", "GOOGLE BERT", "G-BERT", "DISTILBERT"]

# Load the CSV files for all models
model_results = {}
model_avg_results = {}

try:
    for model_name in model_names:
        # Load perplexity data
        results_path = f'../data/perplexity_measure/Lou/perplexity_measure_{typ}/{model_name}/test_all.csv'
        avg_path = f'../data/perplexity_measure/Lou/perplexity_measure_{typ}/{model_name}/test_averaged.csv'
        
        model_results[model_name] = pd.read_csv(results_path)
        model_avg_results[model_name] = pd.read_csv(avg_path)
        
        # Check if prof_gender column exists
        if 'prof_gender' not in model_results[model_name].columns:
            print(f"Error: 'prof_gender' column not found in the dataset for {model_name}!")
            continue
            
        # Filter to only include baseline model data
        model_results[model_name] = model_results[model_name][model_results[model_name]['model'] == 'baseline']
        model_avg_results[model_name] = model_avg_results[model_name][model_avg_results[model_name]['model'] == 'baseline']
            
        print(f"Successfully loaded {model_name} baseline perplexity data with {len(model_results[model_name])} rows")
        
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit(1)

# Define the profession gender groups in the desired order
prof_gender_groups = ['female', 'male', 'balanced']

# Create a figure with subplots in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()  # Flatten to make indexing easier

# Process each model
for i, (model_name, display_name) in enumerate(zip(model_names, display_names)):
    if model_name not in model_results:
        print(f"Skipping {model_name} - data not loaded")
        continue
        
    results = model_results[model_name]
    print(f"Processing model: {display_name}")
    
    # Define y-limit based on model
    if display_name == "GOOGLE BERT":
        y_limit = 750
    elif display_name == "G-BERT":
        y_limit = 100
    else:
        y_limit = 100
    
    # Create separate subplots for each profession group
    grouped_data = []
    
    # First collect data for all profession groups
    for prof_gender in prof_gender_groups:
        # Filter the data for this profession gender group
        current_results = results[results['prof_gender'] == prof_gender]
        
        # Calculate average perplexity for this profession gender group
        agg_data = current_results.groupby(['model']).agg({
            'male_perplexity': 'mean',
            'female_perplexity': 'mean'
        }).reset_index()
        
        if len(agg_data) > 0:
            # Get data for perplexity comparison
            male_perp = agg_data['male_perplexity'].values[0]
            female_perp = agg_data['female_perplexity'].values[0]
            bias_score = female_perp - male_perp
            
            # Format profession gender group
            prof_group_title = {
                'male': 'Male-dominated',
                'female': 'Female-dominated', 
                'balanced': 'Gender-balanced'
            }.get(prof_gender, prof_gender.capitalize())
            
            grouped_data.append((prof_gender, prof_group_title, male_perp, female_perp, bias_score))
    
    # If no data, continue to next model
    if not grouped_data:
        axes[i].text(0.5, 0.5, f"No data for {display_name}", 
                    ha='center', va='center', transform=axes[i].transAxes, fontsize=12)
        continue
    
    # Set up the bar positions for all groups
    bar_width = 0.15
    group_spacing = 0.5
    num_groups = len(grouped_data)
    total_width = num_groups * (2 * bar_width + group_spacing) - group_spacing
    start_pos = -total_width / 2
    
    # Set up x-axis ticks and labels
    xticks = []
    xticklabels = []
    
    # Process each profession group for this model
    for j, (prof_gender, prof_group_title, male_perp, female_perp, bias_score) in enumerate(grouped_data):
        # Calculate positions for this group
        male_pos = start_pos + j * (2 * bar_width + group_spacing)
        female_pos = male_pos + bar_width
        
        # Calculate center point between bars for label
        group_center = male_pos + bar_width/2 + (female_pos - (male_pos + bar_width))/2
        
        # Add to xticks and labels - centered between the bars
        xticks.append(group_center)
        xticklabels.append(prof_gender[0].upper())  # Just use first letter (F, M, B)
        
        # Plot the bars
        male_bar = axes[i].bar(male_pos, male_perp, bar_width, color='blue')
        female_bar = axes[i].bar(female_pos, female_perp, bar_width, color='orange')
        
        # Add value labels - centered directly above each bar
        axes[i].text(male_pos, male_perp + y_limit*0.02, f'{male_perp:.2f}', 
                ha='center', va='bottom', fontsize=8)
        axes[i].text(female_pos, female_perp + y_limit*0.02, f'{female_perp:.2f}', 
                ha='center', va='bottom', fontsize=8)
        
        # Determine position for bias score text
        bias_status = "against women" if bias_score > 0 else "against men" if bias_score < 0 else "neutral"
        
        if display_name == "GOOGLE BERT" and max(male_perp, female_perp) > y_limit*0.8:
            # For Google BERT with very high perplexity
            text_y_pos = y_limit*0.30  # Position at 30% of y-axis height (raised from 15%)
        else:
            # Calculate position based on perplexity value heights
            # Leave more space between perplexity values and bias score box
            text_y_pos = max(male_perp, female_perp) + y_limit*0.10  # 10% of y-limit above highest bar
            
        axes[i].text(group_center, text_y_pos, 
                 f'Bias Score: {bias_score:.2f}\n({bias_status})', 
                 ha='center', va='bottom', fontsize=8,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
    
    # Set up the axes
    axes[i].set_xticks(xticks)
    axes[i].set_xticklabels(xticklabels)
    
    # Set custom y-limits based on model
    axes[i].set_ylim(0, y_limit)
    
    # Add title and labels
    axes[i].set_title(f'{display_name}', fontsize=13, pad=10)
    
    # Add legend to the first subplot only
    if i == 0:
        axes[i].bar(0, 0, color='blue', label='Male')
        axes[i].bar(0, 0, color='orange', label='Female')
        axes[i].legend(loc='upper right', fontsize=10)
    
    # Add ylabel to the left subplots
    if i % 2 == 0:
        axes[i].set_ylabel('Average Perplexity', fontsize=11)

# Add a global title
fig.suptitle('Perplexity-based Gender Bias Measurement for Baseline German Models (Gender-inclusive)', 
             fontsize=16, y=0.96)

# Adjust layout
plt.subplots_adjust(top=0.90, bottom=0.10, wspace=0.2, hspace=0.25)

# Save the figure
output_path = f'../data/perplexity_measure/Lou/perplexity_measure_{typ}/multi_model_gender_neutral_baseline_perplexity_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved multi-model comparison figure to {output_path}")

print("\nAnalysis complete!")




################# BASELINE VS. FINETUNED ###################

################ FOR SINGLE MODEL ###################



print("\nCreating combined figure for bert model analysis...")

# Ensure directory exists
os.makedirs('../data/perplexity_measure/', exist_ok=True)

typ = "zero_difference"

model_name = "dbmdz"

# Load the CSV files for DBMDZ model only
try:
    # Load perplexity data
    bert_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_{typ}/all_seeds_bias_{model_name}_results_one_MASK_groups.csv')
    
    # Load averaged bias results
    bert_avg_results = pd.read_csv(f'../data/perplexity_measure/perplexity_measure_{typ}/averaged_bias_results_{model_name}_one_MASK_groups.csv')
    
    print(f"Successfully loaded bert perplexity data with {len(bert_results)} rows")
    print(f"Successfully loaded bert bias data with {len(bert_avg_results)} rows")
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit(1)

# Calculate average perplexity
bert_data = bert_results.groupby(['model']).agg({
    'male_perplexity': 'mean',
    'female_perplexity': 'mean'
}).reset_index()

# Get data for perplexity comparison
bert_baseline_male = bert_data[bert_data['model'] == 'baseline']['male_perplexity'].values[0]
bert_baseline_female = bert_data[bert_data['model'] == 'baseline']['female_perplexity'].values[0]
bert_finetuned_male = bert_data[bert_data['model'] == 'finetuned']['male_perplexity'].values[0]
bert_finetuned_female = bert_data[bert_data['model'] == 'finetuned']['female_perplexity'].values[0]

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left subplot - Perplexity Comparison
# CHANGED: Reducing the distance between baseline and fine-tuned positions even more
positions = [0, 0.25]  # Much closer spacing (was [0, 0.5])
bar_width = 0.1  # Narrow bars

# Create bars for perplexity comparison
bars = [
    # dbmdz with gender-neutral
    ax1.bar(positions[0] - bar_width/2, bert_baseline_male, bar_width, color='blue', label='Male'),
    ax1.bar(positions[0] + bar_width/2, bert_baseline_female, bar_width, color='orange', label='Female'),
    ax1.bar(positions[1] - bar_width/2, bert_finetuned_male, bar_width, color='blue'),
    ax1.bar(positions[1] + bar_width/2, bert_finetuned_female, bar_width, color='orange'),
]

# Set custom x-ticks for perplexity plot
ax1.set_xticks(positions)
ax1.set_xticklabels(['Baseline', 'Fine-tuned'])

# Adjust x-axis limits to center the bars better
ax1.set_xlim(-0.2, 0.45)

# Set y-axis limit to 500
ax1.set_ylim(0, 600)

# Add title and axis labels for perplexity plot
ax1.set_title('Male vs Female Perplexity for DBMDZ BERT', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Model', fontsize=12)
ax1.set_ylabel('Average Perplexity', fontsize=12)

# Function to add value labels - modified to place them closer to bars
def add_value_label(ax, bar, value):
    height = bar.get_height()
    x_position = bar.get_x() + bar.get_width()/2
    
    # Place labels just above the bars (reduced offset)
    y_offset = height * 0.03  # 3% of the height
    if y_offset < 2:  # Minimum offset
        y_offset = 2
    
    ax.text(x_position, height + y_offset, f'{value:.2f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add value labels to each bar
perplexity_values = [
    bert_baseline_male, bert_baseline_female,
    bert_finetuned_male, bert_finetuned_female
]

for bar, value in zip(bars, perplexity_values):
    add_value_label(ax1, bar[0], value)

# Add a legend for perplexity plot
ax1.legend(loc='upper right')

# Right subplot - Bias Score Distribution
# Function to create KDE plot
def create_bias_kde(data, ax, model_name):
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
                    label='Baseline', ax=ax, warn_singular=False)
    except Exception as e:
        print(f"Warning: Could not create KDE for baseline {model_name}: {e}")
        # Fallback to histogram if KDE fails
        ax.hist(baseline_data['avg_bias_score'], bins=20, alpha=0.7, 
                color='#000080', label='Baseline', density=True)
    
    try:
        sns.kdeplot(data=finetuned_data, x='avg_bias_score',
                    fill=True, alpha=0.7, linewidth=2, color='#008080', 
                    label='Fine-tuned', ax=ax, warn_singular=False)
    except Exception as e:
        print(f"Warning: Could not create KDE for finetuned {model_name}: {e}")
        # Fallback to histogram if KDE fails
        ax.hist(finetuned_data['avg_bias_score'], bins=20, alpha=0.7, 
                color='#008080', label='Fine-tuned', density=True)
    
    # Add a vertical line at x=0 (no bias reference)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='No bias')
    
    # Add vertical dotted lines for the means (using ALL data, not just filtered)
    ax.axvline(x=baseline_mean, color='#000080', linestyle=':', linewidth=2, 
                label=f'Baseline avg: {baseline_mean:.2f}')
    ax.axvline(x=finetuned_mean, color='#008080', linestyle=':', linewidth=2, 
                label=f'Fine-tuned avg: {finetuned_mean:.2f}')
    
    # Set title for bias plot with increased padding
    ax.set_title(f'Bias Score Distribution for DBMDZ BERT', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Bias Score (Female-Male Perplexity Difference)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlim(-150, 150)  # Set x-axis limits
    ax.set_ylim(0.0, 0.2)  # Set x-axis limits
    
    # Add legend
    ax.legend(frameon=True, fontsize=10, loc='upper right')

# Create bias distribution plot
create_bias_kde(bert_avg_results, ax2, 'dbmdz')

# Add global title for the entire figure with increased spacing
fig.suptitle('Perplexity-based Gender Bias Measurement', fontsize=16, y=0.98)

# Adjust layout with increased spacing between main title and subplot titles
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Reduced from 0.88 to create more space

# Save the figure
output_path = f'../data/perplexity_measure/perplexity_measure_{typ}/dbmdz_model_bias_perplexity_groups.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved combined analysis to {output_path}")



############# BASELINE VS. FINE-TUNED ################

################## MULTIPLE MODELS ###################



print("\nCreating figures for perplexity comparison (baseline vs fine-tuned)...")

# Ensure directory exists
os.makedirs('../data/perplexity_measure/combined', exist_ok=True)

typ = "gender_neutral"
model_names = ["dbmdz", "google-bert"]  # First figure: DBMDZ and Google BERT
display_names = ["DBMDZ BERT", "GOOGLE BERT"]

# Load the CSV files for all models
model_results = {}

try:
    for model_name in model_names:
        # Load perplexity data
        results_path = f'../data/perplexity_measure/Lou/perplexity_measure_{typ}/{model_name}/test_all.csv'
        
        model_results[model_name] = pd.read_csv(results_path)
        
        # Check if prof_gender column exists
        if 'prof_gender' not in model_results[model_name].columns:
            print(f"Error: 'prof_gender' column not found in the dataset for {model_name}!")
            continue
            
        print(f"Successfully loaded {model_name} perplexity data with {len(model_results[model_name])} rows")
        
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit(1)

# Define the profession gender groups in the desired order
prof_gender_groups = ['female', 'male', 'balanced']

# Create a figure with 3x2 subplots (3 rows, 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(14, 15))

# Set specific y-limits for each model
y_limits = {
    "DBMDZ BERT": 100,
    "GOOGLE BERT": 750
}

# Process each model in columns
for i, (model_name, display_name) in enumerate(zip(model_names, display_names)):
    if model_name not in model_results:
        print(f"Skipping {model_name} - data not loaded")
        continue
        
    results = model_results[model_name]
    y_limit = y_limits[display_name]
    print(f"Processing model: {display_name}")
    
    # Process each profession group in rows
    for j, prof_gender in enumerate(prof_gender_groups):
        # Get the correct subplot
        ax = axes[j, i]
        
        # Filter the data for this profession gender group
        current_results = results[results['prof_gender'] == prof_gender]
        
        # Calculate average perplexity for baseline
        baseline_data = current_results[current_results['model'] == 'baseline'].groupby(['model']).agg({
            'male_perplexity': 'mean',
            'female_perplexity': 'mean'
        }).reset_index()
        
        # Calculate average perplexity for finetuned
        finetuned_data = current_results[current_results['model'] == 'finetuned'].groupby(['model']).agg({
            'male_perplexity': 'mean',
            'female_perplexity': 'mean'
        }).reset_index()
        
        if len(baseline_data) > 0 and len(finetuned_data) > 0:
            # Get baseline perplexity values
            baseline_male = baseline_data['male_perplexity'].values[0]
            baseline_female = baseline_data['female_perplexity'].values[0]
            baseline_bias = baseline_female - baseline_male
            
            # Get finetuned perplexity values
            finetuned_male = finetuned_data['male_perplexity'].values[0]
            finetuned_female = finetuned_data['female_perplexity'].values[0]
            finetuned_bias = finetuned_female - finetuned_male
            
            # Set up bar positions with reduced bar width and closer spacing
            bar_width = 0.05
            # Reduce spacing between baseline and fine-tuned groups (0.3 instead of 0.5)
            positions = [0, 0.15]  # Baseline, Finetuned
            
            # Plot the bars
            male_baseline = ax.bar(positions[0] - bar_width/2, baseline_male, bar_width, color='blue', label='Male')
            female_baseline = ax.bar(positions[0] + bar_width/2, baseline_female, bar_width, color='orange', label='Female')
            
            male_finetuned = ax.bar(positions[1] - bar_width/2, finetuned_male, bar_width, color='blue')
            female_finetuned = ax.bar(positions[1] + bar_width/2, finetuned_female, bar_width, color='orange')
            
            # Set x-ticks
            ax.set_xticks(positions)
            ax.set_xticklabels(['Baseline', 'Fine-tuned'])
            
            # Format profession gender group for title
            prof_group_title = {
                'male': 'Male-dominated',
                'female': 'Female-dominated', 
                'balanced': 'Gender-balanced'
            }.get(prof_gender, prof_gender.capitalize())
            
            # Set title for subplot with bold font
            ax.set_title(f'{display_name}: {prof_group_title} Professions', fontsize=12, fontweight='bold')
            
            # Add value labels
            ax.text(positions[0] - bar_width/2, baseline_male + y_limit*0.02, f'{baseline_male:.2f}', 
                    ha='center', va='bottom', fontsize=9)
            ax.text(positions[0] + bar_width/2, baseline_female + y_limit*0.02, f'{baseline_female:.2f}', 
                    ha='center', va='bottom', fontsize=9)
            ax.text(positions[1] - bar_width/2, finetuned_male + y_limit*0.02, f'{finetuned_male:.2f}', 
                    ha='center', va='bottom', fontsize=9)
            ax.text(positions[1] + bar_width/2, finetuned_female + y_limit*0.02, f'{finetuned_female:.2f}', 
                    ha='center', va='bottom', fontsize=9)
            
            # Add bias score text boxes directly above the bars
            baseline_bias_status = "against women" if baseline_bias > 0 else "against men" if baseline_bias < 0 else "neutral"
            finetuned_bias_status = "against women" if finetuned_bias > 0 else "against men" if finetuned_bias < 0 else "neutral"
            
            # Position bias scores with special case for Google BERT
            max_baseline_height = max(baseline_male, baseline_female)
            max_finetuned_height = max(finetuned_male, finetuned_female)
            
            # Special case for Google BERT - position text in the middle of the bars
            if display_name == "GOOGLE BERT":
                # For Google BERT, position the baseline bias score in the middle of the plot
                ax.text(positions[0], y_limit*0.4, f'Bias Score: {baseline_bias:.2f}\n({baseline_bias_status})', 
                        ha='center', va='center', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
                
                # Still keep the finetuned bias score above the bars with appropriate spacing
                ax.text(positions[1], max_finetuned_height + y_limit*0.15, f'Bias Score: {finetuned_bias:.2f}\n({finetuned_bias_status})', 
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
            else:
                # For other models, keep the original positioning with the offset
                ax.text(positions[0], max_baseline_height + y_limit*0.15, f'Bias Score: {baseline_bias:.2f}\n({baseline_bias_status})', 
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
                
                ax.text(positions[1], max_finetuned_height + y_limit*0.15, f'Bias Score: {finetuned_bias:.2f}\n({finetuned_bias_status})', 
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
            
            # Set y-limit
            ax.set_ylim(0, y_limit)
            
            # Add y-label only to leftmost plots
            if i == 0:
                ax.set_ylabel('Average Perplexity', fontsize=11)
                
            # Add legend to the first subplot in the upper right corner
            if j == 0 and i == 0:
                ax.legend(loc='upper right', fontsize=10)
        else:
            ax.text(0.5, 0.5, "No data available", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)

# Remove the global legend since we now have it in the first subplot
# fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.98))

# Add a global title
fig.suptitle('Perplexity Comparison: Baseline vs. Fine-tuned Models (DBMDZ BERT and GOOGLE BERT)', 
             fontsize=16, y=0.995)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.3)

# Save the figure
output_path = f'../data/perplexity_measure/Lou/perplexity_measure_{typ}/dbmdz_google_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved DBMDZ and Google BERT comparison to {output_path}")

# Now create second figure for G-BERT and DISTILBERT
model_names = ["deepset-bert", "distilbert"]
display_names = ["G-BERT", "DISTILBERT"]

# Load the CSV files for all models
model_results = {}

try:
    for model_name in model_names:
        # Load perplexity data
        results_path = f'../data/perplexity_measure/Lou/perplexity_measure_{typ}/{model_name}/test_all.csv'
        
        model_results[model_name] = pd.read_csv(results_path)
        
        # Check if prof_gender column exists
        if 'prof_gender' not in model_results[model_name].columns:
            print(f"Error: 'prof_gender' column not found in the dataset for {model_name}!")
            continue
            
        print(f"Successfully loaded {model_name} perplexity data with {len(model_results[model_name])} rows")
        
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit(1)

# Define the profession gender groups in the desired order
prof_gender_groups = ['female', 'male', 'balanced']

# Create a figure with 3x2 subplots (3 rows, 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(14, 15))

# Set specific y-limits for each model
y_limits = {
    "G-BERT": 100,
    "DISTILBERT": 100
}

# Process each model in columns
for i, (model_name, display_name) in enumerate(zip(model_names, display_names)):
    if model_name not in model_results:
        print(f"Skipping {model_name} - data not loaded")
        continue
        
    results = model_results[model_name]
    y_limit = y_limits[display_name]
    print(f"Processing model: {display_name}")
    
    # Process each profession group in rows
    for j, prof_gender in enumerate(prof_gender_groups):
        # Get the correct subplot
        ax = axes[j, i]
        
        # Filter the data for this profession gender group
        current_results = results[results['prof_gender'] == prof_gender]
        
        # Calculate average perplexity for baseline
        baseline_data = current_results[current_results['model'] == 'baseline'].groupby(['model']).agg({
            'male_perplexity': 'mean',
            'female_perplexity': 'mean'
        }).reset_index()
        
        # Calculate average perplexity for finetuned
        finetuned_data = current_results[current_results['model'] == 'finetuned'].groupby(['model']).agg({
            'male_perplexity': 'mean',
            'female_perplexity': 'mean'
        }).reset_index()
        
        if len(baseline_data) > 0 and len(finetuned_data) > 0:
            # Get baseline perplexity values
            baseline_male = baseline_data['male_perplexity'].values[0]
            baseline_female = baseline_data['female_perplexity'].values[0]
            baseline_bias = baseline_female - baseline_male
            
            # Get finetuned perplexity values
            finetuned_male = finetuned_data['male_perplexity'].values[0]
            finetuned_female = finetuned_data['female_perplexity'].values[0]
            finetuned_bias = finetuned_female - finetuned_male
            
            # Set up bar positions with reduced bar width and closer spacing
            bar_width = 0.05
            # Reduce spacing between baseline and fine-tuned groups (0.3 instead of 0.5)
            positions = [0, 0.15]  # Baseline, Finetuned
            
            # Plot the bars
            male_baseline = ax.bar(positions[0] - bar_width/2, baseline_male, bar_width, color='blue', label='Male')
            female_baseline = ax.bar(positions[0] + bar_width/2, baseline_female, bar_width, color='orange', label='Female')
            
            male_finetuned = ax.bar(positions[1] - bar_width/2, finetuned_male, bar_width, color='blue')
            female_finetuned = ax.bar(positions[1] + bar_width/2, finetuned_female, bar_width, color='orange')
            
            # Set x-ticks
            ax.set_xticks(positions)
            ax.set_xticklabels(['Baseline', 'Fine-tuned'])
            
            # Format profession gender group for title
            prof_group_title = {
                'male': 'Male-dominated',
                'female': 'Female-dominated', 
                'balanced': 'Gender-balanced'
            }.get(prof_gender, prof_gender.capitalize())
            
            # Set title for subplot with bold font
            ax.set_title(f'{display_name}: {prof_group_title} Professions', fontsize=12, fontweight='bold')
            
            # Add value labels
            ax.text(positions[0] - bar_width/2, baseline_male + y_limit*0.02, f'{baseline_male:.2f}', 
                    ha='center', va='bottom', fontsize=9)
            ax.text(positions[0] + bar_width/2, baseline_female + y_limit*0.02, f'{baseline_female:.2f}', 
                    ha='center', va='bottom', fontsize=9)
            ax.text(positions[1] - bar_width/2, finetuned_male + y_limit*0.02, f'{finetuned_male:.2f}', 
                    ha='center', va='bottom', fontsize=9)
            ax.text(positions[1] + bar_width/2, finetuned_female + y_limit*0.02, f'{finetuned_female:.2f}', 
                    ha='center', va='bottom', fontsize=9)
            
            # Add bias score text boxes directly above the bars
            baseline_bias_status = "against women" if baseline_bias > 0 else "against men" if baseline_bias < 0 else "neutral"
            finetuned_bias_status = "against women" if finetuned_bias > 0 else "against men" if finetuned_bias < 0 else "neutral"
            
            # Position bias scores directly above the bars with increased vertical offset
            max_baseline_height = max(baseline_male, baseline_female)
            max_finetuned_height = max(finetuned_male, finetuned_female)
            
            # Use a larger vertical offset (0.15 instead of 0.05) to move text boxes higher
            ax.text(positions[0], max_baseline_height + y_limit*0.15, f'Bias Score: {baseline_bias:.2f}\n({baseline_bias_status})', 
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
            
            ax.text(positions[1], max_finetuned_height + y_limit*0.15, f'Bias Score: {finetuned_bias:.2f}\n({finetuned_bias_status})', 
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
            
            # Set y-limit
            ax.set_ylim(0, y_limit)
            
            # Add y-label only to leftmost plots
            if i == 0:
                ax.set_ylabel('Average Perplexity', fontsize=11)
                
            # Add legend to the first subplot in the upper right corner
            if j == 0 and i == 0:
                ax.legend(loc='upper right', fontsize=10)
        else:
            ax.text(0.5, 0.5, "No data available", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)

# Remove the global legend since we now have it in the first subplot
# fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.98))

# Add a global title
fig.suptitle('Perplexity Comparison: Baseline vs. Fine-tuned Models (G-BERT and DISTILBERT)', 
             fontsize=16, y=0.995)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.3)

# Save the figure
output_path = f'../data/perplexity_measure/Lou/perplexity_measure_{typ}/gbert_distilbert_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved G-BERT and DistilBERT comparison to {output_path}")

print("\nAnalysis complete!")