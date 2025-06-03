import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


################### FOR ENGLISH BERT #####################
################### BASELINE VS FINETUNED ################


print("\nCreating figures by profession gender group for bert model analysis...")

typ = "english"
model_name = "bert"

# Ensure directory exists
os.makedirs(f'../results/perplexity_measure/Gap/{typ}/{model_name}/', exist_ok=True)

# Load the CSV files
try:
    # Load perplexity data
    bert_results = pd.read_csv(f'../results/perplexity_measure/Gap/{typ}/{model_name}/all_seeds_bias_{model_name}_{typ}_adapted.csv')
    # Load averaged bias results
    bert_avg_results = pd.read_csv(f'../results/perplexity_measure/Gap/{typ}/{model_name}/averaged_bias_{model_name}_{typ}_adapted.csv')
    
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
output_path = f'../results/perplexity_measure/Gap/{typ}/{model_name}/{model_name}_{typ}_baseline_vs_finetuned_perplexity.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved combined comparison figure to {output_path}")

print("\nAnalysis complete!")








##################### BASELINE MEASUREMENT ######################
#################### SINGLE MODEL ###############################
#################### GERMAN #####################################


print("\nCreating figures by profession gender group for baseline DBMDZ BERT model analysis...")

# Ensure directory exists
os.makedirs('../results/perplexity_measure/Gap/', exist_ok=True)


# here change the typ to regular, token_balanced or gender_neutral

typ = "gender_neutral"
model_name = "dbmdz"

# Load the CSV files
try:
    # Load perplexity data
    bert_results = pd.read_csv(f'../results/perplexity_measure/Gap/{typ}/{model_name}/all_seeds_bias_{model_name}_{typ}_adapted.csv')
    # Load averaged bias results
    bert_avg_results = pd.read_csv(f'../results/perplexity_measure/Gap/{typ}/{model_name}/averaged_bias_{model_name}_{typ}_adapted.csv')
    
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
    axes[i].set_ylim(0, 100)
    
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
fig.suptitle(f'Perplexity-based Gender Bias Measurement for Baseline DBMDZ BERT {typ}', fontsize=14, y=1.0)

# Add legend only to the first subplot, positioned lower right
axes[0].legend(loc='upper right')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(wspace=0.25)  # Add some space between subplots

# Save the figure
output_path = f'../results/perplexity_measure/Gap/{typ}/{model_name}/{model_name}_{typ}_baseline_perplexity.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved baseline comparison figure to {output_path}")

print("\nAnalysis complete!")





############## BASELINE ################################
############## MULTIPLE MODELS #########################
############## GERMAN ##################################


print("\nCreating figures for baseline perplexity comparison...")

# Ensure directory exists
os.makedirs('../results/perplexity_measure/Lou', exist_ok=True)

# change the typ to regular or gender_neutral
typ = "gender_neutral"

# Models to include in the comparison
model_names = ["dbmdz", "google_bert", "deepset_bert", "distilbert"]
display_names = ["DBMDZ BERT", "GOOGLE BERT", "G-BERT", "DISTILBERT"]

# Define the profession gender groups in the desired order
prof_gender_groups = ['female', 'male', 'balanced']
prof_group_titles = {
    'female': 'Female-dominated',
    'male': 'Male-dominated', 
    'balanced': 'Gender-balanced'
}

# Load the CSV files for all models
model_results = {}

try:
    for model_name in model_names:
        # Load perplexity data
        results_path = f'../results/perplexity_measure/Lou/{typ}/{model_name}/lou_all_seeds_bias_{model_name}_{typ}_adapted.csv'
        
        model_results[model_name] = pd.read_csv(results_path)
        
        # Check if prof_gender column exists
        if 'prof_gender' not in model_results[model_name].columns:
            print(f"Error: 'prof_gender' column not found in the dataset for {model_name}!")
            continue
            
        print(f"Successfully loaded {model_name} perplexity data with {len(model_results[model_name])} rows")
        
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit(1)

# Create a figure with three subplots (one for each profession gender group)
# Reduce the overall figure height to make it more compact
fig, axes = plt.subplots(3, 1, figsize=(14, 14))  # Reduced from (14, 18) to (14, 14)

# Process each profession gender group
for j, prof_gender in enumerate(prof_gender_groups):
    # Get the subplot for this profession gender group
    ax = axes[j]
    
    # Format profession gender group for title
    prof_group_title = prof_group_titles.get(prof_gender, prof_gender.capitalize())
    
    # Set title for subplot with bold font - reduce font size
    ax.set_title(f'{prof_group_title} Professions', fontsize=12, fontweight='bold')
    
    # Add y-label - reduce font size
    ax.set_ylabel('Average Perplexity', fontsize=10)
    
    # Set up bar positions and widths - reduce gaps between models
    bar_width = 0.2  # Width of each bar
    model_gap = 0.3  # Gap between different models
    
    # Calculate the total width needed
    total_width = (bar_width * 2) * len(model_names) + model_gap * (len(model_names) - 1)
    
    # Calculate starting position to center the entire group
    start_pos = 0.5 - total_width/2
    
    # Keep track of legend handles
    legend_handles = []
    
    # Process each model
    for i, (model_name, display_name) in enumerate(zip(model_names, display_names)):
        if model_name not in model_results:
            print(f"Skipping {model_name} - data not loaded")
            continue
        
        results = model_results[model_name]
        
        # Filter the data for this profession gender group
        current_results = results[results['prof_gender'] == prof_gender]
        
        # Calculate average perplexity for baseline
        baseline_data = current_results[current_results['model'] == 'baseline'].groupby(['model']).agg({
            'male_perplexity': 'mean',
            'female_perplexity': 'mean'
        }).reset_index()
        
        # Calculate position for this model's bars
        # Each model gets a position with space for 2 bars (male and female)
        model_pos = start_pos + i * (bar_width * 2 + model_gap) + bar_width
        
        if len(baseline_data) > 0:
            # Get baseline perplexity values
            baseline_male = baseline_data['male_perplexity'].values[0]
            baseline_female = baseline_data['female_perplexity'].values[0]
            baseline_bias = baseline_female - baseline_male
            
            # Plot the bars with no gap between male and female
            male_bar = ax.bar(model_pos - bar_width/2, baseline_male, bar_width, 
                         color='blue', label='Male' if i == 0 else "")
            female_bar = ax.bar(model_pos + bar_width/2, baseline_female, bar_width, 
                           color='orange', label='Female' if i == 0 else "")
            
            # Save legend handles for the first model only
            if i == 0:
                legend_handles = [male_bar, female_bar]
            
            # Add value labels (placed higher)
            y_offset = 20  # Fixed offset for value labels
            ax.text(model_pos - bar_width/2, baseline_male + y_offset, f'{baseline_male:.2f}', 
                    ha='center', va='bottom', fontsize=8)
            ax.text(model_pos + bar_width/2, baseline_female + y_offset, f'{baseline_female:.2f}', 
                    ha='center', va='bottom', fontsize=8)
            
            # Add bias score text box
            baseline_bias_status = "against women" if baseline_bias > 0 else "against men" if baseline_bias < 0 else "neutral"
            
            # Position bias scores higher to avoid overlap with bars
            max_height = max(baseline_male, baseline_female)
            
            if display_name == "GOOGLE BERT":
                # For Google BERT, position higher than before
                box_y_pos = 350  # Increased from 250
            else:
                # For other models, position well above the bars and value labels
                box_y_pos = max_height + 80  # Increased from 30 to 80
            
            # Add the bias score text box
            ax.text(model_pos, box_y_pos, f'Bias Score: {baseline_bias:.2f}\n({baseline_bias_status})', 
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
        else:
            ax.text(model_pos, 0.5, "No data", ha='center', va='center', fontsize=8)
    
    # Set x-ticks with model names and "Baseline" underneath
    x_positions = [start_pos + i * (bar_width * 2 + model_gap) + bar_width for i in range(len(display_names))]
    ax.set_xticks(x_positions)
    
    # Create two-line x-tick labels with model name and "Baseline"
    x_labels = [f"{name}\nBaseline" for name in display_names]
    ax.set_xticklabels(x_labels, fontsize=9)  # Reduced font size
    
    # Add legend to the first subplot in the upper right corner
    if j == 0:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9)  # Reduced font size
    
    # Set uniform y-limit of 750 for all subplots
    ax.set_ylim(0, 750)

# Add a global title
fig.suptitle('Perplexity Comparison: Baseline German BERT Models', fontsize=14, y=0.98)  # Reduced font size

# Adjust layout with minimal spacing between subplots
plt.tight_layout()
# Reduce the space between subplots significantly
plt.subplots_adjust(top=0.95, hspace=0.1)  # Reduced hspace from 0.3 to 0.1

# Save the figure
output_path = f'../results/perplexity_measure/Lou/{typ}/lou_all_models_baseline_perplexity.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved all models baseline comparison to {output_path}")

print("\nAnalysis complete!")





################## BASELINE VS. FINETUNED ###################
################## FOR SINGLE MODEL #########################
################## FOR GAP ##################################

print("\nCreating perplexity comparison across profession groups...")


# Ensure directory exists
os.makedirs('../results/perplexity_measure/Gap/', exist_ok=True)

# change the typ to regular or gender_neutral

typ = "gender_neutral"
model_name = "dbmdz"

# Load the CSV files for DBMDZ model
try:
    # Load perplexity data
    bert_results = pd.read_csv(f'../results/perplexity_measure/Gap/{typ}/{model_name}/all_seeds_bias_{model_name}_{typ}_adapted.csv')
    
    # Load averaged bias results
    bert_avg_results = pd.read_csv(f'../results/perplexity_measure/Gap/{typ}/{model_name}/averaged_bias_{model_name}_{typ}_adapted.csv')
    
    print(f"Successfully loaded bert perplexity data with {len(bert_results)} rows")
    print(f"Successfully loaded bert bias data with {len(bert_avg_results)} rows")
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit(1)

# Define profession groups
prof_groups = ['female', 'male', 'balanced']
prof_group_titles = {
    'female': 'Female-dominated Professions',
    'male': 'Male-dominated Professions', 
    'balanced': 'Gender-balanced Professions'
}

# Create a figure with 3 rows and 2 columns (6 subplots total)
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# Process each profession group
for i, prof_group in enumerate(prof_groups):
    # Filter data for this profession group
    group_data = bert_results[bert_results['prof_gender'] == prof_group]
    group_bias_data = bert_avg_results[bert_avg_results['prof_gender'] == prof_group]
    
    # Calculate average perplexity for this group
    group_avg = group_data.groupby(['model']).agg({
        'male_perplexity': 'mean',
        'female_perplexity': 'mean'
    }).reset_index()
    
    # Get perplexity values
    baseline_male = group_avg[group_avg['model'] == 'baseline']['male_perplexity'].values[0]
    baseline_female = group_avg[group_avg['model'] == 'baseline']['female_perplexity'].values[0]
    finetuned_male = group_avg[group_avg['model'] == 'finetuned']['male_perplexity'].values[0]
    finetuned_female = group_avg[group_avg['model'] == 'finetuned']['female_perplexity'].values[0]
    
    # Left subplot - Perplexity Comparison
    ax_left = axes[i, 0]
    
    positions = [0, 0.25]  # Baseline, Fine-tuned
    bar_width = 0.1
    
    # Create bars
    bars = [
        ax_left.bar(positions[0] - bar_width/2, baseline_male, bar_width, color='blue', label='Male' if i == 0 else ""),
        ax_left.bar(positions[0] + bar_width/2, baseline_female, bar_width, color='orange', label='Female' if i == 0 else ""),
        ax_left.bar(positions[1] - bar_width/2, finetuned_male, bar_width, color='blue'),
        ax_left.bar(positions[1] + bar_width/2, finetuned_female, bar_width, color='orange'),
    ]
    
    # Set x-ticks and labels
    ax_left.set_xticks(positions)
    ax_left.set_xticklabels(['Baseline', 'Fine-tuned'])
    ax_left.set_xlim(-0.2, 0.45)
    
    # Set fixed y-axis limit to 700 for all subplots
    ax_left.set_ylim(0, 700)
    
    # Add title and labels
    ax_left.set_title(f'Male vs Female Perplexity: {prof_group_titles[prof_group]}', 
                      fontsize=12, fontweight='bold')
    if i == 2:  # Only add x-label to bottom subplot
        ax_left.set_xlabel('Model', fontsize=10)
    ax_left.set_ylabel('Average Perplexity', fontsize=10)
    
    # Add value labels
    perplexity_values = [baseline_male, baseline_female, finetuned_male, finetuned_female]
    for bar, value in zip(bars, perplexity_values):
        height = bar[0].get_height()
        x_position = bar[0].get_x() + bar[0].get_width()/2
        y_offset = height * 0.03
        if y_offset < 2:
            y_offset = 2
        ax_left.text(x_position, height + y_offset, f'{value:.2f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add legend only to the first subplot
    if i == 0:
        ax_left.legend(loc='upper right', fontsize=9)
    
    # Right subplot - Bias Score Distribution
    ax_right = axes[i, 1]
    
    # Filter out extreme outliers for visualization
    filtered_bias_data = group_bias_data[
        (group_bias_data['avg_bias_score'] > -200) & 
        (group_bias_data['avg_bias_score'] < 150)
    ]
    
    # Calculate means (using all data)
    baseline_mean = group_bias_data[group_bias_data['model'] == 'baseline']['avg_bias_score'].mean()
    finetuned_mean = group_bias_data[group_bias_data['model'] == 'finetuned']['avg_bias_score'].mean()
    
    # Get data for each model
    baseline_bias = filtered_bias_data[filtered_bias_data['model'] == 'baseline']
    finetuned_bias = filtered_bias_data[filtered_bias_data['model'] == 'finetuned']
    
    # Create KDE plots
    try:
        sns.kdeplot(data=baseline_bias, x='avg_bias_score', 
                    fill=True, alpha=0.7, linewidth=2, color='#000080', 
                    label='Baseline', ax=ax_right, warn_singular=False)
    except:
        ax_right.hist(baseline_bias['avg_bias_score'], bins=20, alpha=0.7, 
                     color='#000080', label='Baseline', density=True)
    
    try:
        sns.kdeplot(data=finetuned_bias, x='avg_bias_score',
                    fill=True, alpha=0.7, linewidth=2, color='#008080', 
                    label='Fine-tuned', ax=ax_right, warn_singular=False)
    except:
        ax_right.hist(finetuned_bias['avg_bias_score'], bins=20, alpha=0.7, 
                     color='#008080', label='Fine-tuned', density=True)
    
    # Add reference lines
    ax_right.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    ax_right.axvline(x=baseline_mean, color='#000080', linestyle=':', linewidth=2)
    ax_right.axvline(x=finetuned_mean, color='#008080', linestyle=':', linewidth=2)
    
    # Set title and labels
    ax_right.set_title(f'Bias Score Distribution: {prof_group_titles[prof_group]}', 
                      fontsize=12, fontweight='bold')
    if i == 2:  # Only add x-label to bottom subplot
        ax_right.set_xlabel('Bias Score (Female-Male Perplexity Difference)', fontsize=10)
    ax_right.set_ylabel('Density', fontsize=10)
    ax_right.set_xlim(-150, 150)
    ax_right.set_ylim(0.0, 0.2)
    
    # Add legend with bias scores (for all subplots)
    legend_labels = [
        'Baseline',
        'Fine-tuned', 
        'No bias',
        f'Baseline avg: {baseline_mean:.2f}',
        f'Fine-tuned avg: {finetuned_mean:.2f}'
    ]
    ax_right.legend(legend_labels, frameon=True, fontsize=9, loc='upper right')

# Add global title
fig.suptitle(f'Perplexity-based Gender Bias Measurement for {model_name.upper()} BERT {typ.replace("_", "-").title()} by Profession Category', 
             fontsize=16, y=0.98)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.94, hspace=0.3, wspace=0.3)

# Save the figure
output_path = f'../results/perplexity_measure/Gap/{typ}/{model_name}/{model_name}_{typ}_baseline_vs_finetuned_perplexity.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved across-groups analysis to {output_path}")




################## BASELINE VS. FINETUNED ###################
################## FOR SINGLE MODEL #########################
################## FOR LOU ##################################

print("\nCreating perplexity comparison across profession groups...")


# Ensure directory exists
os.makedirs('../results/perplexity_measure/Lou/', exist_ok=True)

# change the typ to regular or gender_neutral

typ = "gender_neutral"
model_name = "dbmdz"

# Load the CSV files for DBMDZ model
try:
    # Load perplexity data
    bert_results = pd.read_csv(f'../results/perplexity_measure/Lou/{typ}/{model_name}/lou_all_seeds_bias_{model_name}_{typ}_adapted.csv')
    
    # Load averaged bias results
    bert_avg_results = pd.read_csv(f'../results/perplexity_measure/Lou/{typ}/{model_name}/lou_averaged_bias_{model_name}_{typ}_adapted.csv')
    
    print(f"Successfully loaded bert perplexity data with {len(bert_results)} rows")
    print(f"Successfully loaded bert bias data with {len(bert_avg_results)} rows")
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit(1)

# Define profession groups
prof_groups = ['female', 'male', 'balanced']
prof_group_titles = {
    'female': 'Female-dominated Professions',
    'male': 'Male-dominated Professions', 
    'balanced': 'Gender-balanced Professions'
}

# Create a figure with 3 rows and 2 columns (6 subplots total)
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# Process each profession group
for i, prof_group in enumerate(prof_groups):
    # Filter data for this profession group
    group_data = bert_results[bert_results['prof_gender'] == prof_group]
    group_bias_data = bert_avg_results[bert_avg_results['prof_gender'] == prof_group]
    
    # Calculate average perplexity for this group
    group_avg = group_data.groupby(['model']).agg({
        'male_perplexity': 'mean',
        'female_perplexity': 'mean'
    }).reset_index()
    
    # Get perplexity values
    baseline_male = group_avg[group_avg['model'] == 'baseline']['male_perplexity'].values[0]
    baseline_female = group_avg[group_avg['model'] == 'baseline']['female_perplexity'].values[0]
    finetuned_male = group_avg[group_avg['model'] == 'finetuned']['male_perplexity'].values[0]
    finetuned_female = group_avg[group_avg['model'] == 'finetuned']['female_perplexity'].values[0]
    
    # Left subplot - Perplexity Comparison
    ax_left = axes[i, 0]
    
    positions = [0, 0.25]  # Baseline, Fine-tuned
    bar_width = 0.1
    
    # Create bars
    bars = [
        ax_left.bar(positions[0] - bar_width/2, baseline_male, bar_width, color='blue', label='Male' if i == 0 else ""),
        ax_left.bar(positions[0] + bar_width/2, baseline_female, bar_width, color='orange', label='Female' if i == 0 else ""),
        ax_left.bar(positions[1] - bar_width/2, finetuned_male, bar_width, color='blue'),
        ax_left.bar(positions[1] + bar_width/2, finetuned_female, bar_width, color='orange'),
    ]
    
    # Set x-ticks and labels
    ax_left.set_xticks(positions)
    ax_left.set_xticklabels(['Baseline', 'Fine-tuned'])
    ax_left.set_xlim(-0.2, 0.45)
    
    # Set fixed y-axis limit to 700 for all subplots
    ax_left.set_ylim(0, 100)
    
    # Add title and labels
    ax_left.set_title(f'Male vs Female Perplexity: {prof_group_titles[prof_group]}', 
                      fontsize=12, fontweight='bold')
    if i == 2:  # Only add x-label to bottom subplot
        ax_left.set_xlabel('Model', fontsize=10)
    ax_left.set_ylabel('Average Perplexity', fontsize=10)
    
    # Add value labels
    perplexity_values = [baseline_male, baseline_female, finetuned_male, finetuned_female]
    for bar, value in zip(bars, perplexity_values):
        height = bar[0].get_height()
        x_position = bar[0].get_x() + bar[0].get_width()/2
        y_offset = height * 0.03
        if y_offset < 2:
            y_offset = 2
        ax_left.text(x_position, height + y_offset, f'{value:.2f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add legend only to the first subplot
    if i == 0:
        ax_left.legend(loc='upper right', fontsize=9)
    
    # Right subplot - Bias Score Distribution
    ax_right = axes[i, 1]
    
    # Filter out extreme outliers for visualization
    filtered_bias_data = group_bias_data[
        (group_bias_data['avg_bias_score'] > -200) & 
        (group_bias_data['avg_bias_score'] < 150)
    ]
    
    # Calculate means (using all data)
    baseline_mean = group_bias_data[group_bias_data['model'] == 'baseline']['avg_bias_score'].mean()
    finetuned_mean = group_bias_data[group_bias_data['model'] == 'finetuned']['avg_bias_score'].mean()
    
    # Get data for each model
    baseline_bias = filtered_bias_data[filtered_bias_data['model'] == 'baseline']
    finetuned_bias = filtered_bias_data[filtered_bias_data['model'] == 'finetuned']
    
    # Create KDE plots
    try:
        sns.kdeplot(data=baseline_bias, x='avg_bias_score', 
                    fill=True, alpha=0.7, linewidth=2, color='#000080', 
                    label='Baseline', ax=ax_right, warn_singular=False)
    except:
        ax_right.hist(baseline_bias['avg_bias_score'], bins=20, alpha=0.7, 
                     color='#000080', label='Baseline', density=True)
    
    try:
        sns.kdeplot(data=finetuned_bias, x='avg_bias_score',
                    fill=True, alpha=0.7, linewidth=2, color='#008080', 
                    label='Fine-tuned', ax=ax_right, warn_singular=False)
    except:
        ax_right.hist(finetuned_bias['avg_bias_score'], bins=20, alpha=0.7, 
                     color='#008080', label='Fine-tuned', density=True)
    
    # Add reference lines
    ax_right.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    ax_right.axvline(x=baseline_mean, color='#000080', linestyle=':', linewidth=2)
    ax_right.axvline(x=finetuned_mean, color='#008080', linestyle=':', linewidth=2)
    
    # Set title and labels
    ax_right.set_title(f'Bias Score Distribution: {prof_group_titles[prof_group]}', 
                      fontsize=12, fontweight='bold')
    if i == 2:  # Only add x-label to bottom subplot
        ax_right.set_xlabel('Bias Score (Female-Male Perplexity Difference)', fontsize=10)
    ax_right.set_ylabel('Density', fontsize=10)
    ax_right.set_xlim(-150, 150)
    ax_right.set_ylim(0.0, 0.2)
    
    # Add legend with bias scores (for all subplots)
    legend_labels = [
        'Baseline',
        'Fine-tuned', 
        'No bias',
        f'Baseline avg: {baseline_mean:.2f}',
        f'Fine-tuned avg: {finetuned_mean:.2f}'
    ]
    ax_right.legend(legend_labels, frameon=True, fontsize=9, loc='upper right')

# Add global title
fig.suptitle(f'Perplexity-based Gender Bias Measurement for {model_name.upper()} BERT {typ.replace("_", "-").title()} by Profession Category', 
             fontsize=16, y=0.98)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.94, hspace=0.3, wspace=0.3)

# Save the figure
output_path = f'../results/perplexity_measure/Lou/{typ}/{model_name}/lou_{model_name}_{typ}_baseline_vs_finetuned_perplexity.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved across-groups analysis to {output_path}")



############# BASELINE VS. FINE-TUNED ################
################## MULTIPLE MODELS ###################

print("\nCreating figures for perplexity comparison (baseline vs fine-tuned)...")

# Ensure directory exists
os.makedirs('../results/perplexity_measure/Lou/', exist_ok=True)

typ = "gender_neutral"
model_names = ["dbmdz", "google_bert"]  # First figure: DBMDZ and Google BERT
display_names = ["DBMDZ BERT", "GOOGLE BERT"]

# Load the CSV files for all models
model_results = {}

try:
    for model_name in model_names:
        # Load perplexity data
        results_path = f'../results/perplexity_measure/Lou/{typ}/{model_name}/lou_all_seeds_bias_{model_name}_{typ}_adapted.csv'
        
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

# Add a global title
fig.suptitle('Perplexity Comparison: Baseline vs. Fine-tuned Models (DBMDZ BERT and GOOGLE BERT)', 
             fontsize=16, y=0.995)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.3)

# Save the figure
output_path = f'../results/perplexity_measure/Lou/{typ}/lou_dbmdz_google_bert_baseline_vs_finetuned.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved DBMDZ and Google BERT comparison to {output_path}")

# Now create second figure for G-BERT and DISTILBERT
model_names = ["deepset_bert", "distilbert"]
display_names = ["G-BERT", "DISTILBERT"]

# Load the CSV files for all models
model_results = {}

try:
    for model_name in model_names:
        # Load perplexity data
        results_path = f'../results/perplexity_measure/Lou/{typ}/{model_name}/lou_all_seeds_bias_{model_name}_{typ}_adapted.csv'
        
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

# Add a global title
fig.suptitle('Perplexity Comparison: Baseline vs. Fine-tuned Models (G-BERT and DISTILBERT)', 
             fontsize=16, y=0.995)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.3)

# Save the figure
output_path = f'../results/perplexity_measure/Lou/{typ}/lou_gbert_distilbert_baseline_vs_finetuned.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved G-BERT and DistilBERT comparison to {output_path}")

print("\nAnalysis complete!")