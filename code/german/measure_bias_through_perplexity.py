# Purpose: This code measures gender bias in language models by comparing the perplexity (how surprised the model is) when seeing professions with male vs. female forms in sentences.
# Dataset Loading: It loads your data file that contains sentences like "Er ist Mechaniker" and "Sie ist Mechanikerin" with their gender labels.
# Finding Pairs: It tries to match male and female sentences for the same profession (like "Mechaniker" and "Mechanikerin") so it can compare them.
# Perplexity Calculation: For each pair, it:

# Gets the perplexity for the male version ("Er ist Mechaniker")
# Gets the perplexity for the female version ("Sie ist Mechanikerin")
# Calculates the difference (female - male)
# A positive difference suggests bias against females


# Model Loading: It loads your baseline model (epoch 0) and finetuned model (epoch 3) for each random seed to see how bias changed after finetuning.


# Process all 30 profession pairs for both baseline and finetuned models
# Run the analysis for all 4 random seeds
# Calculate average bias scores and their standard deviations
# Compare how bias changed from baseline to finetuned models
# Create visualizations to help interpret the results
# Save all results to CSV files for further analysis

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# Setup CUDA
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

gpu_id = "5"  # Adjust as needed

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_id}")
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')

typ = "gender_neutral"

# Load your template dataset
df = pd.read_csv(f'../BEC-Pro/modified_file_DE_{typ}.csv', sep='\t')
print(f"Loaded template dataset with {len(df)} rows")

name_of_model = "google_bert"

# # Define model configuration and paths
base_model_id = "google-bert/bert-base-german-cased"  # Base model architecture

# Define model configuration and paths
#base_model_id = "deepset/gbert-base"  # Base model architecture

# Define seeds and model paths
seeds = [42, 116, 387, 1980]

# Find all profession pairs with male and female versions
def find_gender_pairs(df):
    """Find all profession pairs with male and female versions"""
    pairs = []
    
    # Group by English profession name for better matching
    for profession_en, group_data in df.groupby('Profession_EN'):
        male_rows = group_data[group_data['Gender'] == 'male']
        female_rows = group_data[group_data['Gender'] == 'female']
        
        if len(male_rows) > 0 and len(female_rows) > 0:
            pair = {
                'profession': profession_en,
                'male_sentence': male_rows.iloc[0]['Sentence'],
                'female_sentence': female_rows.iloc[0]['Sentence'],
                'male_profession': male_rows.iloc[0]['Profession'],
                'female_profession': female_rows.iloc[0]['Profession']
            }
            pairs.append(pair)
    
    return pairs

# Calculate perplexity for a given text
def calculate_perplexity(model, tokenizer, text):
    """Calculate sentence-level perplexity"""
    try:
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            # Using masked language modeling for perplexity calculation
            inputs["labels"] = inputs["input_ids"].clone()
            outputs = model(**inputs)
        
        perplexity = torch.exp(outputs.loss).item()
        return perplexity, token_count
    except Exception as e:
        print(f"Error calculating perplexity for text: {text}")
        print(f"Error: {e}")
        return None, None

# Process profession pairs with a given model
def process_pairs(pairs, model, tokenizer, model_name, seed):
    """Process all profession pairs with the given model and return results"""
    results = []
    
    for pair in tqdm(pairs, desc=f"Processing {model_name} (seed {seed})"):
        profession = pair['profession']
        male_sent = pair['male_sentence']
        female_sent = pair['female_sentence']
        
        male_perplexity, male_tokens = calculate_perplexity(model, tokenizer, male_sent)
        female_perplexity, female_tokens = calculate_perplexity(model, tokenizer, female_sent)
        
        # Skip if perplexity calculation failed
        if male_perplexity is None or female_perplexity is None:
            print(f"Skipping profession {profession} due to perplexity calculation failure")
            continue
        
        bias_score = female_perplexity - male_perplexity
        
        results.append({
            'model': model_name,
            'seed': seed,
            'profession': profession,
            'male_profession': pair['male_profession'],
            'female_profession': pair['female_profession'],
            'male_sentence': male_sent,
            'female_sentence': female_sent,
            'male_perplexity': male_perplexity,
            'female_perplexity': female_perplexity,
            'bias_score': bias_score,
            'male_tokens': male_tokens,
            'female_tokens': female_tokens,
            'token_count_equal': male_tokens == female_tokens
        })
    
    print(f"Processed {len(results)} professions for {model_name} (seed {seed})")
    return pd.DataFrame(results) if results else None

# Find all profession pairs
print("Finding profession pairs...")
profession_pairs = find_gender_pairs(df)
print(f"Found {len(profession_pairs)} profession pairs with both genders")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Process all models across all seeds
all_results = []

for seed in seeds:
    print(f"\n{'='*50}")
    print(f"Processing Seed {seed}")
    print(f"{'='*50}")
    
    try:
        # Load and process baseline model (epoch 0)
        baseline_path = f"../models/{name_of_model}_checkpoints/random_seed_{seed}/finetuned_{name_of_model}_{seed}_epoch_0.pt"
        print(f"Loading baseline model from {baseline_path}...")
        
        baseline_model = AutoModelForMaskedLM.from_pretrained(base_model_id)
        checkpoint = torch.load(baseline_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            baseline_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            baseline_model.load_state_dict(checkpoint)
        
        baseline_model.to(device)
        baseline_model.eval()
        
        baseline_results = process_pairs(profession_pairs, baseline_model, tokenizer, "baseline", seed)
        if baseline_results is not None:
            all_results.append(baseline_results)
        
        # Clean up to free memory
        del baseline_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load and process finetuned model
        finetuned_path = f"../models/finetuned_{name_of_model}_{seed}_final.pt"
        print(f"Loading finetuned model from {finetuned_path}...")
        
        finetuned_model = AutoModelForMaskedLM.from_pretrained(base_model_id)
        checkpoint = torch.load(finetuned_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            finetuned_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            finetuned_model.load_state_dict(checkpoint)
        
        finetuned_model.to(device)
        finetuned_model.eval()
        
        finetuned_results = process_pairs(profession_pairs, finetuned_model, tokenizer, "finetuned", seed)
        if finetuned_results is not None:
            all_results.append(finetuned_results)
        
        # Clean up to free memory
        del finetuned_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"Error processing seed {seed}:")
        traceback.print_exc()
        continue

print("\nCombining results from all models and seeds...")
if not all_results:
    print("No results were collected. Exiting.")
    exit(1)

# Combine all results
combined_results = pd.concat(all_results)
combined_results.to_csv(f'./perplexity_measure_{typ}/{name_of_model}/all_seeds_bias_{typ}_{name_of_model}_results.csv', index=False)
print(f"Saved raw results with {len(combined_results)} rows")

# Calculate average results across seeds
print("Calculating averages across seeds...")
avg_results = combined_results.groupby(['model', 'profession']).agg({
    'male_perplexity': 'mean',
    'female_perplexity': 'mean',
    'bias_score': ['mean', 'std'],
    'male_profession': 'first',
    'female_profession': 'first',
    'male_sentence': 'first',
    'female_sentence': 'first'
}).reset_index()

# Flatten the multi-level columns
avg_results.columns = ['_'.join(col).strip('_') for col in avg_results.columns.values]

# Rename for clarity
avg_results = avg_results.rename(columns={
    'bias_score_mean': 'avg_bias_score',
    'bias_score_std': 'std_bias_score'
})

# Save averaged results
avg_results.to_csv(f'./perplexity_measure_{typ}/{name_of_model}/averaged_bias_results_{typ}_{name_of_model}.csv', index=False)
print("Saved averaged results")

# Create comparison of baseline vs fine-tuned
baseline_avg = avg_results[avg_results['model'] == 'baseline'][['profession', 'avg_bias_score', 'std_bias_score']]
finetuned_avg = avg_results[avg_results['model'] == 'finetuned'][['profession', 'avg_bias_score', 'std_bias_score']]

# Merge for comparison
comparison = pd.merge(
    baseline_avg, 
    finetuned_avg, 
    on='profession',
    suffixes=('_baseline', '_finetuned')
)

# Calculate bias change
comparison['bias_change'] = comparison['avg_bias_score_finetuned'] - comparison['avg_bias_score_baseline']
comparison.to_csv(f'./perplexity_measure_{typ}/{name_of_model}/bias_change_comparison_{typ}_{name_of_model}.csv', index=False)
print("Saved comparison results")

# Print summary statistics
print("\n----- Summary Statistics -----")
for model_name in ['baseline', 'finetuned']:
    model_data = avg_results[avg_results['model'] == model_name]
    
    print(f"\n{model_name.capitalize()} Model:")
    print(f"Average bias score: {model_data['avg_bias_score'].mean():.4f}")
    print(f"Median bias score: {model_data['avg_bias_score'].median():.4f}")
    
    # Top biased professions
    print(f"\nTop professions with higher female perplexity (biased against women) in {model_name}:")
    print(model_data.sort_values('avg_bias_score', ascending=False).head(5)[['profession', 'avg_bias_score', 'std_bias_score']])
    
    print(f"\nTop professions with higher male perplexity (biased against men) in {model_name}:")
    print(model_data.sort_values('avg_bias_score').head(5)[['profession', 'avg_bias_score', 'std_bias_score']])

# Bias change analysis
print("\nBias Change Analysis:")
print(f"Average change in bias: {comparison['bias_change'].mean():.4f}")
print(f"Median change in bias: {comparison['bias_change'].median():.4f}")

# Professions with greatest bias reduction
print("\nTop professions with greatest bias reduction:")
print(comparison.sort_values('bias_change').head(5)[['profession', 'avg_bias_score_baseline', 'avg_bias_score_finetuned', 'bias_change']])

# Professions with greatest bias increase
print("\nTop professions with greatest bias increase:")
print(comparison.sort_values('bias_change', ascending=False).head(5)[['profession', 'avg_bias_score_baseline', 'avg_bias_score_finetuned', 'bias_change']])



# Create visualizations
print("\nCreating visualizations...")

# Bias score distribution by model
plt.figure(figsize=(12, 8))

# Calculate unfiltered mean bias scores (including outliers) -- for google bert
baseline_mean_unfiltered = avg_results[avg_results['model'] == 'baseline']['avg_bias_score'].mean()
finetuned_mean_unfiltered = avg_results[avg_results['model'] == 'finetuned']['avg_bias_score'].mean()

# Filter out extreme outliers for visualization only
filtered_results = avg_results[
    (avg_results['avg_bias_score'] > -100) & 
    (avg_results['avg_bias_score'] < 100)
]

# Create the plot with the filtered data
baseline_data = filtered_results[filtered_results['model'] == 'baseline']
finetuned_data = filtered_results[filtered_results['model'] == 'finetuned']



# Create the plot with the filtered data
# baseline_data = avg_results[avg_results['model'] == 'baseline']
# finetuned_data = avg_results[avg_results['model'] == 'finetuned']

# baseline_mean = baseline_data['avg_bias_score'].mean()
# finetuned_mean = finetuned_data['avg_bias_score'].mean()

# Create KDE plots
sns.kdeplot(data=baseline_data, x='avg_bias_score', 
            fill=True, alpha=0.7, linewidth=2, color='blue', label='baseline')
sns.kdeplot(data=finetuned_data, x='avg_bias_score',
            fill=True, alpha=0.7, linewidth=2, color='orange', label='finetuned')

# Add a vertical line at x=0 (no bias reference)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='No bias')

# Add vertical dotted lines for the means (using ALL data, not just filtered)
plt.axvline(x=baseline_mean_unfiltered, color='blue', linestyle=':', linewidth=2, 
            label=f'Baseline avg: {baseline_mean_unfiltered:.2f}')
plt.axvline(x=finetuned_mean_unfiltered, color='orange', linestyle=':', linewidth=2, 
            label=f'Finetuned avg: {finetuned_mean_unfiltered:.2f}')

plt.title('Distribution of Bias Scores (Averaged Across Seeds)', fontsize=14)
plt.xlabel('Bias Score (Female-Male Perplexity Difference)', fontsize=12)
plt.ylabel('Density', fontsize=12)

# Add legend with explicit labels
plt.legend(title='Model', frameon=True)

# You might need to adjust the x-axis limits to see the distribution properly
plt.xlim(-250, 250)  # Adjust these values based on your data range

# Set the y-axis limit to 0.06
plt.ylim(0, 0.06)

plt.tight_layout()
plt.savefig(f'./perplexity_measure_{typ}/{name_of_model}/bias_distribution_comparison_{typ}_{name_of_model}.png', dpi=300)

# ## Create visualizations
# print("\nCreating visualizations...")

# # Bias score distribution by model
# plt.figure(figsize=(12, 8))

# # Filter out extreme outliers for better visualization
# filtered_results = avg_results[
#     (avg_results['avg_bias_score'] > -100) & 
#     (avg_results['avg_bias_score'] < 100)
# ]

# # Create the plot with the filtered data - add explicit labels
# baseline_data = filtered_results[filtered_results['model'] == 'baseline']
# finetuned_data = filtered_results[filtered_results['model'] == 'finetuned']

# # Calculate mean bias scores for baseline and finetuned models
# baseline_mean = baseline_data['avg_bias_score'].mean()
# finetuned_mean = finetuned_data['avg_bias_score'].mean()

# # Create KDE plots
# sns.kdeplot(data=baseline_data, x='avg_bias_score', 
#             fill=True, alpha=0.7, linewidth=2, color='blue', label=f'baseline (avg: {baseline_mean:.2f})')
# sns.kdeplot(data=finetuned_data, x='avg_bias_score',
#             fill=True, alpha=0.7, linewidth=2, color='orange', label=f'finetuned (avg: {finetuned_mean:.2f})')

# # Add a vertical line at x=0 (no bias reference)
# plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='No bias')

# # Add vertical dotted lines for the means
# plt.axvline(x=baseline_mean, color='blue', linestyle=':', linewidth=2, label=f'Baseline avg: {baseline_mean:.2f}')
# plt.axvline(x=finetuned_mean, color='orange', linestyle=':', linewidth=2, label=f'Finetuned avg: {finetuned_mean:.2f}')

# plt.title('Distribution of Bias Scores (Averaged Across Seeds)', fontsize=14)
# plt.xlabel('Bias Score (Female-Male Perplexity Difference)', fontsize=12)
# plt.ylabel('Density', fontsize=12)

# # Add legend with explicit labels
# plt.legend(title='Model', frameon=True)

# plt.tight_layout()
# plt.savefig(f'./perplexity_measure_{typ}/{name_of_model}/bias_distribution_comparison_{typ}_{name_of_model}.png', dpi=300)



# # Top professions with bias change
# top_increase = comparison.sort_values('bias_change', ascending=False).head(10)
# top_decrease = comparison.sort_values('bias_change').head(10)

# plt.figure(figsize=(12, 8))
# sns.barplot(data=top_increase, x='bias_change', y='profession')
# plt.title('Top 10 Professions with Increased Bias After Fine-tuning')
# plt.xlabel('Bias Change (Positive = More Female Bias)')
# plt.tight_layout()
# plt.savefig(f'./perplexity_measure_{typ}/{name_of_model}/top_bias_increase_{typ}_{name_of_model}.png')

# plt.figure(figsize=(12, 8))
# sns.barplot(data=top_decrease, x='bias_change', y='profession')
# plt.title('Top 10 Professions with Decreased Bias After Fine-tuning')
# plt.xlabel('Bias Change (Negative = Less Female Bias)')
# plt.tight_layout()
# plt.savefig(f'./perplexity_measure_{typ}/{name_of_model}/top_bias_decrease_{typ}_{name_of_model}.png')

# Create a graph to compare male and female perplexity values
print("\nCreating perplexity comparison bar graph...")

# Prepare data for comparison plot
# First get average perplexity values across all seeds
perplexity_data = combined_results.groupby(['model']).agg({
    'male_perplexity': 'mean',
    'female_perplexity': 'mean'
}).reset_index()

# Convert to format suitable for plotting
plot_data = pd.DataFrame({
    'Model': ['Baseline', 'Baseline', 'Fine-tuned', 'Fine-tuned'],
    'Gender': ['Male', 'Female', 'Male', 'Female'],
    'Perplexity': [
        perplexity_data[perplexity_data['model'] == 'baseline']['male_perplexity'].values[0],
        perplexity_data[perplexity_data['model'] == 'baseline']['female_perplexity'].values[0],
        perplexity_data[perplexity_data['model'] == 'finetuned']['male_perplexity'].values[0],
        perplexity_data[perplexity_data['model'] == 'finetuned']['female_perplexity'].values[0]
    ]
})

# Create the bar plot
plt.figure(figsize=(10, 6))
gender_palette = {'Male': 'blue', 'Female': 'orange'}
bar_plot = sns.barplot(x='Model', y='Perplexity', hue='Gender', data=plot_data, palette=gender_palette, width=0.6)

# Add labels and title
plt.title('Comparison of Male vs Female Perplexity (Averaged Across All Professions and Seeds)')
plt.xlabel('Model')
plt.ylabel('Average Perplexity')
# Set y-axis limit to 300
#plt.ylim(0, 300)

# Add value labels on the bars
for i, bar in enumerate(bar_plot.patches):
    bar_plot.text(
        bar.get_x() + bar.get_width()/2.,
        bar.get_height() + 0.1,
        f'{bar.get_height():.2f}',
        ha='center',
        fontsize=10
    )

plt.tight_layout()
plt.savefig(f'./perplexity_measure_{typ}/{name_of_model}/male_vs_female_perplexity_comparison_{typ}_{name_of_model}.png')

# Also create a more detailed plot showing perplexity for each profession
print("Creating detailed profession-level perplexity comparison...")

# Get average perplexity values for each profession across seeds
profession_perplexity = combined_results.groupby(['model', 'profession']).agg({
    'male_perplexity': 'mean',
    'female_perplexity': 'mean'
}).reset_index()

# Reshape the data for easier plotting
profession_plot_data = []

for _, row in profession_perplexity.iterrows():
    profession_plot_data.append({
        'Model': 'Baseline' if row['model'] == 'baseline' else 'Fine-tuned',
        'Profession': row['profession'],
        'Gender': 'Male',
        'Perplexity': row['male_perplexity']
    })
    
    profession_plot_data.append({
        'Model': 'Baseline' if row['model'] == 'baseline' else 'Fine-tuned',
        'Profession': row['profession'],
        'Gender': 'Female',
        'Perplexity': row['female_perplexity']
    })

profession_plot_df = pd.DataFrame(profession_plot_data)

# Select top 10 professions with highest average perplexity for plotting
# (otherwise the plot would be too crowded)
top_professions = profession_plot_df.groupby('Profession')['Perplexity'].mean().nlargest(10).index.tolist()
top_professions_data = profession_plot_df[profession_plot_df['Profession'].isin(top_professions)]

# Create the detailed plot
plt.figure(figsize=(15, 10))
gender_palette = {'Male': 'blue', 'Female': 'orange'}
sns.barplot(
    x='Profession',
    y='Perplexity',
    hue='Gender',
    data=top_professions_data,
    palette=gender_palette,
    errorbar=None,
    dodge=True,
    width=0.7
)

# Add plot elements
plt.title('Male vs Female Perplexity for Top 10 Professions')
plt.xlabel('Profession')
plt.ylabel('Perplexity')
# Set y-axis limit to 600
#plt.ylim(0, 600)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Gender')
plt.tight_layout()
plt.savefig(f'./perplexity_measure_{typ}/{name_of_model}/profession_perplexity_comparison_{typ}_{name_of_model}.png')

print("Perplexity comparison graphs created successfully!")


print("\nAnalysis complete! Results saved to CSV files and visualizations.")