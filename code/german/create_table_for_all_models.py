import pandas as pd
from scipy.stats import wilcoxon
import math
import numpy as np

# Modified version to use Checkpoint associations instead of Pre associations
# Read the CSV file
data = pd.read_csv('../../data/output_csv_files/german/pre_assoc_all_models_DE_zero_difference_387.csv', sep=',')

# For each model, calculate statistics using Checkpoint_Assoc instead of Pre_Assoc
models = ['dbmdz', 'google_bert', 'deepset_bert', 'distilbert', 'gelectra']

# Dictionary to store results for each model
model_results = {}

for model in models:
    # Column name for Checkpoint associations for this model
    assoc_col = f'{model}_Checkpoint_Assoc'
    
    # Skip if the column doesn't exist
    if assoc_col not in data.columns:
        print(f"Column {assoc_col} not found, skipping model {model}")
        continue
    
    print(f"\n--- Results for {model} ---")
    
    # female gender within statistically female professions
    female_gender_in_female_professions = data.loc[(
        data['Prof_Gender'] == 'female') & (data['Gender'] == 'female'), assoc_col].tolist()

    # male gender within statistically female professions
    male_gender_in_female_professions = data.loc[(
        data['Prof_Gender'] == 'female') & (data['Gender'] == 'male'), assoc_col].tolist()

    # female gender within statistically male professions
    female_gender_in_male_professions = data.loc[
        (data['Prof_Gender'] == 'male') & (data['Gender'] == 'female'), assoc_col].tolist()

    # male gender within statistically male professions
    male_gender_in_male_professions = data.loc[
        (data['Prof_Gender'] == 'male') & (data['Gender'] == 'male'), assoc_col].tolist()

    # female gender within statistically balanced professions
    female_gender_in_balanced_professions = data.loc[
        (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'female'), assoc_col].tolist()

    # male gender within statistically balanced professions
    male_gender_in_balanced_professions = data.loc[
        (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'male'), assoc_col].tolist()

    # Compute means
    female_gender_in_female_professions_mean = data.loc[(
        data['Prof_Gender'] == 'female') & (data['Gender'] == 'female'), assoc_col].mean()
    print(f'Mean for female gender within statistically female professions: {female_gender_in_female_professions_mean:.2f}')

    male_gender_in_female_professions_mean = data.loc[(
        data['Prof_Gender'] == 'female') & (data['Gender'] == 'male'), assoc_col].mean()
    print(f'Mean for male gender within statistically female professions: {male_gender_in_female_professions_mean:.2f}')

    female_gender_in_male_professions_mean = data.loc[
        (data['Prof_Gender'] == 'male') & (data['Gender'] == 'female'),
        assoc_col
    ].mean()
    print(f'Mean for female gender within statistically male professions: {female_gender_in_male_professions_mean:.2f}')

    male_gender_in_male_professions_mean = data.loc[
        (data['Prof_Gender'] == 'male') & (data['Gender'] == 'male'),
        assoc_col
    ].mean()
    print(f'Mean for male gender within statistically male professions: {male_gender_in_male_professions_mean:.2f}')

    female_gender_in_balanced_professions_mean = data.loc[
        (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'female'),
        assoc_col
    ].mean()
    print(f'Mean for female gender within statistically balanced professions: {female_gender_in_balanced_professions_mean:.2f}')

    male_gender_in_balanced_professions_mean = data.loc[
        (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'male'),
        assoc_col
    ].mean()
    print(f'Mean for male gender within statistically balanced professions: {male_gender_in_balanced_professions_mean:.2f}')

    # Store results for this model
    model_results[model] = {
        'B': {
            'f': female_gender_in_balanced_professions_mean,
            'm': male_gender_in_balanced_professions_mean
        },
        'F': {
            'f': female_gender_in_female_professions_mean,
            'm': male_gender_in_female_professions_mean
        },
        'M': {
            'f': female_gender_in_male_professions_mean,
            'm': male_gender_in_male_professions_mean
        }
    }
    
    # Perform Wilcoxon tests
    # Test for Female Professions (F)
    if len(male_gender_in_female_professions) == len(female_gender_in_female_professions):
        stat_F, p_value_F = wilcoxon(male_gender_in_female_professions, female_gender_in_female_professions, alternative='two-sided')
        print(f"\nWilcoxon test statistic for F: {stat_F}")
        print(f"P-value: {p_value_F}")
        
        # Interpret the results
        alpha = 0.05  # significance level
        if p_value_F < alpha:
            print("Reject the null hypothesis: There is a significant difference.")
        else:
            print("Fail to reject the null hypothesis: No significant difference.")
    else:
        print("Cannot perform Wilcoxon test for F: lists have different lengths")

    # Test for Male Professions (M)
    if len(male_gender_in_male_professions) == len(female_gender_in_male_professions):
        stat_M, p_value_M = wilcoxon(male_gender_in_male_professions, female_gender_in_male_professions, alternative='two-sided')
        print(f"\nWilcoxon test statistic for M: {stat_M}")
        print(f"P-value: {p_value_M}")
        
        # Interpret the results
        alpha = 0.05  # significance level
        if p_value_M < alpha:
            print("Reject the null hypothesis: There is a significant difference.")
        else:
            print("Fail to reject the null hypothesis: No significant difference.")
    else:
        print("Cannot perform Wilcoxon test for M: lists have different lengths")

    # Test for Balanced Professions (B)
    if len(male_gender_in_balanced_professions) == len(female_gender_in_balanced_professions):
        stat_B, p_value_B = wilcoxon(male_gender_in_balanced_professions, female_gender_in_balanced_professions, alternative='two-sided')
        print(f"\nWilcoxon test statistic for B: {stat_B}")
        print(f"P-value: {p_value_B}")
        
        # Interpret the results
        alpha = 0.05  # significance level
        if p_value_B < alpha:
            print("Reject the null hypothesis: There is a significant difference.")
        else:
            print("Fail to reject the null hypothesis: No significant difference.")
    else:
        print("Cannot perform Wilcoxon test for B: lists have different lengths")

# Just print the individual numbers for updating the table
print("\n\nCheckpoint Association Values (for updating table):")
print("=" * 50)

# Define the order to match your table
jobs_order = ['B', 'F', 'M']
person_order = ['f', 'm']
models_order = ['dbmdz', 'google_bert', 'deepset_bert', 'distilbert', 'gelectra']
model_display = ['dbmdz', 'google-bert', 'gbert', 'distilbert', 'gelectra']

# Print values for each job/person combination by model
for job in jobs_order:
    for person in person_order:
        print(f"Job: {job}, Person: {person}")
        for model, display in zip(models_order, model_display):
            if model in model_results:
                value = model_results[model][job][person]
                print(f"  {display}: {value:.2f}")
            else:
                print(f"  {display}: N/A")
        print("")

print("=" * 50)