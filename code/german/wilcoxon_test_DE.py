import pandas as pd
from scipy.stats import wilcoxon
import numpy as np
import sys
import os

# typ = "star"

# Define output file path
output_file_path = f'../../data/statistics/german/gender_neutral/statistics_DE_pre_all_models_gender_neutral_avg.txt'

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Redirect stdout to file
original_stdout = sys.stdout
sys.stdout = open(output_file_path, 'w')

# Read the CSV file
data = pd.read_csv('../../data/output_csv_files/german/gender_neutral/pre_post_assoc_all_models_DE_gender_neutral_avg.csv')

# List of models to analyze
models = ['dbmdz', 'google_bert', 'deepset_bert', 'distilbert']

# Function to perform analysis for a specific model and pre-association column
def analyze_model(data, model_name, pre_assoc_column):
    print(f"\n----- ANALYSIS FOR {model_name.upper()} MODEL -----\n")
    
    # Extract data for female gender within statistically female professions
    female_gender_in_female_professions = data.loc[
        (data['Prof_Gender'] == 'female') & (data['Gender'] == 'female'), 
        pre_assoc_column
    ].tolist()

    # Extract data for male gender within statistically female professions
    male_gender_in_female_professions = data.loc[
        (data['Prof_Gender'] == 'female') & (data['Gender'] == 'male'), 
        pre_assoc_column
    ].tolist()

    # Extract data for female gender within statistically male professions
    female_gender_in_male_professions = data.loc[
        (data['Prof_Gender'] == 'male') & (data['Gender'] == 'female'), 
        pre_assoc_column
    ].tolist()

    # Extract data for male gender within statistically male professions
    male_gender_in_male_professions = data.loc[
        (data['Prof_Gender'] == 'male') & (data['Gender'] == 'male'), 
        pre_assoc_column
    ].tolist()

    # Extract data for female gender within statistically balanced professions
    female_gender_in_balanced_professions = data.loc[
        (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'female'), 
        pre_assoc_column
    ].tolist()

    # Extract data for male gender within statistically balanced professions
    male_gender_in_balanced_professions = data.loc[
        (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'male'), 
        pre_assoc_column
    ].tolist()

    # Compute means
    print("MEAN VALUES:")
    female_gender_in_female_professions_mean = np.mean(female_gender_in_female_professions)
    print(f'Mean for female gender within statistically female professions: {female_gender_in_female_professions_mean:.4f}')

    male_gender_in_female_professions_mean = np.mean(male_gender_in_female_professions)
    print(f'Mean for male gender within statistically female professions: {male_gender_in_female_professions_mean:.4f}')

    female_gender_in_male_professions_mean = np.mean(female_gender_in_male_professions)
    print(f'Mean for female gender within statistically male professions: {female_gender_in_male_professions_mean:.4f}')

    male_gender_in_male_professions_mean = np.mean(male_gender_in_male_professions)
    print(f'Mean for male gender within statistically male professions: {male_gender_in_male_professions_mean:.4f}')

    female_gender_in_balanced_professions_mean = np.mean(female_gender_in_balanced_professions)
    print(f'Mean for female gender within statistically balanced professions: {female_gender_in_balanced_professions_mean:.4f}')

    male_gender_in_balanced_professions_mean = np.mean(male_gender_in_balanced_professions)
    print(f'Mean for male gender within statistically balanced professions: {male_gender_in_balanced_professions_mean:.4f}')

    # Compute standard deviations
    print("\nSTANDARD DEVIATIONS:")
    sd_female_female = np.std(female_gender_in_female_professions)
    print(f"SD for Female Professions - Female Person Words: {sd_female_female:.4f}")
    
    sd_female_male = np.std(male_gender_in_female_professions)
    print(f"SD for Female Professions - Male Person Words: {sd_female_male:.4f}")
    
    sd_male_female = np.std(female_gender_in_male_professions)
    print(f"SD for Male Professions - Female Person Words: {sd_male_female:.4f}")
    
    sd_male_male = np.std(male_gender_in_male_professions)
    print(f"SD for Male Professions - Male Person Words: {sd_male_male:.4f}")
    
    sd_balanced_female = np.std(female_gender_in_balanced_professions)
    print(f"SD for Balanced Professions - Female Person Words: {sd_balanced_female:.4f}")
    
    sd_balanced_male = np.std(male_gender_in_balanced_professions)
    print(f"SD for Balanced Professions - Male Person Words: {sd_balanced_male:.4f}")

    # Perform Wilcoxon tests
    print("\nWILCOXON TESTS:")
    
    # For female professions
    stat_F, p_value_F = wilcoxon(male_gender_in_female_professions, female_gender_in_female_professions, alternative='two-sided')
    print(f"Female Professions - Wilcoxon test statistic: {stat_F}")
    print(f"Female Professions - P-value: {p_value_F:.6f}")
    if p_value_F < 0.05:
        print("Female Professions - Reject null hypothesis: There is a significant difference.\n")
    else:
        print("Female Professions - Fail to reject null hypothesis: No significant difference.\n")

    # For male professions
    stat_M, p_value_M = wilcoxon(male_gender_in_male_professions, female_gender_in_male_professions, alternative='two-sided')
    print(f"Male Professions - Wilcoxon test statistic: {stat_M}")
    print(f"Male Professions - P-value: {p_value_M:.6f}")
    if p_value_M < 0.05:
        print("Male Professions - Reject null hypothesis: There is a significant difference.\n")
    else:
        print("Male Professions - Fail to reject null hypothesis: No significant difference.\n")

    # For balanced professions
    stat_B, p_value_B = wilcoxon(male_gender_in_balanced_professions, female_gender_in_balanced_professions, alternative='two-sided')
    print(f"Balanced Professions - Wilcoxon test statistic: {stat_B}")
    print(f"Balanced Professions - P-value: {p_value_B:.6f}")
    if p_value_B < 0.05:
        print("Balanced Professions - Reject null hypothesis: There is a significant difference.\n")
    else:
        print("Balanced Professions - Fail to reject null hypothesis: No significant difference.\n")

# Analyze each model separately
for model in models:
    pre_assoc_column = f'{model}_Pre_Assoc_Avg'
    if pre_assoc_column in data.columns:
        analyze_model(data, model, pre_assoc_column)
    else:
        print(f"\nWarning: {pre_assoc_column} not found in the data!")

print("\nAnalysis complete for all models.")

# Restore original stdout
sys.stdout.close()
sys.stdout = original_stdout
print(f"Analysis complete! Results written to: {output_file_path}")