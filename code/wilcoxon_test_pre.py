"""
Statistical Analysis of Gender Bias in BERT Models

This script performs comprehensive statistical analysis on gender bias measurements 
across multiple BERT models using the Wilcoxon signed-rank test. It analyzes
pre-association scores between gendered person words and professions categorized by
their statistical gender distribution (male-dominated, female-dominated, balanced).

Key Features:
- Analyzes 4 German BERT models: dbmdz, google-bert, deepset-bert, distilbert
- Compares gender pre-associations within profession categories (male/female/balanced)
- Performs Wilcoxon signed-rank tests to detect significant bias differences
- Calculates descriptive statistics (means, standard deviations)
- Computes effect sizes for statistical significance
- Outputs detailed results to structured text files

The analysis examines whether models show systematic bias by comparing how strongly
they associate male vs. female person words with professions of different gender
stereotypes. Results help quantify the extent of gender bias present in each model.

Input: Tab-separated CSV files with pre-computed association scores
Output: Statistical analysis report saved to results/statistics/ directory
"""

import pandas as pd
from scipy.stats import wilcoxon
import numpy as np
import sys
import os
import math

typ = "regular"

# Define output file path
output_file_path = f'../results/statistics/german/Lou/{typ}/statistics_Lou_pre_all_models_{typ}_both_adapted.txt'

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Redirect stdout to file
original_stdout = sys.stdout
sys.stdout = open(output_file_path, 'w')

# Read the CSV file
data = pd.read_csv(f'../results/association_files/german/Lou/regular/results_Lou_all_models_replicated_both_{typ}_adapted.csv', sep="\t")

# Print available columns for debugging
print("Available columns in the dataset:")
print(data.columns.tolist())
print("\n")

# Dictionary mapping model names to their corresponding column names
models = {
    "dbmdz": "Pre_Assoc_dbmdz_Avg",
    "google-bert": "Pre_Assoc_google-bert_Avg",
    "deepset-bert": "Pre_Assoc_deepset-bert_Avg",
    "distilbert": "Pre_Assoc_distilbert_Avg"
}

# Function to compute Cohen's W effect size
def compute_cohens_w(wilcoxon_statistic, N):
    """
    Computes the effect size for the Wilcoxon Signed-Rank Test using Cohen's W.
    """
    # Step 1: Compute expected W (E(W))
    E_W = (N * (N + 1)) / 4

    # Step 2: Compute the standard error of W (SE(W))
    SE_W = math.sqrt((N * (N + 1) * (2 * N + 1)) / 24)

    # Step 3: Compute the Z-score
    Z_value = (wilcoxon_statistic - E_W) / SE_W

    # Step 4: Compute Cohen's W effect size
    cohens_w = Z_value / math.sqrt(N)

    return cohens_w

# Function to perform analysis for a specific model and pre-association column
def analyze_model(data, model_name, pre_assoc_column):
    print(f"\n----- ANALYSIS FOR {model_name.upper()} MODEL -----\n")
    print(f"Using column: {pre_assoc_column}")
    
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

    # Print sample counts
    print("SAMPLE COUNTS:")
    print(f"Female gender in female professions: {len(female_gender_in_female_professions)}")
    print(f"Male gender in female professions: {len(male_gender_in_female_professions)}")
    print(f"Female gender in male professions: {len(female_gender_in_male_professions)}")
    print(f"Male gender in male professions: {len(male_gender_in_male_professions)}")
    print(f"Female gender in balanced professions: {len(female_gender_in_balanced_professions)}")
    print(f"Male gender in balanced professions: {len(male_gender_in_balanced_professions)}")

    # Compute means
    print("\nMEAN VALUES:")
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
    # Calculate effect size (r) for female professions
    n_F = len(male_gender_in_female_professions)  # number of pairs
    r_F = stat_F / (n_F * (n_F + 1) / 2)  # divide by max possible sum
    r_F = abs(r_F - 0.5) * 2  # transform to -1 to 1 scale
    print(f"Female Professions - Wilcoxon test statistic: {stat_F}")
    print(f"Female Professions - P-value: {p_value_F:.2e}")  # Scientific notation to handle small values
    print(f"Female Professions - Effect size (r): {r_F:.2f}")
    if p_value_F < 0.05:
        print("Female Professions - Reject null hypothesis: There is a significant difference.\n")
    else:
        print("Female Professions - Fail to reject null hypothesis: No significant difference.\n")

    # For male professions
    stat_M, p_value_M = wilcoxon(male_gender_in_male_professions, female_gender_in_male_professions, alternative='two-sided')
    # Calculate effect size (r) for male professions
    n_M = len(male_gender_in_male_professions)
    r_M = stat_M / (n_M * (n_M + 1) / 2)
    r_M = abs(r_M - 0.5) * 2
    print(f"Male Professions - Wilcoxon test statistic: {stat_M}")
    print(f"Male Professions - P-value: {p_value_M:.2e}")
    print(f"Male Professions - Effect size (r): {r_M:.2f}")
    if p_value_M < 0.05:
        print("Male Professions - Reject null hypothesis: There is a significant difference.\n")
    else:
        print("Male Professions - Fail to reject null hypothesis: No significant difference.\n")

    # For balanced professions
    stat_B, p_value_B = wilcoxon(male_gender_in_balanced_professions, female_gender_in_balanced_professions, alternative='two-sided')
    # Calculate effect size (r) for balanced professions
    n_B = len(male_gender_in_balanced_professions)
    r_B = stat_B / (n_B * (n_B + 1) / 2)
    r_B = abs(r_B - 0.5) * 2
    print(f"Balanced Professions - Wilcoxon test statistic: {stat_B}")
    print(f"Balanced Professions - P-value: {p_value_B:.2e}")
    print(f"Balanced Professions - Effect size (r): {r_B:.2f}")
    if p_value_B < 0.05:
        print("Balanced Professions - Reject null hypothesis: There is a significant difference.\n")
    else:
        print("Balanced Professions - Fail to reject null hypothesis: No significant difference.\n")

# Analyze each model separately
for model_name, pre_assoc_column in models.items():
    if pre_assoc_column in data.columns:
        analyze_model(data, model_name, pre_assoc_column)
    else:
        print(f"\nWarning: {pre_assoc_column} not found in the data!")
        print(f"Available columns: {[col for col in data.columns if 'Pre_Assoc' in col]}")

print("\nAnalysis complete for all models.")

# Restore original stdout
sys.stdout.close()
sys.stdout = original_stdout
print(f"Analysis complete! Results written to: {output_file_path}")