"""
Pre-Post Bias Mitigation Statistical Analysis

This script performs comprehensive statistical analysis comparing gender bias 
measurements before and after applying bias mitigation techniques to both English 
and German BERT models. It evaluates the effectiveness of debiasing interventions 
using paired statistical tests.

Key Features:
- Analyzes English BERT (bert-base-uncased) and German BERT models 
  (dbmdz, google-bert, deepset-bert, distilbert)
- Compares pre-association vs. post-association scores using Wilcoxon signed-rank tests
- Evaluates bias changes across profession categories (male/female/balanced)
- Calculates descriptive statistics (means, standard deviations) for pre/post conditions
- Computes Cohen's W effect sizes to quantify magnitude of bias reduction
- Generates detailed statistical reports with significance testing results

Analysis Structure:
- Pre-association statistics: Baseline bias measurements before intervention
- Post-association statistics: Bias measurements after debiasing treatment
- Statistical tests: Wilcoxon tests for each profession category and overall
- Effect sizes: Cohen's W calculations for practical significance assessment
- Summary: Comparative analysis of bias reduction across all models

The script outputs comprehensive statistical reports showing whether bias mitigation
techniques successfully reduced gender stereotypical associations in the models,
providing quantitative evidence for the effectiveness of debiasing interventions.

Input: Association score CSV files with pre/post intervention measurements
Output: Detailed statistical analysis reports saved to results/statistics/ directories
"""

import pandas as pd
from scipy.stats import wilcoxon
import math
import os

############# FOR ENGLISH BERT ####################################

# Create output directory for statistics
output_file_path = f"../results/statistics/english/"
# Create directory if it doesn't exist
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Read a CSV file
data = pd.read_csv(f'../results/association_files/english/results_bert_replicated_adapted.csv', sep="\t")

models = ["bert"]

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

# Open a file to write all statistics
with open(os.path.join(output_file_path, f"statistics_bert_replicated_post_adapted.txt"), "w") as stats_file:
    # For each model
    for name in models:
        # Check if the model has both pre and post association scores
        pre_col = f'Pre_Assoc_{name}_Avg'
        post_col = f'Post_Assoc_{name}_Avg'
            
        if pre_col not in data.columns or post_col not in data.columns:
            stats_file.write(f"\n\n{'='*50}\n")
            stats_file.write(f"Model: {name}\n")
            stats_file.write(f"{'='*50}\n")
            stats_file.write(f"Missing pre or post association scores for this model.\n")
            continue
            
        # Write header for this model
        stats_file.write(f"\n\n{'='*50}\n")
        stats_file.write(f"Model: {name}\n")
        stats_file.write(f"{'='*50}\n")
        
        # Get pre and post association scores
        pre_association = data[pre_col].to_list()
        post_association = data[post_col].to_list()

        # Get profession gender-specific scores
        female_professions_pre = data.loc[(
                data['Prof_Gender'] == 'female'), pre_col].tolist()
        male_professions_pre = data.loc[(
            data['Prof_Gender'] == 'male'), pre_col].tolist()
        balanced_professions_pre = data.loc[(
            data['Prof_Gender'] == 'balanced'), pre_col].tolist()

        female_professions_post = data.loc[(
            data['Prof_Gender'] == 'female'), post_col].tolist()
        male_professions_post = data.loc[(
            data['Prof_Gender'] == 'male'), post_col].tolist()
        balanced_professions_post = data.loc[(
            data['Prof_Gender'] == 'balanced'), post_col].tolist()

        # PRE-ASSOCIATION STATISTICS
        stats_file.write("PRE-ASSOCIATION STATISTICS\n")
        stats_file.write("-------------------------\n")
        
        # Female gender within statistically female professions
        female_gender_in_female_professions_mean = data.loc[(
            data['Prof_Gender'] == 'female') & (data['Gender'] == 'female'), pre_col].mean()
        stats_file.write(f'Mean for female gender within statistically female professions: {female_gender_in_female_professions_mean}\n')

        # Male gender within statistically female professions
        male_gender_in_female_professions_mean = data.loc[(
            data['Prof_Gender'] == 'female') & (data['Gender'] == 'male'), pre_col].mean()
        stats_file.write(f'Mean for male gender within statistically female professions: {male_gender_in_female_professions_mean}\n')

        # Female gender within statistically male professions
        female_gender_in_male_professions_mean = data.loc[
            (data['Prof_Gender'] == 'male') & (data['Gender'] == 'female'),
            pre_col
        ].mean()
        stats_file.write(f'Mean for female gender within statistically male professions: {female_gender_in_male_professions_mean}\n')

        # Male gender within statistically male professions
        male_gender_in_male_professions_mean = data.loc[
            (data['Prof_Gender'] == 'male') & (data['Gender'] == 'male'),
            pre_col
        ].mean()
        stats_file.write(f'Mean for male gender within statistically male professions: {male_gender_in_male_professions_mean}\n')

        # Female gender within statistically balanced professions
        female_gender_in_balanced_professions_mean = data.loc[
            (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'female'),
            pre_col
        ].mean()
        stats_file.write(f'Mean for female gender within statistically balanced professions: {female_gender_in_balanced_professions_mean}\n')

        # Male gender within statistically balanced professions
        male_gender_in_balanced_professions_mean = data.loc[
            (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'male'),
            pre_col
        ].mean()
        stats_file.write(f'Mean for male gender within statistically balanced professions: {male_gender_in_balanced_professions_mean}\n')

        # POST-ASSOCIATION STATISTICS
        stats_file.write("\nPOST-ASSOCIATION STATISTICS\n")
        stats_file.write("--------------------------\n")
        
        # Female gender within statistically female professions
        female_gender_in_female_professions_post_mean = data.loc[(
            data['Prof_Gender'] == 'female') & (data['Gender'] == 'female'), post_col].mean()
        stats_file.write(f'Mean for female gender within statistically female professions: {female_gender_in_female_professions_post_mean}\n')

        # Male gender within statistically female professions
        male_gender_in_female_professions_post_mean = data.loc[(
            data['Prof_Gender'] == 'female') & (data['Gender'] == 'male'), post_col].mean()
        stats_file.write(f'Mean for male gender within statistically female professions: {male_gender_in_female_professions_post_mean}\n')

        # Female gender within statistically male professions
        female_gender_in_male_professions_post_mean = data.loc[
            (data['Prof_Gender'] == 'male') & (data['Gender'] == 'female'),
            post_col
        ].mean()
        stats_file.write(f'Mean for female gender within statistically male professions: {female_gender_in_male_professions_post_mean}\n')

        # Male gender within statistically male professions
        male_gender_in_male_professions_post_mean = data.loc[
            (data['Prof_Gender'] == 'male') & (data['Gender'] == 'male'),
            post_col
        ].mean()
        stats_file.write(f'Mean for male gender within statistically male professions: {male_gender_in_male_professions_post_mean}\n')

        # Female gender within statistically balanced professions
        female_gender_in_balanced_professions_post_mean = data.loc[
            (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'female'),
            post_col
        ].mean()
        stats_file.write(f'Mean for female gender within statistically balanced professions: {female_gender_in_balanced_professions_post_mean}\n')

        # Male gender within statistically balanced professions
        male_gender_in_balanced_professions_post_mean = data.loc[
            (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'male'),
            post_col
        ].mean()
        stats_file.write(f'Mean for male gender within statistically balanced professions: {male_gender_in_balanced_professions_post_mean}\n')

        # STATISTICAL TESTS
        stats_file.write("\nSTATISTICAL TESTS\n")
        stats_file.write("----------------\n")
        
        # Overall Wilcoxon test
        stat, p_value = wilcoxon(pre_association, post_association, alternative='two-sided')
        stats_file.write(f"Overall Wilcoxon test statistic: {stat}\n")
        stats_file.write(f"P-value: {p_value}\n")
        
        alpha = 0.05  # significance level
        if p_value < alpha:
            stats_file.write("Reject the null hypothesis: There is a significant difference.\n")
        else:
            stats_file.write("Fail to reject the null hypothesis: No significant difference.\n")

        # Female professions Wilcoxon test
        if len(female_professions_pre) > 0 and len(female_professions_post) > 0:
            stat_F, p_value_F = wilcoxon(female_professions_pre, female_professions_post, alternative='two-sided')
            stats_file.write(f"\nWilcoxon test statistic for female professions (F): {stat_F}\n")
            stats_file.write(f"P-value: {p_value_F}\n")
            
            if p_value_F < alpha:
                stats_file.write("Reject the null hypothesis: There is a significant difference.\n")
            else:
                stats_file.write("Fail to reject the null hypothesis: No significant difference.\n")
        else:
            stats_file.write("\nNot enough data for female professions to perform Wilcoxon test.\n")
            stat_F, p_value_F = 0, 1  # Default values for later use

        # Male professions Wilcoxon test
        if len(male_professions_pre) > 0 and len(male_professions_post) > 0:
            stat_M, p_value_M = wilcoxon(male_professions_pre, male_professions_post, alternative='two-sided')
            stats_file.write(f"\nWilcoxon test statistic for male professions (M): {stat_M}\n")
            stats_file.write(f"P-value: {p_value_M}\n")
            
            if p_value_M < alpha:
                stats_file.write("Reject the null hypothesis: There is a significant difference.\n")
            else:
                stats_file.write("Fail to reject the null hypothesis: No significant difference.\n")
        else:
            stats_file.write("\nNot enough data for male professions to perform Wilcoxon test.\n")
            stat_M, p_value_M = 0, 1  # Default values for later use

        # Balanced professions Wilcoxon test
        if len(balanced_professions_pre) > 0 and len(balanced_professions_post) > 0:
            stat_B, p_value_B = wilcoxon(balanced_professions_pre, balanced_professions_post, alternative='two-sided')
            stats_file.write(f"\nWilcoxon test statistic for balanced professions (B): {stat_B}\n")
            stats_file.write(f"P-value: {p_value_B}\n")
            
            if p_value_B < alpha:
                stats_file.write("Reject the null hypothesis: There is a significant difference.\n")
            else:
                stats_file.write("Fail to reject the null hypothesis: No significant difference.\n")
        else:
            stats_file.write("\nNot enough data for balanced professions to perform Wilcoxon test.\n")
            stat_B, p_value_B = 0, 1  # Default values for later use

        # EFFECT SIZES
        stats_file.write("\nEFFECT SIZES\n")
        stats_file.write("-----------\n")
    
        # Number of samples (N)
        N_ALL = len(pre_association)
        N_F = len(female_professions_pre) 
        N_M = len(male_professions_pre)
        N_B = len(balanced_professions_pre)

        # Add ALL to the results dictionary
        groups = {
            "all": {"wilcoxon_statistic": stat, "n": N_ALL},
            "female": {"wilcoxon_statistic": stat_F, "n": N_F},
            "male": {"wilcoxon_statistic": stat_M, "n": N_M},
            "balanced": {"wilcoxon_statistic": stat_B, "n": N_B}
        }
        
        for group, values in groups.items():
            wilcoxon_statistic = values["wilcoxon_statistic"]
            group_n = values["n"]
            if group_n > 0:  # Avoid division by zero
                effect_size = compute_cohens_w(wilcoxon_statistic, group_n)
                stats_file.write(f"{group.capitalize()} group - Cohen's W effect size: {effect_size}\n")
            else:
                stats_file.write(f"{group.capitalize()} group - Not enough data to compute effect size\n")

    # Write a summary at the end
    stats_file.write(f"\n\n{'='*50}\n")
    stats_file.write("SUMMARY OF MODELS\n")
    stats_file.write(f"{'='*50}\n")

    for name in models:
        pre_col = f'Pre_Assoc_{name}_Avg'
        post_col = f'Post_Assoc_{name}_Avg'
            
        if pre_col in data.columns and post_col in data.columns:
            # Overall means
            pre_mean = data[pre_col].mean()
            pre_std = data[pre_col].std()
            post_mean = data[post_col].mean()
            post_std = data[post_col].std()
            
            # Female person words
            female_pre_mean = data.loc[data['Gender'] == 'female', pre_col].mean()
            female_pre_std = data.loc[data['Gender'] == 'female', pre_col].std()
            female_post_mean = data.loc[data['Gender'] == 'female', post_col].mean()
            female_post_std = data.loc[data['Gender'] == 'female', post_col].std()
            
            # Male person words
            male_pre_mean = data.loc[data['Gender'] == 'male', pre_col].mean()
            male_pre_std = data.loc[data['Gender'] == 'male', pre_col].std() 
            male_post_mean = data.loc[data['Gender'] == 'male', post_col].mean()
            male_post_std = data.loc[data['Gender'] == 'male', post_col].std()
            
            stats_file.write(f"\nModel: {name}\n")
            stats_file.write(f"  Overall Pre-association - Mean: {pre_mean:.4f}, Std: {pre_std:.4f}\n")
            stats_file.write(f"  Overall Post-association - Mean: {post_mean:.4f}, Std: {post_std:.4f}\n")
            stats_file.write(f"  Overall Change: {post_mean - pre_mean:.4f}\n")
            
            stats_file.write(f"  Female person words:\n")
            stats_file.write(f"    Pre-association - Mean: {female_pre_mean:.4f}, Std: {female_pre_std:.4f}\n")
            stats_file.write(f"    Post-association - Mean: {female_post_mean:.4f}, Std: {female_post_std:.4f}\n")
            stats_file.write(f"    Change: {female_post_mean - female_pre_mean:.4f}\n")
            
            stats_file.write(f"  Male person words:\n")
            stats_file.write(f"    Pre-association - Mean: {male_pre_mean:.4f}, Std: {male_pre_std:.4f}\n")
            stats_file.write(f"    Post-association - Mean: {male_post_mean:.4f}, Std: {male_post_std:.4f}\n")
            stats_file.write(f"    Change: {male_post_mean - male_pre_mean:.4f}\n")

print(f"Statistics for all models have been written to {os.path.join(output_file_path, f'statistics_bert_replicated_post_adapted.txt')}")

################### FOR GERMAN MODELS #########################

typ = "regular"

# Create output directory for statistics
output_file_path = f"../results/statistics/german/Lou/{typ}/"
# Create directory if it doesn't exist
os.makedirs(output_file_path, exist_ok=True)

# Read the CSV file
data = pd.read_csv(f'../results/association_files/german/Lou/{typ}/results_Lou_all_models_replicated_both_{typ}_adapted.csv', sep="\t")

# Print available columns to help with debugging
print("Available columns in dataset:")
print([col for col in data.columns if 'Pre_Assoc' in col or 'Post_Assoc' in col])

# Dictionary of models to analyze
models = {
    "dbmdz": "dbmdz",
    "google-bert": "google-bert",
    "deepset-bert": "deepset-bert",
    "distilbert": "distilbert"
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

# Open a file to write all statistics
with open(os.path.join(output_file_path, f"statistics_Lou_post_all_models_{typ}_both_adapted.txt"), "w") as stats_file:
    # For each model
    for model_name, model_prefix in models.items():
        # Check if the model has both pre and post association scores
        pre_col = f'Pre_Assoc_{model_prefix}_Avg'
        post_col = f'Post_Assoc_{model_prefix}_Avg'
        
        # Check if these columns exist
        if pre_col not in data.columns or post_col not in data.columns:
            # Try to find matching columns
            pre_cols = [col for col in data.columns if col.startswith('Pre_Assoc') and model_prefix in col]
            post_cols = [col for col in data.columns if col.startswith('Post_Assoc') and model_prefix in col]
            
            if pre_cols and post_cols:
                pre_col = pre_cols[0]
                post_col = post_cols[0]
                stats_file.write(f"\nFound alternative column names for {model_name}:")
                stats_file.write(f"\n  Pre: {pre_col}")
                stats_file.write(f"\n  Post: {post_col}\n")
            else:
                stats_file.write(f"\n\n{'='*50}\n")
                stats_file.write(f"Model: {model_name}\n")
                stats_file.write(f"{'='*50}\n")
                stats_file.write(f"Missing pre or post association scores for this model.\n")
                stats_file.write(f"Available columns: {[col for col in data.columns if 'Pre_Assoc' in col or 'Post_Assoc' in col]}\n")
                continue
            
        # Write header for this model
        stats_file.write(f"\n\n{'='*50}\n")
        stats_file.write(f"Model: {model_name}\n")
        stats_file.write(f"{'='*50}\n")
        stats_file.write(f"Using columns: {pre_col} and {post_col}\n\n")
        
        # Get pre and post association scores
        pre_association = data[pre_col].to_list()
        post_association = data[post_col].to_list()

        # Get profession gender-specific scores
        female_professions_pre = data.loc[(
                data['Prof_Gender'] == 'female'), pre_col].tolist()
        male_professions_pre = data.loc[(
            data['Prof_Gender'] == 'male'), pre_col].tolist()
        balanced_professions_pre = data.loc[(
            data['Prof_Gender'] == 'balanced'), pre_col].tolist()

        female_professions_post = data.loc[(
            data['Prof_Gender'] == 'female'), post_col].tolist()
        male_professions_post = data.loc[(
            data['Prof_Gender'] == 'male'), post_col].tolist()
        balanced_professions_post = data.loc[(
            data['Prof_Gender'] == 'balanced'), post_col].tolist()

        # PRE-ASSOCIATION STATISTICS
        stats_file.write("PRE-ASSOCIATION STATISTICS\n")
        stats_file.write("-------------------------\n")
        
        # Female gender within statistically female professions
        female_gender_in_female_professions_mean = data.loc[(
            data['Prof_Gender'] == 'female') & (data['Gender'] == 'female'), pre_col].mean()
        stats_file.write(f'Mean for female gender within statistically female professions: {female_gender_in_female_professions_mean:.4f}\n')

        # Male gender within statistically female professions
        male_gender_in_female_professions_mean = data.loc[(
            data['Prof_Gender'] == 'female') & (data['Gender'] == 'male'), pre_col].mean()
        stats_file.write(f'Mean for male gender within statistically female professions: {male_gender_in_female_professions_mean:.4f}\n')

        # Female gender within statistically male professions
        female_gender_in_male_professions_mean = data.loc[
            (data['Prof_Gender'] == 'male') & (data['Gender'] == 'female'),
            pre_col
        ].mean()
        stats_file.write(f'Mean for female gender within statistically male professions: {female_gender_in_male_professions_mean:.4f}\n')

        # Male gender within statistically male professions
        male_gender_in_male_professions_mean = data.loc[
            (data['Prof_Gender'] == 'male') & (data['Gender'] == 'male'),
            pre_col
        ].mean()
        stats_file.write(f'Mean for male gender within statistically male professions: {male_gender_in_male_professions_mean:.4f}\n')

        # Female gender within statistically balanced professions
        female_gender_in_balanced_professions_mean = data.loc[
            (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'female'),
            pre_col
        ].mean()
        stats_file.write(f'Mean for female gender within statistically balanced professions: {female_gender_in_balanced_professions_mean:.4f}\n')

        # Male gender within statistically balanced professions
        male_gender_in_balanced_professions_mean = data.loc[
            (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'male'),
            pre_col
        ].mean()
        stats_file.write(f'Mean for male gender within statistically balanced professions: {male_gender_in_balanced_professions_mean:.4f}\n')

        # POST-ASSOCIATION STATISTICS
        stats_file.write("\nPOST-ASSOCIATION STATISTICS\n")
        stats_file.write("--------------------------\n")
        
        # Female gender within statistically female professions
        female_gender_in_female_professions_post_mean = data.loc[(
            data['Prof_Gender'] == 'female') & (data['Gender'] == 'female'), post_col].mean()
        stats_file.write(f'Mean for female gender within statistically female professions: {female_gender_in_female_professions_post_mean:.4f}\n')

        # Male gender within statistically female professions
        male_gender_in_female_professions_post_mean = data.loc[(
            data['Prof_Gender'] == 'female') & (data['Gender'] == 'male'), post_col].mean()
        stats_file.write(f'Mean for male gender within statistically female professions: {male_gender_in_female_professions_post_mean:.4f}\n')

        # Female gender within statistically male professions
        female_gender_in_male_professions_post_mean = data.loc[
            (data['Prof_Gender'] == 'male') & (data['Gender'] == 'female'),
            post_col
        ].mean()
        stats_file.write(f'Mean for female gender within statistically male professions: {female_gender_in_male_professions_post_mean:.4f}\n')

        # Male gender within statistically male professions
        male_gender_in_male_professions_post_mean = data.loc[
            (data['Prof_Gender'] == 'male') & (data['Gender'] == 'male'),
            post_col
        ].mean()
        stats_file.write(f'Mean for male gender within statistically male professions: {male_gender_in_male_professions_post_mean:.4f}\n')

        # Female gender within statistically balanced professions
        female_gender_in_balanced_professions_post_mean = data.loc[
            (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'female'),
            post_col
        ].mean()
        stats_file.write(f'Mean for female gender within statistically balanced professions: {female_gender_in_balanced_professions_post_mean:.4f}\n')

        # Male gender within statistically balanced professions
        male_gender_in_balanced_professions_post_mean = data.loc[
            (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'male'),
            post_col
        ].mean()
        stats_file.write(f'Mean for male gender within statistically balanced professions: {male_gender_in_balanced_professions_post_mean:.4f}\n')

        # STATISTICAL TESTS
        stats_file.write("\nSTATISTICAL TESTS\n")
        stats_file.write("----------------\n")
        
        # Overall Wilcoxon test
        stat, p_value = wilcoxon(pre_association, post_association, alternative='two-sided')
        stats_file.write(f"Overall Wilcoxon test statistic: {stat}\n")
        stats_file.write(f"P-value: {p_value}\n")
        
        alpha = 0.05  # significance level
        if p_value < alpha:
            stats_file.write("Reject the null hypothesis: There is a significant difference.\n")
        else:
            stats_file.write("Fail to reject the null hypothesis: No significant difference.\n")

        # Female professions Wilcoxon test
        if len(female_professions_pre) > 0 and len(female_professions_post) > 0:
            stat_F, p_value_F = wilcoxon(female_professions_pre, female_professions_post, alternative='two-sided')
            stats_file.write(f"\nWilcoxon test statistic for female professions (F): {stat_F}\n")
            stats_file.write(f"P-value: {p_value_F}\n")
            
            if p_value_F < alpha:
                stats_file.write("Reject the null hypothesis: There is a significant difference.\n")
            else:
                stats_file.write("Fail to reject the null hypothesis: No significant difference.\n")
        else:
            stats_file.write("\nNot enough data for female professions to perform Wilcoxon test.\n")
            stat_F, p_value_F = 0, 1  # Default values for later use

        # Male professions Wilcoxon test
        if len(male_professions_pre) > 0 and len(male_professions_post) > 0:
            stat_M, p_value_M = wilcoxon(male_professions_pre, male_professions_post, alternative='two-sided')
            stats_file.write(f"\nWilcoxon test statistic for male professions (M): {stat_M}\n")
            stats_file.write(f"P-value: {p_value_M}\n")
            
            if p_value_M < alpha:
                stats_file.write("Reject the null hypothesis: There is a significant difference.\n")
            else:
                stats_file.write("Fail to reject the null hypothesis: No significant difference.\n")
        else:
            stats_file.write("\nNot enough data for male professions to perform Wilcoxon test.\n")
            stat_M, p_value_M = 0, 1  # Default values for later use

        # Balanced professions Wilcoxon test
        if len(balanced_professions_pre) > 0 and len(balanced_professions_post) > 0:
            stat_B, p_value_B = wilcoxon(balanced_professions_pre, balanced_professions_post, alternative='two-sided')
            stats_file.write(f"\nWilcoxon test statistic for balanced professions (B): {stat_B}\n")
            stats_file.write(f"P-value: {p_value_B}\n")
            
            if p_value_B < alpha:
                stats_file.write("Reject the null hypothesis: There is a significant difference.\n")
            else:
                stats_file.write("Fail to reject the null hypothesis: No significant difference.\n")
        else:
            stats_file.write("\nNot enough data for balanced professions to perform Wilcoxon test.\n")
            stat_B, p_value_B = 0, 1  # Default values for later use

        # EFFECT SIZES
        stats_file.write("\nEFFECT SIZES\n")
        stats_file.write("-----------\n")
    
        # Number of samples (N)
        N_ALL = len(pre_association)
        N_F = len(female_professions_pre) 
        N_M = len(male_professions_pre)
        N_B = len(balanced_professions_pre)

        # Add ALL to the results dictionary
        groups = {
            "all": {"wilcoxon_statistic": stat, "n": N_ALL},
            "female": {"wilcoxon_statistic": stat_F, "n": N_F},
            "male": {"wilcoxon_statistic": stat_M, "n": N_M},
            "balanced": {"wilcoxon_statistic": stat_B, "n": N_B}
        }
        
        for group, values in groups.items():
            wilcoxon_statistic = values["wilcoxon_statistic"]
            group_n = values["n"]
            if group_n > 0:  # Avoid division by zero
                effect_size = compute_cohens_w(wilcoxon_statistic, group_n)
                stats_file.write(f"{group.capitalize()} group - Cohen's W effect size: {effect_size}\n")
            else:
                stats_file.write(f"{group.capitalize()} group - Not enough data to compute effect size\n")

    # Write a summary at the end
    stats_file.write(f"\n\n{'='*50}\n")
    stats_file.write("SUMMARY OF ALL MODELS\n")
    stats_file.write(f"{'='*50}\n")

    for model_name, model_prefix in models.items():
        pre_col = f'Pre_Assoc_{model_prefix}_Avg'
        post_col = f'Post_Assoc_{model_prefix}_Avg'
            
        # Check if these columns exist and find alternatives if needed
        if pre_col not in data.columns or post_col not in data.columns:
            pre_cols = [col for col in data.columns if col.startswith('Pre_Assoc') and model_prefix in col]
            post_cols = [col for col in data.columns if col.startswith('Post_Assoc') and model_prefix in col]
            
            if pre_cols and post_cols:
                pre_col = pre_cols[0]
                post_col = post_cols[0]
            else:
                stats_file.write(f"\nModel: {model_name} - No valid columns found\n")
                continue
            
        if pre_col in data.columns and post_col in data.columns:
            # Overall means
            pre_mean = data[pre_col].mean()
            pre_std = data[pre_col].std()
            post_mean = data[post_col].mean()
            post_std = data[post_col].std()
            
            # Female person words
            female_pre_mean = data.loc[data['Gender'] == 'female', pre_col].mean()
            female_pre_std = data.loc[data['Gender'] == 'female', pre_col].std()
            female_post_mean = data.loc[data['Gender'] == 'female', post_col].mean()
            female_post_std = data.loc[data['Gender'] == 'female', post_col].std()
            
            # Male person words
            male_pre_mean = data.loc[data['Gender'] == 'male', pre_col].mean()
            male_pre_std = data.loc[data['Gender'] == 'male', pre_col].std() 
            male_post_mean = data.loc[data['Gender'] == 'male', post_col].mean()
            male_post_std = data.loc[data['Gender'] == 'male', post_col].std()
            
            stats_file.write(f"\nModel: {model_name}\n")
            stats_file.write(f"  Overall Pre-association - Mean: {pre_mean:.4f}, Std: {pre_std:.4f}\n")
            stats_file.write(f"  Overall Post-association - Mean: {post_mean:.4f}, Std: {post_std:.4f}\n")
            stats_file.write(f"  Overall Change: {post_mean - pre_mean:.4f}\n")
            
            stats_file.write(f"  Female person words:\n")
            stats_file.write(f"    Pre-association - Mean: {female_pre_mean:.4f}, Std: {female_pre_std:.4f}\n")
            stats_file.write(f"    Post-association - Mean: {female_post_mean:.4f}, Std: {female_post_std:.4f}\n")
            stats_file.write(f"    Change: {female_post_mean - female_pre_mean:.4f}\n")
            
            stats_file.write(f"  Male person words:\n")
            stats_file.write(f"    Pre-association - Mean: {male_pre_mean:.4f}, Std: {male_pre_std:.4f}\n")
            stats_file.write(f"    Post-association - Mean: {male_post_mean:.4f}, Std: {male_post_std:.4f}\n")
            stats_file.write(f"    Change: {male_post_mean - male_pre_mean:.4f}\n")

print(f"Statistics for all models have been written to {os.path.join(output_file_path, f'statistics_Lou_post_all_models_{typ}_both_adapted.txt')}")