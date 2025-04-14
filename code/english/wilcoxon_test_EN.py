import pandas as pd
from scipy.stats import wilcoxon
import math
import os

typ = "neutral"

# Create output directory for statistics
output_file_path = "../../data/statistics/german/zero_difference/Lou/"
# Create directory if it doesn't exist
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Read a CSV file
data = pd.read_csv('../../data/output_csv_files/german/Lou/associations_all_models_DE_zero_difference_neutral_avg.csv', sep="\t")

# List of all models to process
models = ["dbmdz", "google_bert", "deepset_bert", "distilbert"]

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
with open(os.path.join(output_file_path, f"statistics_DE_post_all_models_zero_difference_{typ}_avg.txt"), "w") as stats_file:
    # For each model
    for name in models:
        # Check if the model has both pre and post association scores
        pre_col = f'Pre_Assoc_{name}_Avg'
        post_col = f'Post_Assoc_{name}_Avg'
        
        # For some models, the column might not have "_Avg" suffix
        if pre_col not in data.columns:
            pre_col = f'Pre_Assoc_{name}'
        if post_col not in data.columns:
            post_col = f'Post_Assoc_{name}'
            
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
        
        # For some models, the column might not have "_Avg" suffix
        if pre_col not in data.columns:
            pre_col = f'Pre_Assoc_{name}'
        if post_col not in data.columns:
            post_col = f'Post_Assoc_{name}'
            
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

print(f"Statistics for all models have been written to {os.path.join(output_file_path, f'statistics_DE_post_all_models_zero_difference_{typ}_avg.txt')}")
# Additionally, save individual files for each model if needed
# for name in models:
#     if f'Pre_Assoc' in data.columns and f'Post_Assoc' in data.columns:
#         with open(os.path.join(output_dir, f"statistics.txt"), "w") as model_file:
#             # Write model-specific statistics (similar to above)
#             model_file.write(f"Statistics for model: n")
#             model_file.write(f"{'='*50}\n")
            
#             # Add the relevant statistics for this model
#             # (Code omitted for brevity - you can copy the relevant sections from above)
            
#         print(f"Statistics for have been written to {os.path.join(output_dir, f'statistics.txt')}")



## FOR ENGLISH FILE ## 

# # Read a CSV file
# data = pd.read_csv('../../data/output_csv_files/english/results_EN_bert_42.csv', sep='\t')

# pre_association = data['Pre_Assoc'].to_list()
# post_association = data['Post_Assoc'].to_list()

# female_professions_pre = data.loc[(
#         data['Prof_Gender'] == 'female'), 'Pre_Assoc'].tolist()
# male_professions_pre = data.loc[(
#     data['Prof_Gender'] == 'male'), 'Pre_Assoc'].tolist()
# balanced_professions_pre = data.loc[(
#     data['Prof_Gender'] == 'balanced'), 'Pre_Assoc'].tolist()

# female_professions_post = data.loc[(
#     data['Prof_Gender'] == 'female'), 'Post_Assoc'].tolist()
# male_professions_post = data.loc[(
#     data['Prof_Gender'] == 'male'), 'Post_Assoc'].tolist()
# balanced_professions_post = data.loc[(
#     data['Prof_Gender'] == 'balanced'), 'Post_Assoc'].tolist()

# # PRE
# # female gender within statistically female professions
# female_gender_in_female_professions = data.loc[(
#     data['Prof_Gender'] == 'female') & (data['Gender'] == 'female'), 'Pre_Assoc'].tolist()

# # male gender within statistically female professions
# male_gender_in_female_professions = data.loc[(
#     data['Prof_Gender'] == 'female') & (data['Gender'] == 'male'), 'Pre_Assoc'].tolist()

# # female gender within statistically male professions
# female_gender_in_male_professions = data.loc[
#     (data['Prof_Gender'] == 'male') & (data['Gender'] == 'female'), 'Pre_Assoc'].tolist()

# # male gender within statistically male professions
# male_gender_in_male_professions = data.loc[
#     (data['Prof_Gender'] == 'male') & (data['Gender'] == 'male'), 'Pre_Assoc'].tolist()

# # female gender within statistically balanced professions
# female_gender_in_balanced_professions = data.loc[
#     (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'female'), 'Pre_Assoc'].tolist()

# # male gender within statistically balanced professions
# male_gender_in_balanced_professions = data.loc[
#     (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'male'), 'Pre_Assoc'].tolist()



# # Compute mean of the female gender within statistically female professions
# female_gender_in_female_professions_mean = data.loc[(
#     data['Prof_Gender'] == 'female') & (data['Gender'] == 'female'), 'Pre_Assoc'].mean()
# print(f'Mean for female gender within statistically female professions: {female_gender_in_female_professions_mean}')

# # Compute mean of the male gender within statistically female professions
# male_gender_in_female_professions_mean = data.loc[(
#     data['Prof_Gender'] == 'female') & (data['Gender'] == 'male'), 'Pre_Assoc'].mean()
# print(f'Mean for male gender within statistically female professions: {male_gender_in_female_professions_mean}')

# # Compute mean of female gender within statistically male professions
# female_gender_in_male_professions_mean = data.loc[
#     (data['Prof_Gender'] == 'male') & (data['Gender'] == 'female'),
#     'Pre_Assoc'
# ].mean()
# print(f'Mean for female gender within statistically male professions: {female_gender_in_male_professions_mean}')

# # Compute mean of male gender within statistically male professions
# male_gender_in_male_professions_mean = data.loc[
#     (data['Prof_Gender'] == 'male') & (data['Gender'] == 'male'),
#     'Pre_Assoc'
# ].mean()
# print(f'Mean for male gender within statistically male professions: {male_gender_in_male_professions_mean}')

# # Compute mean of female gender within statistically balanced professions
# female_gender_in_balanced_professions_mean = data.loc[
#     (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'female'),
#     'Pre_Assoc'
# ].mean()
# print(f'Mean for female gender within statistically balanced professions: {female_gender_in_balanced_professions_mean}')

# # Compute mean of male gender within statistically balanced professions
# male_gender_in_balanced_professions_mean = data.loc[
#     (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'male'),
#     'Pre_Assoc'
# ].mean()
# print(f'Mean for male gender within statistically balanced professions: {male_gender_in_balanced_professions_mean}')



# # POST
# # female gender within statistically female professions
# female_gender_in_female_professions = data.loc[(
#     data['Prof_Gender'] == 'female') & (data['Gender'] == 'female'), 'Post_Assoc'].tolist()

# # male gender within statistically female professions
# male_gender_in_female_professions = data.loc[(
#     data['Prof_Gender'] == 'female') & (data['Gender'] == 'male'), 'Post_Assoc'].tolist()

# # female gender within statistically male professions
# female_gender_in_male_professions = data.loc[
#     (data['Prof_Gender'] == 'male') & (data['Gender'] == 'female'), 'Post_Assoc'].tolist()

# # male gender within statistically male professions
# male_gender_in_male_professions = data.loc[
#     (data['Prof_Gender'] == 'male') & (data['Gender'] == 'male'), 'Post_Assoc'].tolist()

# # female gender within statistically balanced professions
# female_gender_in_balanced_professions = data.loc[
#     (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'female'), 'Post_Assoc'].tolist()

# # male gender within statistically balanced professions
# male_gender_in_balanced_professions = data.loc[
#     (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'male'), 'Post_Assoc'].tolist()



# # Compute mean of the female gender within statistically female professions
# female_gender_in_female_professions_mean = data.loc[(
#     data['Prof_Gender'] == 'female') & (data['Gender'] == 'female'), 'Post_Assoc'].mean()
# print(f'POST: Mean for female gender within statistically female professions: {female_gender_in_female_professions_mean}')

# # Compute mean of the male gender within statistically female professions
# male_gender_in_female_professions_mean = data.loc[(
#     data['Prof_Gender'] == 'female') & (data['Gender'] == 'male'), 'Post_Assoc'].mean()
# print(f'POST: Mean for male gender within statistically female professions: {male_gender_in_female_professions_mean}')

# # Compute mean of female gender within statistically male professions
# female_gender_in_male_professions_mean = data.loc[
#     (data['Prof_Gender'] == 'male') & (data['Gender'] == 'female'),
#     'Post_Assoc'
# ].mean()
# print(f'POST: Mean for female gender within statistically male professions: {female_gender_in_male_professions_mean}')

# # Compute mean of male gender within statistically male professions
# male_gender_in_male_professions_mean = data.loc[
#     (data['Prof_Gender'] == 'male') & (data['Gender'] == 'male'),
#     'Post_Assoc'
# ].mean()
# print(f'POST: Mean for male gender within statistically male professions: {male_gender_in_male_professions_mean}')

# # Compute mean of female gender within statistically balanced professions
# female_gender_in_balanced_professions_mean = data.loc[
#     (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'female'),
#     'Post_Assoc'
# ].mean()
# print(f'POST: Mean for female gender within statistically balanced professions: {female_gender_in_balanced_professions_mean}')

# # Compute mean of male gender within statistically balanced professions
# male_gender_in_balanced_professions_mean = data.loc[
#     (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'male'),
#     'Post_Assoc'
# ].mean()
# print(f'POST: Mean for male gender within statistically balanced professions: {male_gender_in_balanced_professions_mean}')


# # Compute effect size function
# def compute_cohens_w(wilcoxon_statistic, N):
#     """
#     Computes the effect size for the Wilcoxon Signed-Rank Test using Cohen's W.

#     Parameters:
#         wilcoxon_statistic (float): The Wilcoxon signed-rank test statistic (Z-value).
#         N (int): Sample size.

#     Returns:
#         float: The computed Cohen's W effect size.
#     """
#     # Step 1: Compute expected W (E(W))
#     E_W = (N * (N + 1)) / 4

#     # Step 2: Compute the standard error of W (SE(W))
#     SE_W = math.sqrt((N * (N + 1) * (2 * N + 1)) / 24)

#     # Step 3: Compute the Z-score
#     Z_value = (wilcoxon_statistic - E_W) / SE_W

#     # Step 4: Compute Cohen's W effect size
#     cohens_w = Z_value / math.sqrt(N)

#     return cohens_w


# # Calculate the differences for ALL professions (not just F, M, B separately)
# differences = [after - before for before, after in zip(pre_association, post_association)]

# # Perform the Wilcoxon signed-rank test
# stat, p_value = wilcoxon(pre_association, post_association, alternative='two-sided')

# # Output the results
# print(f"Wilcoxon test statistic (ALL): {stat}")
# print(f"P-value (ALL): {p_value}")

# # Interpret the results
# alpha = 0.05  # significance level
# if p_value < alpha:
#     print("Reject the null hypothesis (ALL): There is a significant difference.")
# else:
#     print("Fail to reject the null hypothesis (ALL): No significant difference.")

# # Calculate the differences for F
# differences_F = [after - before for before, after in zip(female_professions_pre, female_professions_post)]

# # Perform the Wilcoxon signed-rank test for F
# stat_F, p_value_F = wilcoxon(female_professions_pre, female_professions_post, alternative='two-sided')

# # Output the results
# print(f"Wilcoxon test statistic for F: {stat_F}")
# print(f"P-value: {p_value_F}")

# # Interpret the results
# alpha = 0.05  # significance level
# if p_value_F < alpha:
#     print("Reject the null hypothesis: There is a significant difference.")
# else:
#     print("Fail to reject the null hypothesis: No significant difference.")


# # Calculate the differences for M
# differences_M = [after - before for before, after in zip(male_professions_pre, male_professions_post)]

# # Perform the Wilcoxon signed-rank test for M
# stat_M, p_value_M = wilcoxon(male_professions_pre, male_professions_post, alternative='two-sided')

# # Output the results
# print(f"Wilcoxon test statistic for M: {stat_M}")
# print(f"P-value: {p_value_M}")

# # Interpret the results
# alpha = 0.05  # significance level
# if p_value_M < alpha:
#     print("Reject the null hypothesis: There is a significant difference.")
# else:
#     print("Fail to reject the null hypothesis: No significant difference.")


# # Calculate the differences for B
# differences_B = [after - before for before, after in zip(balanced_professions_pre, balanced_professions_post)]

# # Perform the Wilcoxon signed-rank test for B
# stat_B, p_value_B = wilcoxon(balanced_professions_pre, balanced_professions_post, alternative='two-sided')

# # Output the results
# print(f"Wilcoxon test statistic for B: {stat_B}")
# print(f"P-value: {p_value_B}")

# # Interpret the results
# alpha = 0.05  # significance level
# if p_value_B < alpha:
#     print("Reject the null hypothesis: There is a significant difference.")
# else:
#     print("Fail to reject the null hypothesis: No significant difference.")


# # Number of samples (N)
# N_ALL = len(pre_association)
# N_F = len(female_professions_pre) 
# N_M = len(male_professions_pre)
# N_B = len(balanced_professions_pre)

# # Add ALL to the results dictionary
# results = {
#     "all": {"wilcoxon_statistic": stat, "n": N_ALL},
#     "female": {"wilcoxon_statistic": stat_F, "n": N_F},
#     "male": {"wilcoxon_statistic": stat_M, "n": N_M},
#     "balanced": {"wilcoxon_statistic": stat_B, "n": N_B}
# }

# # Compute Cohen's W for each group and display results
# for group, values in results.items():
#     wilcoxon_statistic = values["wilcoxon_statistic"]
#     N = values["n"]

#     effect_size = compute_cohens_w(wilcoxon_statistic, N)
#     print(f"{group.capitalize()} group - Cohen's W effect size: {effect_size}")


# if f'Pre_Assoc' in data.columns and f'Post_Assoc' in data.columns:
#     # Overall means
#     pre_mean = data[f'Pre_Assoc'].mean()
#     pre_std = data[f'Pre_Assoc'].std()
#     post_mean = data[f'Post_Assoc'].mean()
#     post_std = data[f'Post_Assoc'].std()

#     # Female person words
#     female_pre_mean = data.loc[data['Gender'] == 'female', f'Pre_Assoc'].mean()
#     female_pre_std = data.loc[data['Gender'] == 'female', f'Pre_Assoc'].std()
#     female_post_mean = data.loc[data['Gender'] == 'female', f'Post_Assoc'].mean()
#     female_post_std = data.loc[data['Gender'] == 'female', f'Post_Assoc'].std()

#     # Male person words
#     male_pre_mean = data.loc[data['Gender'] == 'male', f'Pre_Assoc'].mean()
#     male_pre_std = data.loc[data['Gender'] == 'male', f'Pre_Assoc'].std() 
#     male_post_mean = data.loc[data['Gender'] == 'male', f'Post_Assoc'].mean()
#     male_post_std = data.loc[data['Gender'] == 'male', f'Post_Assoc'].std()

    
#     print(f"  Overall Pre-association - Mean: {pre_mean:.4f}, Std: {pre_std:.4f}")
#     print(f"  Overall Post-association - Mean: {post_mean:.4f}, Std: {post_std:.4f}")
#     print(f"  Overall Change: {post_mean - pre_mean:.4f}")

#     print(f"  Female person words:")
#     print(f"    Pre-association - Mean: {female_pre_mean:.4f}, Std: {female_pre_std:.4f}")
#     print(f"    Post-association - Mean: {female_post_mean:.4f}, Std: {female_post_std:.4f}")
#     print(f"    Change: {female_post_mean - female_pre_mean:.4f}")

#     print(f"  Male person words:")
#     print(f"    Pre-association - Mean: {male_pre_mean:.4f}, Std: {male_pre_std:.4f}")
#     print(f"    Post-association - Mean: {male_post_mean:.4f}, Std: {male_post_std:.4f}")
#     print(f"    Change: {male_post_mean - male_pre_mean:.4f}\n")

# Function to compute standard deviation
# SD = root(1/N*sum(xi-u)^2)
# def compute_standard_deviation(data):
#     return np.std(data)

# # Compute SD for each group
# sd_balanced_female = compute_standard_deviation(female_gender_in_balanced_professions)
# sd_balanced_male = compute_standard_deviation(male_gender_in_balanced_professions)
# sd_female_female = compute_standard_deviation(female_gender_in_female_professions)
# sd_female_male = compute_standard_deviation(male_gender_in_female_professions)
# sd_male_female = compute_standard_deviation(female_gender_in_male_professions)
# sd_male_male = compute_standard_deviation(male_gender_in_male_professions)

# # Print results
# print(f"Standard Deviation for Balanced Professions - Female Person Words: {sd_balanced_female:.2f}") 
# print(f"Standard Deviation for Balanced Professions - Male Person Words: {sd_balanced_male:.2f}")
# print(f"Standard Deviation for Female Professions - Female Person Words: {sd_female_female:.2f}")
# print(f"Standard Deviation for Female Professions - Male Person Words: {sd_female_male:.2f}")
# print(f"Standard Deviation for Male Professions - Female Person Words: {sd_male_female:.2f}")
# print(f"Standard Deviation for Male Professions - Male Person Words: {sd_male_male:.2f}")