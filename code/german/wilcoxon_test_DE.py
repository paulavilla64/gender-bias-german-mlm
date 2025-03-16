import pandas as pd
from scipy.stats import wilcoxon
import math
import numpy as np

# Due to the ineffectiveness of the method for German, we only report on pre- and not on post-associations
# In order to statistically test the difference between associations for male and female person
# words, the Wilcoxon signed-rank test was again computed for each profession group separately.


# Read a CSV file
data = pd.read_csv('../data/output_csv_files/results_DE_with_perplexity_validation_gender_neutral.csv', sep='\t')

# female gender within statistically female professions
female_gender_in_female_professions = data.loc[(
    data['Prof_Gender'] == 'female') & (data['Gender'] == 'female'), 'Pre_Assoc'].tolist()

# male gender within statistically female professions
male_gender_in_female_professions = data.loc[(
    data['Prof_Gender'] == 'female') & (data['Gender'] == 'male'), 'Pre_Assoc'].tolist()

# female gender within statistically male professions
female_gender_in_male_professions = data.loc[
    (data['Prof_Gender'] == 'male') & (data['Gender'] == 'female'), 'Pre_Assoc'].tolist()

# male gender within statistically male professions
male_gender_in_male_professions = data.loc[
    (data['Prof_Gender'] == 'male') & (data['Gender'] == 'male'), 'Pre_Assoc'].tolist()

# female gender within statistically balanced professions
female_gender_in_balanced_professions = data.loc[
    (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'female'), 'Pre_Assoc'].tolist()

# male gender within statistically balanced professions
male_gender_in_balanced_professions = data.loc[
    (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'male'), 'Pre_Assoc'].tolist()

# Compute mean of the female gender within statistically female professions
female_gender_in_female_professions_mean = data.loc[(
    data['Prof_Gender'] == 'female') & (data['Gender'] == 'female'), 'Pre_Assoc'].mean()
print(f'Mean for female gender within statistically female professions: {female_gender_in_female_professions_mean}')

# Compute mean of the male gender within statistically female professions
male_gender_in_female_professions_mean = data.loc[(
    data['Prof_Gender'] == 'female') & (data['Gender'] == 'male'), 'Pre_Assoc'].mean()
print(f'Mean for male gender within statistically female professions: {male_gender_in_female_professions_mean}')

# Compute mean of female gender within statistically male professions
female_gender_in_male_professions_mean = data.loc[
    (data['Prof_Gender'] == 'male') & (data['Gender'] == 'female'),
    'Pre_Assoc'
].mean()
print(f'Mean for female gender within statistically male professions: {female_gender_in_male_professions_mean}')

# Compute mean of male gender within statistically male professions
male_gender_in_male_professions_mean = data.loc[
    (data['Prof_Gender'] == 'male') & (data['Gender'] == 'male'),
    'Pre_Assoc'
].mean()
print(f'Mean for male gender within statistically male professions: {male_gender_in_male_professions_mean}')

# Compute mean of female gender within statistically balanced professions
female_gender_in_balanced_professions_mean = data.loc[
    (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'female'),
    'Pre_Assoc'
].mean()
print(f'Mean for female gender within statistically balanced professions: {female_gender_in_balanced_professions_mean}')

# Compute mean of male gender within statistically balanced professions
male_gender_in_balanced_professions_mean = data.loc[
    (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'male'),
    'Pre_Assoc'
].mean()
print(f'Mean for male gender within statistically balanced professions: {male_gender_in_balanced_professions_mean}')


# Calculate the differences for F
differences_F = [after - before for before, after in zip(male_gender_in_female_professions, female_gender_in_female_professions)]

# Perform the Wilcoxon signed-rank test for F
stat_F, p_value_F = wilcoxon(male_gender_in_female_professions, female_gender_in_female_professions, alternative='two-sided')

# Output the results
print(f"Wilcoxon test statistic for F: {stat_F}")
print(f"P-value: {p_value_F}")

# Interpret the results
alpha = 0.05  # significance level
if p_value_F < alpha:
    print("Reject the null hypothesis: There is a significant difference.")
else:
    print("Fail to reject the null hypothesis: No significant difference.")


# Calculate the differences for M
differences_M = [after - before for before, after in zip(male_gender_in_male_professions, female_gender_in_male_professions)]

# Perform the Wilcoxon signed-rank test for F
stat_M, p_value_M = wilcoxon(male_gender_in_male_professions, female_gender_in_male_professions, alternative='two-sided')

# Output the results
print(f"Wilcoxon test statistic for M: {stat_M}")
print(f"P-value: {p_value_M}")

# Interpret the results
alpha = 0.05  # significance level
if p_value_M < alpha:
    print("Reject the null hypothesis: There is a significant difference.")
else:
    print("Fail to reject the null hypothesis: No significant difference.")


# Calculate the differences for F
differences_B = [after - before for before, after in zip(male_gender_in_balanced_professions, female_gender_in_balanced_professions)]

# Perform the Wilcoxon signed-rank test for F
stat_B, p_value_B = wilcoxon(male_gender_in_balanced_professions, female_gender_in_balanced_professions, alternative='two-sided')

# Output the results
print(f"Wilcoxon test statistic for B: {stat_B}")
print(f"P-value: {p_value_B}")

# Interpret the results
alpha = 0.05  # significance level
if p_value_B < alpha:
    print("Reject the null hypothesis: There is a significant difference.")
else:
    print("Fail to reject the null hypothesis: No significant difference.")


# Compute effect size

def compute_cohens_w(wilcoxon_statistic, N):
    """
    Computes the effect size for the Wilcoxon Signed-Rank Test using Cohen's W.

    Parameters:
        wilcoxon_statistic (float): The Wilcoxon signed-rank test statistic (Z-value).
        N (int): Sample size.

    Returns:
        float: The computed Cohen's W effect size.
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


# Number of samples (N)
N = 1800 


# Example usage with the provided results
results = {
    "female": {"wilcoxon_statistic": stat_F, "n": N},
    "male": {"wilcoxon_statistic": stat_M, "n": N},
    "balanced": {"wilcoxon_statistic": stat_B, "n": N}
}

# Compute Cohen's W for each group and display results
for group, values in results.items():
    wilcoxon_statistic = values["wilcoxon_statistic"]
    N = values["n"]

    effect_size = compute_cohens_w(wilcoxon_statistic, N)
    print(f"{group.capitalize()} group - Cohen's W effect size: {effect_size}")

# Function to compute standard deviation
# SD = root(1/N*sum(xi-u)^2)
def compute_standard_deviation(data):
    return np.std(data)

# Compute SD for each group
sd_balanced_female = compute_standard_deviation(female_gender_in_balanced_professions)
sd_balanced_male = compute_standard_deviation(male_gender_in_balanced_professions)
sd_female_female = compute_standard_deviation(female_gender_in_female_professions)
sd_female_male = compute_standard_deviation(male_gender_in_female_professions)
sd_male_female = compute_standard_deviation(female_gender_in_male_professions)
sd_male_male = compute_standard_deviation(male_gender_in_male_professions)

# Print results
print(f"Standard Deviation for Balanced Professions - Female Person Words: {sd_balanced_female:.2f}") 
print(f"Standard Deviation for Balanced Professions - Male Person Words: {sd_balanced_male:.2f}")
print(f"Standard Deviation for Female Professions - Female Person Words: {sd_female_female:.2f}")
print(f"Standard Deviation for Female Professions - Male Person Words: {sd_female_male:.2f}")
print(f"Standard Deviation for Male Professions - Female Person Words: {sd_male_female:.2f}")
print(f"Standard Deviation for Male Professions - Male Person Words: {sd_male_male:.2f}")




