import pandas as pd
from scipy.stats import wilcoxon

# Read a CSV file
data = pd.read_csv('../data/results_EN.csv', sep='\t')

pre_association = data['Pre_Assoc'].to_list()
post_association = data['Post_Assoc'].to_list()

female_professions_pre = data.loc[(
        data['Prof_Gender'] == 'female'), 'Pre_Assoc'].tolist()
male_professions_pre = data.loc[(
    data['Prof_Gender'] == 'male'), 'Pre_Assoc'].tolist()
balanced_professions_pre = data.loc[(
    data['Prof_Gender'] == 'balanced'), 'Pre_Assoc'].tolist()

female_professions_post = data.loc[(
    data['Prof_Gender'] == 'female'), 'Post_Assoc'].tolist()
male_professions_post = data.loc[(
    data['Prof_Gender'] == 'male'), 'Post_Assoc'].tolist()
balanced_professions_post = data.loc[(
    data['Prof_Gender'] == 'balanced'), 'Post_Assoc'].tolist()

# Calculate the differences
differences = [after - before for before, after in zip(pre_association, post_association)]
print(f"Differences: {differences}")

# Perform the Wilcoxon signed-rank test
stat, p_value = wilcoxon(pre_association, post_association, alternative='two-sided')

# Output the results
print(f"Wilcoxon test statistic: {stat}")
print(f"P-value: {p_value}")

# Interpret the results
alpha = 0.05  # significance level
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference.")
else:
    print("Fail to reject the null hypothesis: No significant difference.")


# Calculate the differences for F
differences_F = [after - before for before, after in zip(female_professions_pre, female_professions_post)]
print(f"Differences_F: {differences_F}")

# Perform the Wilcoxon signed-rank test for F
stat_F, p_value_F = wilcoxon(female_professions_pre, female_professions_post, alternative='two-sided')

# Output the results
print(f"Wilcoxon test statistic for F: {stat_F}")
print(f"P-value: {p_value_F}")

# Interpret the results
alpha = 0.05  # significance level
if p_value_F < alpha:
    print("Reject the null hypothesis: There is a significant difference.")
else:
    print("Fail to reject the null hypothesis: No significant difference.")


# Calculate the differences for F
differences_F = [after - before for before, after in zip(female_professions_pre, female_professions_post)]
print(f"Differences_F: {differences_F}")

# Perform the Wilcoxon signed-rank test for F
stat_F, p_value_F = wilcoxon(female_professions_pre, female_professions_post, alternative='two-sided')

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
differences_M = [after - before for before, after in zip(male_professions_pre, male_professions_post)]
print(f"Differences_M: {differences_M}")

# Perform the Wilcoxon signed-rank test for F
stat_M, p_value_M = wilcoxon(male_professions_pre, male_professions_post, alternative='two-sided')

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
differences_B = [after - before for before, after in zip(balanced_professions_pre, balanced_professions_post)]
print(f"Differences_B: {differences_B}")

# Perform the Wilcoxon signed-rank test for F
stat_B, p_value_B = wilcoxon(balanced_professions_pre, balanced_professions_post, alternative='two-sided')

# Output the results
print(f"Wilcoxon test statistic for B: {stat_B}")
print(f"P-value: {p_value_B}")

# Interpret the results
alpha = 0.05  # significance level
if p_value_B < alpha:
    print("Reject the null hypothesis: There is a significant difference.")
else:
    print("Fail to reject the null hypothesis: No significant difference.")
