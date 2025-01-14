import pandas as pd
from scipy.stats import wilcoxon

# Due to the ineffectiveness of the method for German, we only report on pre- and not on post-associations
# In order to statistically test the difference between associations for male and female person
# words, the Wilcoxon signed-rank test was again computed for each profession group separately.


# Read a CSV file
data = pd.read_csv('../data/results_DE.csv', sep='\t')

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


# Calculate the differences for F
differences_F = [after - before for before, after in zip(male_gender_in_female_professions, female_gender_in_female_professions)]
print(f"Differences_F: {differences_F}")

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
print(f"Differences_M: {differences_M}")

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
print(f"Differences_B: {differences_B}")

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
