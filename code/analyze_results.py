import math
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import wilcoxon
import pandas as pd
import numpy as np
# from scipy import stats

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


# Example usage with the provided results
results = {
    "balanced": {"wilcoxon_statistic": 90393, "n": 1800},
    "female": {"wilcoxon_statistic": 100814, "n": 1800},
    "male": {"wilcoxon_statistic": 107845, "n": 1800},
}

# Compute Cohen's W for each group and display results
for group, values in results.items():
    wilcoxon_statistic = values["wilcoxon_statistic"]
    N = values["n"]

    effect_size = compute_cohens_w(wilcoxon_statistic, N)
    print(f"{group.capitalize()} group - Cohen's W effect size: {effect_size}")

