import pandas as pd
import numpy as np

model_name = "deepset_bert"

typ = "neutral"

def average_post_associations(seed_files, output_file):
    """
    Read CSV files from different random seeds and compute average Post_Assoc.
    
    Parameters:
    seed_files (list): List of file paths for CSV files from different seeds
    output_file (str): Path for the output CSV file
    
    Returns:
    pandas.DataFrame: Averaged DataFrame
    """
    # Read all CSV files
    dataframes = [pd.read_csv(file, sep='\t') for file in seed_files]
    
    # Compute average of Post_Assoc
    post_assoc_avgs = []
    for i in range(len(dataframes[0])):
        # Collect Post_Assoc values for this row across all seeds
        row_values = [df.loc[i, 'Post_Assoc'] for df in dataframes]
        post_assoc_avgs.append(np.mean(row_values))
    
    # Take the first dataframe as base and add the averaged column
    merged_df = dataframes[0].copy()
    merged_df['Post_Assoc_Avg'] = post_assoc_avgs
    
    # Drop the original Post_Assoc column
    merged_df = merged_df.drop(columns=['Post_Assoc'])
    
    # Save to output file
    merged_df.to_csv(output_file, index=False, sep='\t')
    
    return merged_df


# Example usage
seed_files = [
    f'../../data/output_csv_files/german/Lou/{model_name}/results_Lou_DE_zero_difference_{typ}_deepset_bert_42.csv',
    f'../../data/output_csv_files/german/Lou/{model_name}/results_Lou_DE_zero_difference_{typ}_deepset_bert_116.csv',
    f'../../data/output_csv_files/german/Lou/{model_name}/results_Lou_DE_zero_difference_{typ}_deepset_bert_387.csv',
    f'../../data/output_csv_files/german/Lou/{model_name}/results_Lou_DE_zero_difference_{typ}_deepset_bert_1980.csv'
]

output_file = f'../../data/output_csv_files/german/Lou/{model_name}/results_Lou_DE_zero_difference_{typ}_deepset_bert_avg.csv'

# Call the function
result_df = average_post_associations(seed_files, output_file)

print("Averaged CSV created successfully!")