import pandas as pd
import numpy as np
import os

# Define constants
typ = "neutral"
model_names = ['dbmdz', 'google-bert', 'deepset-bert', 'distilbert']

def average_associations_across_seeds(seed_files, output_file):
    """
    Read CSV files from different random seeds and compute average Pre_Assoc and Post_Assoc for each model.
    Creates new columns with "_Avg" suffix containing the averages.
    
    Parameters:
    seed_files (list): List of file paths for CSV files from different seeds
    output_file (str): Path for the output CSV file
    
    Returns:
    pandas.DataFrame: DataFrame with averaged pre and post associations
    """
    # Check if files exist
    existing_files = []
    for file in seed_files:
        if os.path.exists(file):
            existing_files.append(file)
        else:
            print(f"Warning: File {file} does not exist")
    
    if not existing_files:
        print("Error: No valid files found")
        return None
        
    # Read all CSV files
    print(f"Reading {len(existing_files)} files...")
    dataframes = []
    for file in existing_files:
        try:
            df = pd.read_csv(file, sep="\t")
            print(f"Read {file} with {len(df)} rows")
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dataframes:
        print("No data could be read")
        return None
    
    # Take the first dataframe as base for the result
    result_df = dataframes[0].copy()
    
    # For each model, compute average Pre_Assoc and Post_Assoc across all seeds
    for model_name in model_names:
        print(f"\nProcessing model: {model_name}")
        
        # Column names for this model
        pre_col = f"Pre_Assoc_{model_name}"
        post_col = f"Post_Assoc_{model_name}"
        
        # New column names for the averages
        pre_avg_col = f"{pre_col}_Avg"
        post_avg_col = f"{post_col}_Avg"
        
        # Check if columns exist in all dataframes
        missing_pre = [i for i, df in enumerate(dataframes) if pre_col not in df.columns]
        missing_post = [i for i, df in enumerate(dataframes) if post_col not in df.columns]
        
        if missing_pre:
            print(f"Warning: {pre_col} missing in {len(missing_pre)} files")
        if missing_post:
            print(f"Warning: {post_col} missing in {len(missing_post)} files")
        
        if len(missing_pre) == len(dataframes) or len(missing_post) == len(dataframes):
            print(f"Skipping {model_name}: columns not found in any file")
            continue
        
        # Collect pre and post association values for each row across all seeds
        pre_avg_values = []
        post_avg_values = []
        
        for row_idx in range(len(result_df)):
            # Get pre-association values for this row from all dataframes that have the column
            pre_values = [df.loc[row_idx, pre_col] for df in dataframes 
                         if pre_col in df.columns and row_idx < len(df)]
            
            # Get post-association values for this row from all dataframes that have the column
            post_values = [df.loc[row_idx, post_col] for df in dataframes 
                          if post_col in df.columns and row_idx < len(df)]
            
            # Compute averages
            if pre_values:
                pre_avg_values.append(np.mean(pre_values))
            else:
                pre_avg_values.append(np.nan)
                
            if post_values:
                post_avg_values.append(np.mean(post_values))
            else:
                post_avg_values.append(np.nan)
        
        # Add the averaged columns to result dataframe
        result_df[pre_avg_col] = pre_avg_values
        result_df[post_avg_col] = post_avg_values
        
        # Print some stats
        if pre_avg_values and post_avg_values:
            print(f"  Pre-assoc average: {np.nanmean(pre_avg_values):.4f}")
            print(f"  Post-assoc average: {np.nanmean(post_avg_values):.4f}")
            print(f"  Difference: {np.nanmean(post_avg_values) - np.nanmean(pre_avg_values):.4f}")
    
    # Reorder columns to group the averaged columns together
    # First, identify all column categories
    meta_cols = [col for col in result_df.columns if not (col.startswith('Pre_Assoc') or col.startswith('Post_Assoc'))]
    pre_cols = [col for col in result_df.columns if col.startswith('Pre_Assoc') and not col.endswith('_Avg')]
    post_cols = [col for col in result_df.columns if col.startswith('Post_Assoc') and not col.endswith('_Avg')]
    pre_avg_cols = [col for col in result_df.columns if col.startswith('Pre_Assoc') and col.endswith('_Avg')]
    post_avg_cols = [col for col in result_df.columns if col.startswith('Post_Assoc') and col.endswith('_Avg')]
    
    # Optional: reorder columns (metadata, then all pre_avg, then all post_avg)
    reordered_cols = meta_cols + sorted(pre_avg_cols) + sorted(post_avg_cols)
    
    # Only reorder if all original columns are accounted for
    if len(reordered_cols) == len(result_df.columns) - len(pre_cols) - len(post_cols):
        result_df = result_df[reordered_cols]
    
    # Save to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_csv(output_file, index=False, sep='\t')
    print(f"\nSaved output to {output_file}")
    
    return result_df

# Define file paths
seed_files = [
    f'../data/output_csv_files/german/Lou/post_assocs/one_mask/post_assoc_all_{typ}_DE_regular_one_mask_42.csv',
    f'../data/output_csv_files/german/Lou/post_assocs/one_mask/post_assoc_all_{typ}_DE_regular_one_mask_116.csv',
    f'../data/output_csv_files/german/Lou/post_assocs/one_mask/post_assoc_all_{typ}_DE_regular_one_mask_387.csv',
    f'../data/output_csv_files/german/Lou/post_assocs/one_mask/post_assoc_all_{typ}_DE_regular_one_mask_1980.csv'
]

# Create output directory if it doesn't exist
output_dir = f'../data/output_csv_files/german/Lou/post_assocs/one_mask/regular'
os.makedirs(output_dir, exist_ok=True)
output_file = f'{output_dir}/results_Lou_{typ}_DE_regular_avg_one_mask.csv'

# Call the function
result_df = average_associations_across_seeds(seed_files, output_file)

if result_df is not None:
    print("\nAveraged CSV created successfully!")
    print(f"Output file saved to: {output_file}")
    
    # Print summary stats for all models
    print("\nSummary statistics for all models:")
    for model_name in model_names:
        pre_avg_col = f"Pre_Assoc_{model_name}_Avg"
        post_avg_col = f"Post_Assoc_{model_name}_Avg"
        
        if pre_avg_col in result_df.columns and post_avg_col in result_df.columns:
            print(f"\n{model_name}:")
            print(f"  Pre-assoc average: {result_df[pre_avg_col].mean():.4f}")
            print(f"  Post-assoc average: {result_df[post_avg_col].mean():.4f}")
            print(f"  Difference: {result_df[post_avg_col].mean() - result_df[pre_avg_col].mean():.4f}")
else:
    print("Failed to create averaged CSV!")