import pandas as pd
import os

typ = "neutral"

def create_comprehensive_associations_file(input_file, model_files, output_file, reorder_columns=True):
    """
    Create a comprehensive CSV with averaged pre and post associations for multiple models.
    
    Parameters:
    input_file (str): Path to the base input CSV file
    model_files (dict): Dictionary of model names and their association files
    output_file (str): Path for the output CSV file
    reorder_columns (bool): Whether to reorder columns for better readability
    
    Returns:
    pandas.DataFrame: DataFrame with all models' association scores
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist!")
        return None
        
    # Read the base input file
    print(f"Reading base input file: {input_file}")
    base_df = pd.read_csv(input_file, sep='\t')
    print(f"Base file has {len(base_df)} rows and columns: {base_df.columns.tolist()}")
    
    # Keep track of added columns
    added_pre_columns = []
    added_post_columns = []
    
    # Process each model
    for model_name, model_file in model_files.items():
        print(f"\nProcessing model: {model_name}")
        
        # Check if model file exists
        if not os.path.exists(model_file):
            print(f"Warning: Model file {model_file} does not exist. Skipping model {model_name}.")
            continue
            
        try:
            # Read the model's association file
            model_df = pd.read_csv(model_file, sep='\t')
            print(f"Model file has columns: {model_df.columns.tolist()}")
            
            # Create standard column names for this model's associations in output
            post_column_name = f'Post_Assoc_{model_name}_Avg'
            pre_column_name = f'Pre_Assoc_{model_name}_Avg'
            
            # Look for post-association column in model file
            post_cols = [col for col in model_df.columns 
                        if col.startswith('Post_Assoc') and (model_name in col or col == 'Post_Assoc')]
            
            if post_cols:
                # Found a matching column - use the first one
                post_source_col = post_cols[0]
                base_df[post_column_name] = model_df[post_source_col]
                added_post_columns.append(post_column_name)
                print(f"Added post-association from column: {post_source_col} as {post_column_name}")
            else:
                print(f"Warning: No Post_Assoc column found for model {model_name}")
            
            # Look for pre-association column in model file
            pre_cols = [col for col in model_df.columns 
                       if col.startswith('Pre_Assoc') and (model_name in col or col == 'Pre_Assoc')]
            
            if pre_cols:
                # Found a matching column - use the first one
                pre_source_col = pre_cols[0]
                base_df[pre_column_name] = model_df[pre_source_col]
                added_pre_columns.append(pre_column_name)
                print(f"Added pre-association from column: {pre_source_col} as {pre_column_name}")
            else:
                print(f"Warning: No Pre_Assoc column found for model {model_name}")
                
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
    
    # Reorder columns for better readability if requested
    if reorder_columns:
        # First, identify the base columns (non-association columns)
        base_columns = [col for col in base_df.columns 
                       if not col.startswith('Pre_Assoc') and not col.startswith('Post_Assoc')]
        
        # Create a new column order:
        # 1. Base columns (identity, descriptions)
        # 2. All Pre_Assoc columns together
        # 3. All Post_Assoc columns together
        new_column_order = base_columns + sorted(added_pre_columns) + sorted(added_post_columns)
        
        # Reorder the columns in the dataframe
        base_df = base_df[new_column_order]
        print(f"\nReordered columns for better readability")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the comprehensive file
    base_df.to_csv(output_file, sep='\t', index=False)
    print(f"\nSaved comprehensive file to {output_file}")
    print(f"Final file has {len(base_df)} rows and {len(base_df.columns)} columns")
    
    return base_df

# Define file paths
input_file = f'./regular/dbmdz/results_Lou_DE_regular_{typ}_dbmdz_avg.csv'
model_files = {
    'dbmdz': f'./regular/dbmdz/results_Lou_DE_regular_{typ}_dbmdz_avg.csv',
    'google-bert': f'./regular/google-bert/results_Lou_DE_regular_{typ}_google-bert_avg.csv',
    'deepset-bert': f'./regular/deepset-bert/results_Lou_DE_regular_{typ}_deepset-bert_avg.csv',
    'distilbert': f'./regular/distilbert/results_Lou_DE_regular_{typ}_distilbert_avg.csv',
}
output_file = f'./post_assocs/one_mask/associations_all_models_regular_{typ}_DE_one_mask_avg.csv'

# Create the comprehensive file with reordered columns
result_df = create_comprehensive_associations_file(input_file, model_files, output_file, reorder_columns=True)

if result_df is not None:
    print("\nComprehensive associations file created successfully!")
    
    # Print summary statistics
    print("\nSummary of model scores in comprehensive file:")
    
    # First all Pre_Assoc averages
    print("\nPre-Association Averages:")
    for model_name in model_files.keys():
        pre_col = f'Pre_Assoc_{model_name}_Avg'
        if pre_col in result_df.columns:
            pre_mean = result_df[pre_col].mean()
            pre_std = result_df[pre_col].std()
            print(f"  {model_name}: {pre_mean:.4f} (±{pre_std:.4f})")
    
    # Then all Post_Assoc averages
    print("\nPost-Association Averages:")
    for model_name in model_files.keys():
        post_col = f'Post_Assoc_{model_name}_Avg'
        if post_col in result_df.columns:
            post_mean = result_df[post_col].mean()
            post_std = result_df[post_col].std()
            print(f"  {model_name}: {post_mean:.4f} (±{post_std:.4f})")
    
    # Then differences
    print("\nDifferences (Post - Pre):")
    for model_name in model_files.keys():
        pre_col = f'Pre_Assoc_{model_name}_Avg'
        post_col = f'Post_Assoc_{model_name}_Avg'
        
        if pre_col in result_df.columns and post_col in result_df.columns:
            pre_mean = result_df[pre_col].mean()
            post_mean = result_df[post_col].mean()
            diff = post_mean - pre_mean
            
            print(f"  {model_name}: {diff:.4f} ({diff/pre_mean*100:.2f}%)")