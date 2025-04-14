import pandas as pd

typ = "neutral"


def create_comprehensive_associations_file(input_file, model_files, output_file):
    """
    Create a comprehensive CSV with averaged pre and post associations for multiple models.
    
    Parameters:
    input_file (str): Path to the base input CSV file
    model_files (dict): Dictionary of model names and their association files
    output_file (str): Path for the output CSV file
    """
    # Read the base input file
    base_df = pd.read_csv(input_file, sep='\t')
    
    # Read and average associations for each model
    for model_name, model_file in model_files.items():
        # Read the model's association file
        model_df = pd.read_csv(model_file, sep='\t')
        
        # Create column names for this model's averaged associations
        post_column_name = f'Post_Assoc_{model_name}_Avg'
        pre_column_name = f'Pre_Assoc_{model_name}_Avg'
        
        # Add the averaged associations to the base dataframe
        base_df[post_column_name] = model_df['Post_Assoc_Avg']
        
        # Check if Pre_Assoc_Avg exists in the model dataframe
        if 'Pre_Assoc' in model_df.columns:
            base_df[pre_column_name] = model_df['Pre_Assoc']
        else:
            print(f"Warning: Pre_Assoc not found in {model_name} file. Skipping this model for pre-associations.")
    
    # Save the comprehensive file
    base_df.to_csv(output_file, sep='\t', index=False)
    
    return base_df

# Define file paths
input_file = f'./test_results_Lou_DE_zero_difference_{typ}_dbmdz_test_avg.csv'
model_files = {
    'dbmdz': f'./test_results_Lou_DE_zero_difference_{typ}_dbmdz_test_avg.csv',
    'google_bert': f'./test_results_Lou_DE_zero_difference_{typ}_google_bert_test_avg.csv',
    'deepset_bert': f'./test_results_Lou_DE_zero_difference_{typ}_deepset_bert_test_avg.csv',
    'distilbert': f'./test_results_Lou_DE_zero_difference_{typ}_distilbert_test_avg.csv'
}
output_file = f'./associations_all_models_DE_zero_difference_{typ}_test_avg.csv'

# Create the comprehensive file
result_df = create_comprehensive_associations_file(input_file, model_files, output_file)

print("Comprehensive associations file created successfully!")