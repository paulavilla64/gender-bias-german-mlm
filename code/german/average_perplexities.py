import pandas as pd
import numpy as np
import re
import os
from collections import defaultdict

# Keep original specific model and type setup
model_name = "dbmdz"
typ = "neutral"

def extract_seed(filename):
    """Extract the random seed from the checkpoint filename."""
    match = re.search(r'_(\d+)_epoch_', filename)
    if match:
        return match.group(1)
    return None

def average_perplexity_files(input_files, output_file):
    """
    Average perplexity values from multiple CSV files with different random seeds.
    
    Parameters:
    input_files (list): List of paths to CSV files with perplexity data
    output_file (str): Path to save the averaged CSV file
    """
    # Dictionary to store data for each epoch across all seeds
    epoch_data = defaultdict(list)
    
    # Read and process each file
    for file_path in input_files:
        try:
            df = pd.read_csv(file_path)
            
            # Process each row in the file (all epochs, not just 0 and 3)
            for _, row in df.iterrows():
                epoch = row['epoch']
                # Store the relevant data for this epoch
                epoch_data[epoch].append({
                    'validation_loss': row['validation_loss'],
                    'perplexity': row['perplexity'],
                    'checkpoint': row['checkpoint']
                })
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Calculate averages for each epoch
    results = []
    
    for epoch, data_list in sorted(epoch_data.items()):
        # Skip if we don't have data from all seeds for this epoch
        if len(data_list) < len(input_files):
            print(f"Warning: Epoch {epoch} only has data from {len(data_list)} files, expected {len(input_files)}")
        
        # Calculate average validation loss and perplexity
        avg_val_loss = np.mean([d['validation_loss'] for d in data_list])
        avg_perplexity = np.mean([d['perplexity'] for d in data_list])
        
        # Create a generic checkpoint name that indicates it's an average
        # Extract model name from the first checkpoint
        model_pattern = re.search(r'finetuned_(\w+)_', data_list[0]['checkpoint'])
        model_name = model_pattern.group(1) if model_pattern else "model"
        
        avg_checkpoint = f"finetuned_{model_name}_{typ}_avg_epoch_{int(epoch)}.pt"
        
        results.append({
            'checkpoint': avg_checkpoint,
            'epoch': int(epoch),
            'validation_loss': avg_val_loss,
            'perplexity': avg_perplexity
        })
    
    # Create and save the averaged data
    result_df = pd.DataFrame(results)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    
    print(f"Averaged data from {len(input_files)} files successfully written to {output_file}")
    return result_df

# Example usage
if __name__ == "__main__":
    # List of input files with perplexity data for different seeds
    input_files = [
        f"perplexity/{model_name}/results_DE_{model_name}_{typ}_42_perplexity.csv",
        f"perplexity/{model_name}/results_DE_{model_name}_{typ}_116_perplexity.csv",
        f"perplexity/{model_name}/results_DE_{model_name}_{typ}_387_perplexity.csv",
        f"perplexity/{model_name}/results_DE_{model_name}_{typ}_1980_perplexity.csv"
    ]
    
    # Output file path
    output_file = f"perplexity/{model_name}/results_DE_{model_name}_{typ}_perplexity_avg.csv"
    
    # Run the averaging function
    averaged_df = average_perplexity_files(input_files, output_file)
    
    # Display the results
    print("\nAveraged Results:")
    print(averaged_df.to_string(index=False))
    
    # Print improvement statistics if we have both baseline and other epochs
    if len(averaged_df) > 1 and 0 in averaged_df['epoch'].values:
        baseline = averaged_df.loc[averaged_df['epoch'] == 0, 'perplexity'].values[0]
        print(f"\nImprovement from baseline (Epoch 0, {baseline:.2f}):")
        
        for _, row in averaged_df.iterrows():
            if row['epoch'] > 0:
                improvement = ((baseline - row['perplexity']) / baseline) * 100
                status = "improved" if improvement > 0 else "worsened"
                print(f"  Epoch {int(row['epoch'])}: {row['perplexity']:.2f} ({status} by {abs(improvement):.2f}%)")