import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

def parse_perplexity_from_file(file_path):
    """
    Parse perplexity results from a text file with model evaluation output.
    
    Args:
        file_path (str): Path to the text file containing perplexity results
        
    Returns:
        pd.DataFrame: DataFrame containing the parsed results
    """
    results = []
    
    # Read the full file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract baseline perplexity
    baseline_pattern = r"Baseline Loss: (\d+\.\d+), Perplexity: (\d+\.\d+)"
    baseline_match = re.search(baseline_pattern, content)
    
    if baseline_match:
        baseline_loss = float(baseline_match.group(1))
        baseline_perplexity = float(baseline_match.group(2))
        
        # Add baseline to results
        results.append({
            'epoch': 0,
            'validation_loss': baseline_loss,
            'perplexity': baseline_perplexity
        })
    
    # Extract checkpoint results
    checkpoint_pattern = r"=== Processing checkpoint: finetuned_bert_epoch_(\d+)\.pt ===.*?Epoch \d+ Loss: (\d+\.\d+), Perplexity: (\d+\.\d+)"
    checkpoint_matches = re.finditer(checkpoint_pattern, content, re.DOTALL)
    
    for match in checkpoint_matches:
        epoch = int(match.group(1))
        val_loss = float(match.group(2))
        perplexity = float(match.group(3))
        
        results.append({
            'epoch': epoch,
            'validation_loss': val_loss,
            'perplexity': perplexity
        })
    
    return pd.DataFrame(results)

def plot_metrics(df, output_file):
    """
    Create and save a plot of loss and perplexity across epochs.
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'epoch', 'validation_loss', and 'perplexity'
        output_file (str): Path to save the output plot
    """
    # Sort DataFrame by epoch
    df = df.sort_values('epoch')
    
    # Set Seaborn style
    sns.set(style="whitegrid")
    
    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Validation Loss
    line1, = ax1.plot(df['epoch'], df['validation_loss'], marker='o', color='b', linewidth=2, label="Loss")
    
    # Create a second y-axis for Perplexity
    ax2 = ax1.twinx()
    line2, = ax2.plot(df['epoch'], df['perplexity'], marker='s', color='r', linewidth=2, label="Perplexity")
    
    # Set x-ticks to be integers
    ax1.set_xticks(df['epoch'])
    
    # Labels and Title
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Validation Loss", color='b', fontsize=12)
    ax2.set_ylabel("Perplexity", color='r', fontsize=12)
    plt.title("DBMDZ Fine-tuning Evaluation Metrics", fontsize=14)
    
    # Manually combine legends
    ax1.legend([line1, line2], ["Validation Loss", "Perplexity"], loc="upper right", fontsize=10)
    
    # Grid settings
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Show plot
    plt.show()


# Give arguments
results_df = parse_perplexity_from_file(file_path="../data/output_txt_files/perplexity/results_DE_gender_neutral_with_model_save_epochs_perplexity.txt")

# Check if results were found
if results_df.empty:
    print("No perplexity results found in the file.")

# Display the extracted data
print("\nExtracted data for plotting:")
print(results_df)

# Create and save the plot
plot_metrics(results_df, output_file="../data/plots/german/evaluation_metrics.png")
