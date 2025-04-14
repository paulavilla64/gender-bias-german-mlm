import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

typ = "neutral"

def plot_model_perplexity_comparison_from_csv(csv_files, output_file=f"{typ}_models_perplexity_comparison_lou.png"):
    """
    Create a plot comparing perplexity across epochs for different models,
    reading data from CSV files.
    
    Parameters:
    csv_files (dict): Dictionary mapping model names to CSV file paths
    output_file (str): Path for saving the output plot
    """
    # Set up data structures
    models_data = {}
    epochs = []
    
    # Process each model's CSV file
    for model_name, csv_path in csv_files.items():
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Sort by epoch
            df = df.sort_values('epoch')
            
            # Extract perplexity values
            perplexity_values = df['perplexity'].tolist()
            
            # Update list of epochs if needed
            model_epochs = df['epoch'].tolist()
            epochs = model_epochs if len(model_epochs) > len(epochs) else epochs
            
            # Assign color and marker based on model name
            color_marker_map = {
                'dbmdz': {'color': 'blue', 'marker': 'o'},
                'google-bert': {'color': 'orange', 'marker': 's'},
                'deepset-bert': {'color': 'green', 'marker': '^'},
                'distilbert': {'color': 'red', 'marker': 'D'},
            }
            
            # Default color/marker if model not in map
            color = color_marker_map.get(model_name, {}).get('color', 'gray')
            marker = color_marker_map.get(model_name, {}).get('marker', 'x')
            
            # Store model data
            models_data[model_name] = {
                'perplexity': perplexity_values,
                'color': color,
                'marker': marker
            }
            
            print(f"Loaded data for {model_name}: {perplexity_values}")
            
        except Exception as e:
            print(f"Error loading data for {model_name} from {csv_path}: {e}")
    
    # Skip plotting if no data was loaded
    if not models_data:
        print("No valid model data found. Exiting.")
        return
    
    # For stats table
    stats_data = {}
    avg_baseline = 0
    avg_post_finetuning = 0
    count = 0
    
    # Find best model and lowest perplexity
    best_model = None
    best_epoch = None
    best_perplexity = float('inf')
    
    # Create a new figure with specific size
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Set up the plot aesthetics
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Perplexity Comparison Across Different Models and Epochs', fontsize=14)
    
    # Models to exclude from standard plot (will be in log plot only)
    exclude_from_standard = []
    
    # First pass: identify extreme values models
    for model_name, model_info in models_data.items():
        if len(model_info['perplexity']) > 0 and model_info['perplexity'][0] > 500:
            exclude_from_standard.append(model_name)
    
    # Plot each model
    for model_name, model_info in models_data.items():
        # Skip models with extreme values in the standard plot
        if model_name in exclude_from_standard:
            continue
            
        perplexity_values = model_info['perplexity']
        
        # Skip if no perplexity values
        if not perplexity_values:
            continue
        
        # If model doesn't have values for all epochs, continue with what it has
        model_epochs = epochs[:len(perplexity_values)]
        
        # Store stats data for all epochs
        stats_data[model_name] = {
            'Baseline (Epoch 0)': perplexity_values[0],
            'Epoch 1': perplexity_values[1] if len(perplexity_values) > 1 else None,
            'Epoch 2': perplexity_values[2] if len(perplexity_values) > 2 else None,
            'Epoch 3': perplexity_values[3] if len(perplexity_values) > 3 else None,
            'Change': perplexity_values[-1] - perplexity_values[0] if len(perplexity_values) > 0 else 0,
        }
        
        # Calculate average post-finetuning perplexity (epochs 1 onward)
        post_finetuning_avg = np.mean(perplexity_values[1:]) if len(perplexity_values) > 1 else 0
        stats_data[model_name]['Avg Post-Finetuning'] = post_finetuning_avg
        
        # Find best epoch and perplexity for this model
        min_ppl = min(perplexity_values)
        min_epoch = perplexity_values.index(min_ppl)
        stats_data[model_name]['Best Epoch'] = min_epoch
        stats_data[model_name]['Best Perplexity'] = min_ppl
        
        # Track global best
        if min_ppl < best_perplexity:
            best_perplexity = min_ppl
            best_epoch = min_epoch
            best_model = model_name
            
        # Add to average baseline and post-finetuning calculations
        if perplexity_values[0] < 200:
            avg_baseline += perplexity_values[0]
            avg_post_finetuning += post_finetuning_avg
            count += 1
        
        # Plot the line for this model
        ax.plot(model_epochs, perplexity_values, 
                marker=model_info['marker'], 
                color=model_info['color'],
                linewidth=2,
                markersize=8,
                label=model_name)
        
        # Add perplexity values as text labels
        for i, val in enumerate(perplexity_values):
            ax.annotate(f'{val:.2f}', 
                       (model_epochs[i], val), 
                       textcoords="offset points",
                       xytext=(0,10), 
                       ha='center',
                       fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Calculate average baseline and post-finetuning perplexity
    avg_baseline_ppl = avg_baseline / count if count > 0 else 0
    avg_post_finetuning_ppl = avg_post_finetuning / count if count > 0 else 0
    
    # Add horizontal line for average baseline perplexity
    ax.axhline(y=avg_baseline_ppl, color='red', linestyle='--', 
               label=f'Avg. Baseline Perplexity: {avg_baseline_ppl:.2f}')
    
    # Add horizontal line for average post-finetuning perplexity
    ax.axhline(y=avg_post_finetuning_ppl, color='green', linestyle='--', 
               label=f'Avg. Post-Finetuning: {avg_post_finetuning_ppl:.2f}')
    
    # Mark the best model/epoch with an arrow and annotation
    if best_model and best_epoch is not None:
        best_ppl = models_data[best_model]['perplexity'][best_epoch]
        ax.annotate(f'Best: {best_model}, Epoch {best_epoch}, {best_ppl:.2f}',
                   xy=(best_epoch, best_ppl),
                   xytext=(best_epoch, best_ppl - 1.5),  # offset text position
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle="->", color='black'),
                   ha='center')
    
    # Adjust y-axis to focus on the important range
    min_ppl_values = [min(model_info['perplexity']) for model_name, model_info 
                      in models_data.items() if model_name not in exclude_from_standard
                      and len(model_info['perplexity']) > 0]
    
    if min_ppl_values:
        min_ppl_overall = min(min_ppl_values)
        max_display_ppl = 30  # Limit the upper y-range to 30 for better visibility
        ax.set_ylim(min_ppl_overall - 0.5, max_display_ppl)
    
    # Set x-ticks to integers
    ax.set_xticks(epochs)
    
    # Add legend
    ax.legend(loc='best', fontsize=10)
    
    # Create DataFrame for stats table
    stats_df = pd.DataFrame.from_dict(stats_data, orient='index')
    
    # Sort by best perplexity
    stats_df = stats_df.sort_values('Best Perplexity')
    
    # Format table data with all epochs
    table_data = [
        ['Model', 'Epoch 0', 'Epoch 1', 'Epoch 2', 'Epoch 3', 'Avg Post-Fine', 'Change', 'Best Epoch', 'Best PPL']
    ]
    
    for model, row in stats_df.iterrows():
        table_data.append([
            model,
            f"{row['Baseline (Epoch 0)']:.2f}",
            f"{row['Epoch 1']:.2f}" if pd.notnull(row['Epoch 1']) else "N/A",
            f"{row['Epoch 2']:.2f}" if pd.notnull(row['Epoch 2']) else "N/A",
            f"{row['Epoch 3']:.2f}" if pd.notnull(row['Epoch 3']) else "N/A",
            f"{row['Avg Post-Finetuning']:.2f}",
            f"{row['Change']:.2f}",
            f"{int(row['Best Epoch'])}",
            f"{row['Best Perplexity']:.2f}"
        ])
    
    # Add table at the bottom
    plt.subplots_adjust(bottom=0.25)  # Make room for the table
    table = plt.table(cellText=table_data,
                     loc='bottom',
                     cellLoc='center',
                     bbox=[0, -0.5, 1, 0.3])  # [left, bottom, width, height]
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)  # Adjust table scaling
    
    # Tight layout (will respect the subplots_adjust)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Create a secondary log-scale plot that includes all models
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    
    # Calculate average post-finetuning perplexity for all models including extreme values
    avg_post_finetuning_all = 0
    count_all = 0
    
    for model_name, model_info in models_data.items():
        if len(model_info['perplexity']) > 1:
            post_finetuning_avg = np.mean(model_info['perplexity'][1:])
            avg_post_finetuning_all += post_finetuning_avg
            count_all += 1
    
    avg_post_finetuning_all_ppl = avg_post_finetuning_all / count_all if count_all > 0 else 0
    
    # Plot each model with log scale, including all models
    for model_name, model_info in models_data.items():
        perplexity_values = model_info['perplexity']
        
        # Skip if no perplexity values
        if not perplexity_values:
            continue
            
        # If model doesn't have values for all epochs, continue with what it has
        model_epochs = epochs[:len(perplexity_values)]
        
        ax2.plot(model_epochs, perplexity_values, 
                marker=model_info['marker'], 
                color=model_info['color'],
                linewidth=2,
                markersize=8,
                label=model_name)
        
        # Add perplexity values as text labels in log plot too
        for i, val in enumerate(perplexity_values):
            # For log scale, adjust vertical offset based on value
            y_offset = 0.2 * val if val < 100 else 0.05 * val
            ax2.annotate(f'{val:.2f}', 
                       (model_epochs[i], val), 
                       textcoords="offset points",
                       xytext=(0,10), 
                       ha='center',
                       fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Set log scale for y-axis
    ax2.set_yscale('log')
    
    # Add horizontal line for average post-finetuning perplexity (all models)
    if avg_post_finetuning_all_ppl > 0:
        ax2.axhline(y=avg_post_finetuning_all_ppl, color='green', linestyle='--', 
                   label=f'Avg. Post-Finetuning (All Models): {avg_post_finetuning_all_ppl:.2f}')
    
    # Set up the plot aesthetics
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Perplexity (log scale)', fontsize=12)
    ax2.set_title('Perplexity Comparison Across All Models and Epochs (Log Scale)', fontsize=14)
    
    # Set x-ticks to integers
    ax2.set_xticks(epochs)
    
    # Add legend
    ax2.legend(loc='best', fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the log-scale figure
    log_scale_output = output_file.replace('.png', '_log_scale.png')
    plt.savefig(log_scale_output, dpi=300, bbox_inches='tight')
    print(f"Log scale plot saved to {log_scale_output}")

# Example usage
if __name__ == "__main__":
    # Define paths to CSV files for each model
    csv_files = {
        'dbmdz': f'perplexity/dbmdz/results_DE_dbmdz_{typ}_perplexity_avg.csv',
        'google-bert': f'perplexity/google_bert/results_DE_google_bert_{typ}_perplexity_avg.csv',
        'deepset-bert': f'perplexity/deepset_bert/results_DE_deepset_bert_{typ}_perplexity_avg.csv',
        'distilbert': f'perplexity/distilbert/results_DE_distilbert_{typ}_perplexity_avg.csv',
    }
    
    # Create the plot
    plot_model_perplexity_comparison_from_csv(csv_files, f"{typ}_models_perplexity_comparison_lou.png")

