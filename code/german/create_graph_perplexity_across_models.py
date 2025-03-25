import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_model_perplexity_comparison(output_file="model_perplexity_comparison.png"):
    """
    Create a plot comparing perplexity across epochs for different models,
    using the averaged values across random seeds.
    """
    # Data from our calculations (averaged across random seeds)
    epochs = [0, 1, 2, 3]
    
    # Model data - excluding gelectra due to extreme values
    models_data = {
        'dbmdz': {
            'perplexity': [8.22, 8.76, 9.25, 8.96],
            'color': 'blue',
            'marker': 'o'
        },
        'google-bert': {
            'perplexity': [111.77, 9.78, 9.91, 9.64], 
            'color': 'orange',
            'marker': 's'
        },
        'gbert': {
            'perplexity': [15.69, 9.93, 10.30, 10.01],
            'color': 'green',
            'marker': '^'
        },
        'distilbert': {
            'perplexity': [14.18, 9.50, 9.95, 9.94],
            'color': 'red',
            'marker': 'D'
        }
    }
    
    # For stats table at the bottom
    stats_data = {}
    avg_baseline = 0
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
    
    # Plot each model
    for model_name, model_info in models_data.items():
        perplexity_values = model_info['perplexity']
        
        # Skip gelectra-like extreme values for better visualization
        if perplexity_values[0] > 500:
            continue
            
        # Store stats data
        stats_data[model_name] = {
            'Baseline (Epoch 0)': perplexity_values[0],
            'Final (Epoch 3)': perplexity_values[3],
            'Change': perplexity_values[3] - perplexity_values[0],
        }
        
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
            
        # Add to average baseline calculation (excluding extreme values)
        if perplexity_values[0] < 200:
            avg_baseline += perplexity_values[0]
            count += 1
        
        # Plot the line for this model
        ax.plot(epochs, perplexity_values, 
                marker=model_info['marker'], 
                color=model_info['color'],
                linewidth=2,
                markersize=8,
                label=model_name)
    
    # Calculate average baseline perplexity
    avg_baseline_ppl = avg_baseline / count if count > 0 else 0
    
    # Add horizontal line for average baseline perplexity
    ax.axhline(y=avg_baseline_ppl, color='red', linestyle='--', 
               label=f'Avg. Baseline Perplexity: {avg_baseline_ppl:.2f}')
    
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
    min_ppl_overall = min([min(model_info['perplexity']) for model_name, model_info in models_data.items()])
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
    
    # Format table data
    table_data = [
        ['Model', 'Baseline (Epoch 0)', 'Final (Epoch 3)', 'Change', 'Best Epoch', 'Best Perplexity']
    ]
    
    for model, row in stats_df.iterrows():
        table_data.append([
            model,
            f"{row['Baseline (Epoch 0)']:.2f}",
            f"{row['Final (Epoch 3)']:.2f}",
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
    
    # Create a secondary log-scale plot that includes gelectra
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    
    # Add gelectra to model data for log plot
    all_models_data = models_data.copy()
    all_models_data['gelectra'] = {
        'perplexity': [1.96e+6, 252.90, 180.25, 157.13],
        'color': 'purple',
        'marker': '*'
    }
    
    # Plot each model with log scale
    for model_name, model_info in all_models_data.items():
        perplexity_values = model_info['perplexity']
        ax2.plot(epochs, perplexity_values, 
                marker=model_info['marker'], 
                color=model_info['color'],
                linewidth=2,
                markersize=8,
                label=model_name)
    
    # Set log scale for y-axis
    ax2.set_yscale('log')
    
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

# Run the function to create the plots
plot_model_perplexity_comparison("german_models_perplexity_comparison.png")