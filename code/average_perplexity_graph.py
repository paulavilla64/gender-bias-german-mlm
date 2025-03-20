import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Creating DataFrames for each random seed
seed_42_data = pd.DataFrame({
    'checkpoint': ['finetuned_bert_42_epoch_0.pt', 'finetuned_bert_42_epoch_1.pt', 
                   'finetuned_bert_42_epoch_2.pt', 'finetuned_bert_42_epoch_3.pt', 'final_model'],
    'epoch': [0, 1, 2, 3, 'final'],
    'validation_loss': [3.3151, 3.282703, 3.238998, 3.260289, 3.260289],
    'perplexity': [27.53, 26.647694, 25.508152, 26.057075, 26.057075]
})

seed_116_data = pd.DataFrame({
    'checkpoint': ['finetuned_bert_116_epoch_0.pt', 'finetuned_bert_116_epoch_1.pt', 
                   'finetuned_bert_116_epoch_2.pt', 'finetuned_bert_116_epoch_3.pt', 'final_model'],
    'epoch': [0, 1, 2, 3, 'final'],
    'validation_loss': [3.306466, 3.210497, 3.276864, 3.288094, 3.288094],
    'perplexity': [27.288531, 24.791394, 26.492575, 26.791744, 26.791744]
})

seed_387_data = pd.DataFrame({
    'checkpoint': ['finetuned_bert_387_epoch_0.pt', 'finetuned_bert_387_epoch_1.pt', 
                   'finetuned_bert_387_epoch_2.pt', 'finetuned_bert_387_epoch_3.pt', 'final_model'],
    'epoch': [0, 1, 2, 3, 'final'],
    'validation_loss': [3.318822, 3.272234, 3.250521, 3.269582, 3.269582],
    'perplexity': [27.627791, 26.370181, 25.803769, 26.300334, 26.300334]
})

seed_1980_data = pd.DataFrame({
    'checkpoint': ['finetuned_bert_1980_epoch_0.pt', 'finetuned_bert_1980_epoch_1.pt', 
                   'finetuned_bert_1980_epoch_2.pt', 'finetuned_bert_1980_epoch_3.pt', 'final_model'],
    'epoch': [0, 1, 2, 3, 'final'],
    'validation_loss': [3.322012, 3.293379, 3.337455, 3.270470, 3.270470],
    'perplexity': [27.716069, 26.933706, 28.147401, 26.323703, 26.323703]
})

# Filter out the final model rows and convert epoch to numeric for plotting
def prepare_data(df):
    return df[df['epoch'] != 'final'].copy()

seed_42 = prepare_data(seed_42_data)
seed_42['epoch'] = seed_42['epoch'].astype(int)

seed_116 = prepare_data(seed_116_data)
seed_116['epoch'] = seed_116['epoch'].astype(int)

seed_387 = prepare_data(seed_387_data)
seed_387['epoch'] = seed_387['epoch'].astype(int)

seed_1980 = prepare_data(seed_1980_data)
seed_1980['epoch'] = seed_1980['epoch'].astype(int)

# Calculate average baseline perplexity (at epoch 0)
baseline_perplexities = [
    seed_42['perplexity'].iloc[0],
    seed_116['perplexity'].iloc[0],
    seed_387['perplexity'].iloc[0],
    seed_1980['perplexity'].iloc[0]
]
avg_baseline = np.mean(baseline_perplexities)

# Create the figure and plot the data
plt.figure(figsize=(12, 8))

# Plot perplexity for each random seed
plt.plot(seed_42['epoch'], seed_42['perplexity'], marker='o', linewidth=2, label='Seed 42')
plt.plot(seed_116['epoch'], seed_116['perplexity'], marker='s', linewidth=2, label='Seed 116')
plt.plot(seed_387['epoch'], seed_387['perplexity'], marker='^', linewidth=2, label='Seed 387')
plt.plot(seed_1980['epoch'], seed_1980['perplexity'], marker='d', linewidth=2, label='Seed 1980')

# Add baseline perplexity reference line
plt.axhline(y=avg_baseline, color='r', linestyle='--', alpha=0.7, label=f'Avg. Baseline Perplexity: {avg_baseline:.2f}')

# Mark baseline perplexities
plt.scatter([0, 0, 0, 0], baseline_perplexities, color='black', s=100, alpha=0.6)

# Customize the plot appearance
plt.title('Perplexity Comparison Across Different Random Seeds and Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Perplexity', fontsize=14)
plt.xticks(range(0, 4))
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Set y-axis limits for better visualization
plt.ylim(24.5, 28.5)

# Add annotations for best perplexity for each seed
best_perplexities = {
    'Seed 42': (seed_42['perplexity'].min(), seed_42.loc[seed_42['perplexity'].idxmin(), 'epoch']),
    'Seed 116': (seed_116['perplexity'].min(), seed_116.loc[seed_116['perplexity'].idxmin(), 'epoch']),
    'Seed 387': (seed_387['perplexity'].min(), seed_387.loc[seed_387['perplexity'].idxmin(), 'epoch']),
    'Seed 1980': (seed_1980['perplexity'].min(), seed_1980.loc[seed_1980['perplexity'].idxmin(), 'epoch'])
}

# Find overall best perplexity
best_seed = min(best_perplexities.items(), key=lambda x: x[1][0])
plt.annotate(f'Best: {best_seed[0]}, Epoch {int(best_seed[1][1])}, {best_seed[1][0]:.2f}',
             xy=(best_seed[1][1], best_seed[1][0]), xytext=(best_seed[1][1], best_seed[1][0] - 0.7),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

# Create a summary table below the plot
summary_data = [
    ['Random Seed', 'Baseline (Epoch 0)', 'Final (Epoch 3)', 'Change', 'Best Epoch', 'Best Perplexity'],
    ['42', f"{seed_42['perplexity'].iloc[0]:.2f}", f"{seed_42['perplexity'].iloc[3]:.2f}", 
     f"{seed_42['perplexity'].iloc[3] - seed_42['perplexity'].iloc[0]:.2f}", 
     f"{seed_42['perplexity'].idxmin()}", f"{seed_42['perplexity'].min():.2f}"],
    ['116', f"{seed_116['perplexity'].iloc[0]:.2f}", f"{seed_116['perplexity'].iloc[3]:.2f}", 
     f"{seed_116['perplexity'].iloc[3] - seed_116['perplexity'].iloc[0]:.2f}", 
     f"{seed_116['perplexity'].idxmin()}", f"{seed_116['perplexity'].min():.2f}"],
    ['387', f"{seed_387['perplexity'].iloc[0]:.2f}", f"{seed_387['perplexity'].iloc[3]:.2f}", 
     f"{seed_387['perplexity'].iloc[3] - seed_387['perplexity'].iloc[0]:.2f}", 
     f"{seed_387['perplexity'].idxmin()}", f"{seed_387['perplexity'].min():.2f}"],
    ['1980', f"{seed_1980['perplexity'].iloc[0]:.2f}", f"{seed_1980['perplexity'].iloc[3]:.2f}", 
     f"{seed_1980['perplexity'].iloc[3] - seed_1980['perplexity'].iloc[0]:.2f}", 
     f"{seed_1980['perplexity'].idxmin()}", f"{seed_1980['perplexity'].min():.2f}"]
]

# Add text below plot for summary stats
table_text = ''
for row in summary_data:
    table_text += '  '.join(str(cell).ljust(20) for cell in row) + '\n'

plt.figtext(0.5, -0.05, table_text, ha="center", fontfamily="monospace", 
            bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # Make room for the table
plt.savefig('perplexity_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Alternative: Create a subplot with the table as a separate axes
def create_figure_with_table():
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot perplexity for each random seed
    ax.plot(seed_42['epoch'], seed_42['perplexity'], marker='o', linewidth=2, label='Seed 42')
    ax.plot(seed_116['epoch'], seed_116['perplexity'], marker='s', linewidth=2, label='Seed 116')
    ax.plot(seed_387['epoch'], seed_387['perplexity'], marker='^', linewidth=2, label='Seed 387')
    ax.plot(seed_1980['epoch'], seed_1980['perplexity'], marker='d', linewidth=2, label='Seed 1980')
    
    # Add baseline perplexity reference line
    ax.axhline(y=avg_baseline, color='r', linestyle='--', alpha=0.7, label=f'Avg. Baseline: {avg_baseline:.2f}')
    
    # Customize the plot appearance
    ax.set_title('Perplexity Comparison Across Different Random Seeds and Epochs', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Perplexity', fontsize=14)
    ax.set_xticks(range(0, 4))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_ylim(24.5, 28.5)
    
    # Format table data
    column_labels = ['Random Seed', 'Baseline\n(Epoch 0)', 'Final\n(Epoch 3)', 'Change', 'Best\nEpoch', 'Best\nPerplexity']
    row_labels = ['42', '116', '387', '1980']
    table_data = [
        [f"{seed_42['perplexity'].iloc[0]:.2f}", f"{seed_42['perplexity'].iloc[3]:.2f}", 
         f"{seed_42['perplexity'].iloc[3] - seed_42['perplexity'].iloc[0]:.2f}", 
         f"{seed_42['perplexity'].idxmin()}", f"{seed_42['perplexity'].min():.2f}"],
        [f"{seed_116['perplexity'].iloc[0]:.2f}", f"{seed_116['perplexity'].iloc[3]:.2f}", 
         f"{seed_116['perplexity'].iloc[3] - seed_116['perplexity'].iloc[0]:.2f}", 
         f"{seed_116['perplexity'].idxmin()}", f"{seed_116['perplexity'].min():.2f}"],
        [f"{seed_387['perplexity'].iloc[0]:.2f}", f"{seed_387['perplexity'].iloc[3]:.2f}", 
         f"{seed_387['perplexity'].iloc[3] - seed_387['perplexity'].iloc[0]:.2f}", 
         f"{seed_387['perplexity'].idxmin()}", f"{seed_387['perplexity'].min():.2f}"],
        [f"{seed_1980['perplexity'].iloc[0]:.2f}", f"{seed_1980['perplexity'].iloc[3]:.2f}", 
         f"{seed_1980['perplexity'].iloc[3] - seed_1980['perplexity'].iloc[0]:.2f}", 
         f"{seed_1980['perplexity'].idxmin()}", f"{seed_1980['perplexity'].min():.2f}"]
    ]
    
    # Add a table at the bottom
    table = ax.table(cellText=table_data,
                    rowLabels=row_labels,
                    colLabels=column_labels,
                    cellLoc='center',
                    loc='bottom',
                    bbox=[0, -0.50, 1, 0.30])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)  # Make room for the table
    plt.savefig('perplexity_comparison_with_table.png', dpi=300, bbox_inches='tight')
    plt.show()

# Uncomment to use the alternative figure with embedded table
# create_figure_with_table()


# Function to create a more comprehensive analysis
def analyze_perplexity_results():
    """
    Create a comprehensive analysis of perplexity results across different seeds
    """
    # Combine all data for easier analysis
    all_data = pd.DataFrame({
        'epoch': list(range(4)) * 4,
        'seed': ['42'] * 4 + ['116'] * 4 + ['387'] * 4 + ['1980'] * 4,
        'perplexity': list(seed_42['perplexity']) + list(seed_116['perplexity']) + 
                       list(seed_387['perplexity']) + list(seed_1980['perplexity'])
    })
    
    # Analysis by epoch
    epoch_stats = all_data.groupby('epoch')['perplexity'].agg(['mean', 'std', 'min', 'max'])
    epoch_stats.columns = ['Mean Perplexity', 'Std Dev', 'Min Perplexity', 'Max Perplexity']
    
    print("Average performance by epoch:")
    print(epoch_stats)
    print("\n")
    
    # Find best performing seed at each epoch
    best_by_epoch = all_data.loc[all_data.groupby('epoch')['perplexity'].idxmin()]
    print("Best performing seed by epoch:")
    print(best_by_epoch[['epoch', 'seed', 'perplexity']])
    print("\n")
    
    # Find overall best performance
    overall_best = all_data.loc[all_data['perplexity'].idxmin()]
    print("Overall best performance:")
    print(f"Seed {overall_best['seed']} at epoch {overall_best['epoch']} with perplexity {overall_best['perplexity']:.3f}")
    
    # Create visualization of epoch-wise statistics
    plt.figure(figsize=(12, 6))
    
    # Plot mean perplexity with error bars
    plt.errorbar(epoch_stats.index, epoch_stats['Mean Perplexity'], 
                 yerr=epoch_stats['Std Dev'], fmt='o-', capsize=5, 
                 label='Mean Perplexity (with Std Dev)')
    
    # Add min/max range as a shaded area
    plt.fill_between(epoch_stats.index, 
                     epoch_stats['Min Perplexity'], 
                     epoch_stats['Max Perplexity'], 
                     alpha=0.2, color='blue', label='Min-Max Range')
    
    plt.title('Perplexity Statistics by Epoch Across All Seeds', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.xticks(range(4))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('perplexity_statistics_by_epoch.png', dpi=300)
    plt.show()

# Uncomment to run the analysis function
analyze_perplexity_results()