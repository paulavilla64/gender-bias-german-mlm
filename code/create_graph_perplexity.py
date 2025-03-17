import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
epochs = [0, 1, 2, 3]
loss_values = [3.31, 3.23, 3.28, 3.26]
perplexity_values = [27.38, 25.31, 26.64, 26.00]

# Create a DataFrame
df = pd.DataFrame({
    'Epoch': epochs,
    'Average Loss': loss_values,
    'Perplexity': perplexity_values
})

# Set Seaborn style
sns.set(style="whitegrid")

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Average Loss
line1, = ax1.plot(df['Epoch'], df['Average Loss'], marker='o', color='b', label="Loss")

# Create a second y-axis for Perplexity
ax2 = ax1.twinx()
line2, = ax2.plot(df['Epoch'], df['Perplexity'], marker='s', color='r', label="Perplexity")

# Labels and Title
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Average Loss", color='b')
ax2.set_ylabel("Perplexity", color='r')
plt.title("Evaluation Metrics on Validation Set Across Epochs")

# Manually combine legends
ax1.legend([line1, line2], ["Loss", "Perplexity"], loc="upper right")

# Save the figure
plt.savefig("validation_metrics.png", dpi=300, bbox_inches='tight')
plt.show()
