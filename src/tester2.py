import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load data
print("Loading data...")
data = pd.read_csv("./data/raw/MixSeq/mixseq_lfc.csv")
print(f"Data shape: {data.shape}")

# Remove the first column (index)
print("Removing first column...")
data = data.iloc[:, 1:]
print(f"New data shape: {data.shape}")

# Define thresholds for analysis
UPPER_THRESHOLD = 10.0  # Values above this are considered extreme
VERY_EXTREME_THRESHOLD = 50.0  # Values above this are considered very extreme

# Find columns with max values above threshold
print("Finding columns with extreme values...")
max_vals = data.max()
extreme_cols = max_vals[max_vals > UPPER_THRESHOLD].sort_values(ascending=False)

print(f"\nFound {len(extreme_cols)} out of {data.shape[1]} columns with values > {UPPER_THRESHOLD}")
print(f"Top 20 columns with highest maximum values:")
print(extreme_cols.head(20))

# Calculate summary statistics for extreme values
total_cells = data.shape[0] * data.shape[1]
extreme_cells_count = (data > UPPER_THRESHOLD).sum().sum()
very_extreme_cells_count = (data > VERY_EXTREME_THRESHOLD).sum().sum()

print(f"\nTotal cells in dataset: {total_cells}")
print(f"Cells with values > {UPPER_THRESHOLD}: {extreme_cells_count} ({extreme_cells_count/total_cells:.6%})")
print(f"Cells with values > {VERY_EXTREME_THRESHOLD}: {very_extreme_cells_count} ({very_extreme_cells_count/total_cells:.6%})")

# Get counts of extreme values by column
print("\nBreakdown of extreme values by column:")
extreme_counts = {}
for col in extreme_cols.index:
    count = (data[col] > UPPER_THRESHOLD).sum()
    extreme_counts[col] = count

# Sort by count in descending order
extreme_counts = {k: v for k, v in sorted(extreme_counts.items(), key=lambda item: item[1], reverse=True)}

# Print the top 20 columns with most extreme values
print("\nTop 20 columns with the most cells containing extreme values:")
for i, (col, count) in enumerate(list(extreme_counts.items())[:20]):
    print(f"{i+1}. {col}: {count} cells ({count/data.shape[0]:.2%} of rows)")

# For the top 5 problematic columns, show their distribution
print("\nGenerating histograms for top 5 problematic columns...")
plt.figure(figsize=(15, 12))
for i, col in enumerate(list(extreme_counts.keys())[:5]):
    plt.subplot(3, 2, i+1)
    
    # Get values excluding extremes for better visualization
    values = data[col].values
    non_extreme_values = values[values <= UPPER_THRESHOLD]
    
    # Plot histogram
    plt.hist(non_extreme_values, bins=30, alpha=0.7, color='blue', label='Normal values')
    plt.axvline(x=UPPER_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({UPPER_THRESHOLD})')
    
    # Add text annotation about extreme values
    extreme_count = (values > UPPER_THRESHOLD).sum()
    max_value = values.max()
    plt.annotate(f"{extreme_count} values > {UPPER_THRESHOLD}\nMax value: {max_value:.2f}", 
                 xy=(0.7, 0.85), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
    
    plt.title(f"Distribution of {col}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    
plt.tight_layout()
plt.savefig('extreme_values_analysis.png')
print("Saved histogram plot to 'extreme_values_analysis.png'")

# Check if extreme values are concentrated in specific rows
print("\nAnalyzing distribution of extreme values across samples...")
extreme_values_per_row = (data > UPPER_THRESHOLD).sum(axis=1)
rows_with_extreme = extreme_values_per_row[extreme_values_per_row > 0]

print(f"Number of rows with at least one extreme value: {len(rows_with_extreme)} ({len(rows_with_extreme)/data.shape[0]:.2%} of all rows)")
print(f"Maximum number of extreme values in a single row: {extreme_values_per_row.max()}")

# Plot distribution of extreme values across rows
plt.figure(figsize=(12, 6))
plt.hist(extreme_values_per_row[extreme_values_per_row > 0], bins=30, alpha=0.7)
plt.title("Distribution of Extreme Values Across Rows")
plt.xlabel("Number of Extreme Values in Row")
plt.ylabel("Frequency")
plt.savefig('extreme_values_by_row.png')
print("Saved row distribution plot to 'extreme_values_by_row.png'")

# Analyze patterns in the data
if len(extreme_cols) > 0:
    print("\nChecking for patterns in extreme values...")
    
    # Get top problematic column
    worst_col = list(extreme_counts.keys())[0]
    
    # Find rows with extreme values in this column
    extreme_rows = data[data[worst_col] > UPPER_THRESHOLD].index.tolist()
    
    print(f"\nSample of rows with extreme values in '{worst_col}':")
    for row_idx in extreme_rows[:5]:
        max_val = data.loc[row_idx, worst_col]
        print(f"Row {row_idx}: Max value = {max_val:.2f}")
        
        # Calculate how many other columns also have extreme values in this row
        other_extreme = sum(1 for col in data.columns if col != worst_col and data.loc[row_idx, col] > UPPER_THRESHOLD)
        print(f"  - This row has extreme values in {other_extreme} other columns")

print("\nAnalysis complete.")