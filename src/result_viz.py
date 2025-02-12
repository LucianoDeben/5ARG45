import pandas as pd
import matplotlib.pyplot as plt
import logging

# Read the CSV file that was written out.
# We assume that the CSV file contains a MultiIndex with two levels.
# Read the CSV file that was written out.
results_df = pd.read_csv("results_all_min_n_and_networks.csv", index_col=[0, 1])
results_df = results_df.reset_index()

# Since we are now using only Dorothea, update the plot title accordingly.
results_df["Network"] = results_df["Network_min_n"].apply(lambda x: "dorothea")
results_df["min_n"] = results_df["Network_min_n"].apply(lambda x: int(x.split("_")[3]))

# Create a dictionary to store (min_n, R²) pairs for each feature set.
performance_data = {}
for dataset_name, group in results_df.groupby("Dataset_and_Model"):
    group = group.sort_values("min_n")
    min_n_vals = group["min_n"].tolist()
    r2_vals = group["Pearson Correlation"].tolist()
    performance_data[dataset_name] = (min_n_vals, r2_vals)

plt.figure(figsize=(10, 6))
for dataset_name, (min_n_vals, r2_vals) in performance_data.items():
    plt.plot(min_n_vals, r2_vals, marker='o', label=dataset_name)

plt.xlabel("Target Gene Threshold Value")
plt.ylabel("Pearson Correlation")
plt.title("Performance vs. Target Threshold for Different Feature Sets (Dorothea)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("performance_vs_min_n_dorothea.png")
plt.show()

logging.info("Plot saved as performance_vs_min_n_dorothea.png")
print("Plot completed")
