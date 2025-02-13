# %% Import required libraries and modules
import logging
import os
import sys
import warnings

from sklearn.preprocessing import StandardScaler

# Suppress all FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)

import decoupler as dc
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from evaluation import evaluate_model
from models import FlexibleFCNN

# Use our unified splitting and k_fold functions.
from preprocess import (
    k_fold_cross_validation,
    run_tf_activity_inference,
    split_data_flexible,
)
from training import train_model
from utils import (
    create_dataloader,
    create_dataset,
    load_config,
    load_sampled_data,
    sanity_check,
)

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join("..", "src")))

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load Config
config = load_config("../config.yaml")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

logging.info("Loading original datasets...")
SAMPLE_SIZE = 10000  # Or set a sample size if desired

# Load Dorothea network (we are now using only Dorothea)
dorothea_net = dc.get_collectri(organism="human")

# Load datasets into a dictionary.
original_datasets = {
    "Landmark Data": (
        load_sampled_data(
            config["data_paths"]["preprocessed_landmark_file"], sample_size=SAMPLE_SIZE
        ),
        "viability",
    ),
    "Best Inferred Data": (
        load_sampled_data(
            config["data_paths"]["preprocessed_best_inferred_file"],
            sample_size=SAMPLE_SIZE,
        ),
        "viability",
    ),
    "Gene Data": (
        load_sampled_data(
            config["data_paths"]["preprocessed_gene_file"],
            sample_size=SAMPLE_SIZE,
            use_chunks=True,
            chunk_size=2000,
        ),
        "viability",
    ),
}

# Define the list of min_n values to test.
min_n_values = [1]

# Set the algorithm manually (e.g. "ulm")
fixed_algorithm = "ulm"

# Dictionary to store all results.
all_results = {}


# %% Define a helper to convert a pandas DataFrame/Series pair into a PyTorch Dataset.
def create_dataset(X: pd.DataFrame, y: pd.Series) -> TensorDataset:
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


# %% Main processing loop (using only k-fold CV with Dorothea)
for min_n in min_n_values:
    logging.info(
        f"Processing TF activity inference with min_n = {min_n} and algorithm = {fixed_algorithm}"
    )

    # Apply TF activity inference for each dataset using the current min_n and the Dorothea network.
    datasets = {
        name: (
            run_tf_activity_inference(
                data, dorothea_net, min_n=min_n, algorithm=fixed_algorithm
            ),
            target,
        )
        for name, (data, target) in original_datasets.items()
    }

    # Log the shapes of the data after TF activity inference.
    for name, (X, y_col) in datasets.items():
        logging.info(f"min_n={min_n}, algo={fixed_algorithm} - {name} shape: {X.shape}")

    # %% Split each dataset into Train and Test sets using split_data_flexible (return_val=False).
    split_datasets = {}
    for name, (X, y_col) in datasets.items():
        X_train, y_train, X_test, y_test = split_data_flexible(
            df=X,
            config=config,
            target_name=y_col,
            stratify_by="cell_mfc_name",
            random_state=42,
            return_val=False,
        )
        split_datasets[name] = {
            "train": (X_train, y_train),
            "test": (X_test, y_test),
        }

    # %% Standardize the splits using StandardScaler.
    for name, splits in split_datasets.items():
        X_train, y_train = splits["train"]
        X_test, y_test = splits["test"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.values)
        X_train_scaled_df = pd.DataFrame(
            X_train_scaled, index=X_train.index, columns=X_train.columns
        )
        splits["train"] = (X_train_scaled_df, y_train)

        X_test_scaled = scaler.transform(X_test.values)
        X_test_scaled_df = pd.DataFrame(
            X_test_scaled, index=X_test.index, columns=X_test.columns
        )
        splits["test"] = (X_test_scaled_df, y_test)

    # %% Create a DataLoader for the Test set (used later for final evaluation).
    dataloaders = {}
    for name, splits in split_datasets.items():
        _, (X_test, y_test) = None, splits["test"]
        dataloaders[name] = {
            "test": create_dataloader(X_test, y_test, batch_size=1028, shuffle=False)
        }

    # %% K-Fold Cross-Validation on the Training set.
    kfold_results = {}
    for name, splits in split_datasets.items():
        logging.info(f"Performing 5-fold CV on training data for dataset: {name}")
        X_train, y_train = splits["train"]
        # Convert training data to a PyTorch Dataset.
        train_dataset = create_dataset(X_train, y_train)

        fold_metrics = k_fold_cross_validation(
            model_class=lambda: FlexibleFCNN(
                input_dim=X_train.shape[1],
                hidden_dims=[512, 256, 128, 64],
                output_dim=1,
                activation_fn="relu",
                dropout_prob=0.2,
                residual=False,
                norm_type="batchnorm",
                weight_init="kaiming",
            ),
            train_dataset=train_dataset,
            k_folds=5,
            batch_size=1028,
            epochs=20,  # adjust epochs as needed
            criterion_fn=nn.MSELoss(),
            optimizer_fn=lambda params: optim.AdamW(
                params, lr=0.001, weight_decay=1e-5
            ),
            scheduler_fn=lambda opt: ReduceLROnPlateau(
                opt, mode="min", patience=5, verbose=True
            ),
            device=device,
            use_mixed_precision=True,
        )
        # Average metrics over folds.
        avg_metrics = {
            key: sum(fold_metrics[f][key] for f in fold_metrics) / len(fold_metrics)
            for key in fold_metrics[0]
        }
        kfold_results[name] = {"fold_metrics": fold_metrics, "avg_metrics": avg_metrics}
        logging.info(f"Dataset: {name} - Average K-Fold CV Metrics: {avg_metrics}")

    # %% Final Training on Full Training Set and Evaluation on Test Set.
    final_results = {}
    for name, splits in split_datasets.items():
        logging.info(f"Final training on full training set for dataset: {name}")
        X_train, y_train = splits["train"]
        X_test, y_test = splits["test"]

        train_loader = create_dataloader(
            X_train, y_train, batch_size=1028, shuffle=True
        )
        test_loader = dataloaders[name]["test"]

        # Create a new model instance.
        fcnn_model = FlexibleFCNN(
            input_dim=X_train.shape[1],
            hidden_dims=[512, 256, 128, 64],
            output_dim=1,
            activation_fn="relu",
            dropout_prob=0.2,
            residual=False,
            norm_type="batchnorm",
            weight_init="kaiming",
        ).to(device)

        fcnn_criterion = nn.MSELoss()
        fcnn_optimizer = optim.AdamW(
            fcnn_model.parameters(), lr=0.001, weight_decay=1e-5
        )

        # Quick sanity check on the model.
        sanity_check(fcnn_model, train_loader, device)

        # Train the model.
        fcnn_train_losses, fcnn_val_losses = train_model(
            model=fcnn_model,
            train_loader=train_loader,
            val_loader=test_loader,  # Using the test set for final evaluation.
            criterion=fcnn_criterion,
            optimizer=fcnn_optimizer,
            scheduler=ReduceLROnPlateau(
                fcnn_optimizer, mode="min", patience=5, verbose=True
            ),
            epochs=20,
            device=device,
            gradient_clipping=5.0,
            early_stopping_patience=10,
            model_name=f"FlexibleFCNN_{name}_dorothea_min_n_{min_n}",
            use_mixed_precision=True,
        )

        # Evaluate the final model on the test set.
        fcnn_metrics = evaluate_model(
            fcnn_model,
            test_loader,
            fcnn_criterion,
            device=device,
            calculate_metrics=True,
        )
        final_results[name] = fcnn_metrics
        logging.info(f"Final Evaluation Metrics for dataset {name}: {fcnn_metrics}")

    # Store the results for the current min_n configuration.
    all_results[f"dorothea_min_n_{min_n}"] = {
        "CV": kfold_results,
        "Final": final_results,
    }

# %% Log and save the combined results.
logging.info(f"Training and evaluation completed. Combined Results: {all_results}")

# Optionally, flatten the nested dictionary into a DataFrame for easier analysis.
flat_results = {}
for key_outer, res in all_results.items():
    for dataset_key, metrics in res["Final"].items():
        composite_key = (dataset_key, key_outer)
        flat_results[composite_key] = metrics

results_df = pd.DataFrame.from_dict(flat_results, orient="index")
results_df.index = pd.MultiIndex.from_tuples(
    list(flat_results.keys()),
    names=["Dataset_and_Model", "Network_min_n"],
)

results_df.to_csv("results_all_min_n_and_networks.csv", index=True)
logging.info("All results saved to results_all_min_n_and_networks.csv.")

# %% Plotting: Create a line plot for performance (e.g. R²) versus min_n value for each feature set.
import matplotlib.pyplot as plt

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
    r2_vals = group["R²"].tolist()
    performance_data[dataset_name] = (min_n_vals, r2_vals)

plt.figure(figsize=(10, 6))
for dataset_name, (min_n_vals, r2_vals) in performance_data.items():
    plt.plot(min_n_vals, r2_vals, marker="o", label=dataset_name)

plt.xlabel("Target Gene Threshold Value")
plt.ylabel("R² Score")
plt.title("Performance vs. Target Threshold for Different Feature Sets (Dorothea)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("performance_vs_min_n_dorothea.png")
plt.show()

logging.info("Plot saved as performance_vs_min_n_dorothea.png")
