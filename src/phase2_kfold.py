# %% Import required libraries and modules
import logging
import os
import sys
import warnings
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Suppress all FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)

import decoupler as dc
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

from evaluation import evaluate_model
from models import FlexibleFCNN
# We still use our unified splitting and k_fold functions.
from preprocess import split_data_flexible, k_fold_cross_validation  
from training import train_model  
from utils import create_dataloader, create_dataset, load_config, load_sampled_data, sanity_check

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
SAMPLE_SIZE = None  # Or set a sample size if desired

# Note: Since we want to train on the genes directly, we do not use any TF network.
# Instead, we load the raw preprocessed datasets.
original_datasets = {
    "Landmark Data": (
        load_sampled_data(config["data_paths"]["preprocessed_landmark_file"], sample_size=SAMPLE_SIZE),
        "viability",
    ),
    "Best Inferred Data": (
        load_sampled_data(config["data_paths"]["preprocessed_best_inferred_file"], sample_size=SAMPLE_SIZE),
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

# Dictionary to store all results.
all_results = {}

# %% Define a helper to convert a pandas DataFrame/Series pair into a PyTorch Dataset.
def create_dataset(X: pd.DataFrame, y: pd.Series) -> TensorDataset:
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)

# %% Main processing loop (using only k-fold CV on the gene data)
for dataset_name, (data, target) in original_datasets.items():
    logging.info(f"Processing feature set: {dataset_name}")
    
    # Here, we do not perform any TF activity inference; we use the raw gene data.
    # Optionally, you could perform some other gene-level preprocessing here.
    # For now, we assume 'data' is already the desired DataFrame.
    X = data
    y_col = target

    # Log the shape of the dataset.
    logging.info(f"{dataset_name} shape: {X.shape}")

    # %% Split the dataset into Train and Test sets using split_data_flexible (return_val=False).
    X_train, y_train, X_test, y_test = split_data_flexible(
        df=X,
        config=config,
        target_name=y_col,
        stratify_by="cell_mfc_name",
        random_state=42,
        return_val=False
    )
    split_data = {"train": (X_train, y_train), "test": (X_test, y_test)}

    # %% Standardize the splits using StandardScaler.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled = scaler.transform(X_test.values)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    split_data["train"] = (X_train_scaled_df, y_train)
    split_data["test"] = (X_test_scaled_df, y_test)

    # %% Create a DataLoader for the Test set (used later for final evaluation).
    test_loader = create_dataloader(X_test_scaled_df, y_test, batch_size=1028, shuffle=False)
    
    # %% K-Fold Cross-Validation on the Training set.
    train_dataset = create_dataset(X_train_scaled_df, y_train)
    fold_metrics = k_fold_cross_validation(
        model_class=lambda: FlexibleFCNN(
            input_dim=X_train_scaled_df.shape[1],
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
        optimizer_fn=lambda params: optim.AdamW(params, lr=0.001, weight_decay=1e-5),
        scheduler_fn=lambda opt: ReduceLROnPlateau(opt, mode="min", patience=5, verbose=True),
        device=device,
        use_mixed_precision=True,
    )
    # Average metrics over folds.
    avg_metrics = {key: sum(fold_metrics[f][key] for f in fold_metrics) / len(fold_metrics)
                   for key in fold_metrics[0]}
    cv_results = {"fold_metrics": fold_metrics, "avg_metrics": avg_metrics}
    logging.info(f"{dataset_name} - Average K-Fold CV Metrics: {avg_metrics}")

    # %% Final Training on Full Training Set and Evaluation on Test Set.
    train_loader = create_dataloader(X_train_scaled_df, y_train, batch_size=1028, shuffle=True)
    # Create a new model instance.
    model_instance = FlexibleFCNN(
        input_dim=X_train_scaled_df.shape[1],
        hidden_dims=[512, 256, 128, 64],
        output_dim=1,
        activation_fn="relu",
        dropout_prob=0.2,
        residual=False,
        norm_type="batchnorm",
        weight_init="kaiming",
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model_instance.parameters(), lr=0.001, weight_decay=1e-5)

    # Quick sanity check on the model.
    sanity_check(model_instance, train_loader, device)

    # Train the model on the full training set.
    train_losses, val_losses = train_model(
        model=model_instance,
        train_loader=train_loader,
        val_loader=test_loader,  # Using the test set for final evaluation.
        criterion=criterion,
        optimizer=optimizer,
        scheduler=ReduceLROnPlateau(optimizer, mode="min", patience=5, verbose=True),
        epochs=20,
        device=device,
        gradient_clipping=5.0,
        early_stopping_patience=10,
        model_name=f"FlexibleFCNN_{dataset_name}",
        use_mixed_precision=True,
    )

    # Evaluate the final model on the test set.
    final_metrics = evaluate_model(model_instance, test_loader, criterion, device=device, calculate_metrics=True)
    logging.info(f"Final Evaluation Metrics for {dataset_name}: {final_metrics}")

    # Store the results for this feature set.
    all_results[dataset_name] = {
        "CV": cv_results,
        "Final": final_metrics,
    }

# %% Log and save the combined results.
logging.info(f"Training and evaluation completed. Combined Results: {all_results}")

# Flatten the nested dictionary into a DataFrame for easier analysis.
flat_results = {}
for dataset_key, metrics in all_results.items():
    flat_results[(dataset_key, "final")] = metrics["Final"]

results_df = pd.DataFrame.from_dict(flat_results, orient="index")
results_df.index = pd.MultiIndex.from_tuples(
    list(flat_results.keys()),
    names=["Dataset_and_Model", "Result_Type"],
)

results_df.to_csv("results_feature_sets.csv", index=True)
logging.info("All results saved to results_feature_sets.csv.")

# %% Plotting: Create a bar plot for performance (e.g. R²) for each feature set.
# Read the CSV file that was written out.
results_df = pd.read_csv("results_feature_sets.csv", index_col=[0, 1])
results_df = results_df.reset_index()

# We assume that the column "R²" contains the R² scores.
performance_data = {}
for dataset_name, group in results_df.groupby("Dataset_and_Model"):
    # For this analysis, we take the final R² score for each feature set.
    # (If multiple rows exist, you might take the mean; here we assume one value per dataset.)
    r2_score = group["Pearson Correlation"].mean()  # Use mean() in case there are multiple rows.
    performance_data[dataset_name] = r2_score

# Create a bar plot.
plt.figure(figsize=(8, 6))
plt.bar(performance_data.keys(), performance_data.values(), color="skyblue")
plt.xlabel("Feature Set")
plt.ylabel("Pearson Correlation")
plt.title("Final Performance for Different Feature Sets (Genes Directly)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("performance_feature_sets.png")
plt.show()

logging.info("Plot saved as performance_feature_sets.png")
