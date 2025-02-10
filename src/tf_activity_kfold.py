# %% Import required libraries and modules
import logging
import os
import sys
import warnings

# Suppress all FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)

import decoupler as dc
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Import your evaluation, preprocessing, and utility functions
from evaluation import evaluate_regression_metrics  # our helper that returns (mse, mae, r2, pearson_coef)
from models import FlexibleFCNN  # your original model (if needed)
from preprocess import run_tf_activity_inference, split_data_train_test  # new two-way split function
from utils import create_dataloader, load_config, load_sampled_data, sanity_check

# Import PyTorch Lightning and your Lightning module
import pytorch_lightning as pl
from models_lightning import LitFCNN  # Your LightningModule wrapping FlexibleFCNN

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join("..", "src")))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load Config
config = load_config("../config.yaml")

# Set device (Lightning will pick this automatically via accelerator settings)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# %% Load original datasets (before TF activity inference)
logging.info("Loading original datasets...")
SAMPLE_SIZE = 1000

# Load Collectri network
collectri_net = dc.get_collectri(organism="human", split_complexes=False)

# Load datasets into a dictionary.
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

# Define the list of min_n values to test and algorithms to benchmark.
min_n_values = [1]
algorithms = ["ulm"]

# Dictionary to store results
all_results = {}

# Define a simple helper to convert pandas DataFrame/Series to a PyTorch Dataset.
from torch.utils.data import TensorDataset

def create_dataset(X: pd.DataFrame, y: pd.Series) -> TensorDataset:
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)

# %% Main processing loop
for min_n in min_n_values:
    for algorithm in algorithms:
        logging.info(f"Processing TF activity inference with min_n = {min_n} and algorithm = {algorithm}")

        # Apply TF activity inference for each dataset.
        datasets = {
            name: (
                run_tf_activity_inference(data, collectri_net, min_n=min_n, algorithm=algorithm),
                target,
            )
            for name, (data, target) in original_datasets.items()
        }

        # Log data shapes after inference.
        for name, (X, y_col) in datasets.items():
            logging.info(f"min_n={min_n}, algo={algorithm} - {name} shape: {X.shape}")

        # %% Split each dataset into Train and Test sets (no fixed validation set)
        split_datasets = {}
        for name, (X, y_col) in datasets.items():
            # Use the new train-test split function.
            X_train, y_train, X_test, y_test = split_data_train_test(
                X,
                config=config,
                target_name=y_col,
                stratify_by="cell_mfc_name",
                random_state=42,
            )
            split_datasets[name] = {
                "train": (X_train, y_train),
                "test": (X_test, y_test),
            }

        # %% Standardize the Train and Test sets using StandardScaler
        for name, splits in split_datasets.items():
            X_train, y_train = splits["train"]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.values)
            X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
            split_datasets[name]["train"] = (X_train_scaled_df, y_train)

            X_test, y_test = splits["test"]
            X_test_scaled = scaler.transform(X_test.values)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
            split_datasets[name]["test"] = (X_test_scaled_df, y_test)

        # %% K-Fold Cross-Validation on the Training Set using Lightning
        cv_results = {}
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        for name, splits in split_datasets.items():
            logging.info(f"Performing 5-fold CV on training data for dataset: {name}")
            X_train, y_train = splits["train"]
            # Convert training DataFrame to a PyTorch Dataset.
            full_train_dataset = create_dataset(X_train, y_train)

            fold_metrics_list = []
            for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_train_dataset)))):
                logging.info(f"  Fold {fold+1}")
                from torch.utils.data import Subset
                train_subset = Subset(full_train_dataset, train_idx)
                val_subset = Subset(full_train_dataset, val_idx)
                train_loader = torch.utils.data.DataLoader(train_subset, batch_size=1028, shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_subset, batch_size=1028, shuffle=False)

                # Create a new Lightning model instance for this fold.
                lit_model = LitFCNN(input_dim=X_train.shape[1], lr=1e-3)
                # (Optional) Add callbacks here (e.g., EarlyStopping, ModelCheckpoint)
                trainer = pl.Trainer(
                    max_epochs=10,
                    accelerator="gpu" if device.type == "cuda" else "cpu",
                    devices=1 if device.type == "cuda" else None,
                    log_every_n_steps=10,
                    enable_checkpointing=False,
                    logger=False,
                )
                trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                # Validate on the fold's validation set.
                val_result = trainer.validate(lit_model, dataloaders=val_loader)
                # val_result is a list of dicts; we record the first dictionary.
                fold_metrics_list.append(val_result[0])
            # Average the metrics over folds.
            avg_fold_metrics = {}
            for key in fold_metrics_list[0]:
                avg_fold_metrics[key] = sum(d[key] for d in fold_metrics_list) / len(fold_metrics_list)
            cv_results[name] = {"fold_metrics": fold_metrics_list, "avg_metrics": avg_fold_metrics}
            logging.info(f"Dataset: {name} - Average CV Metrics: {avg_fold_metrics}")

        # %% Final Training on Full Training Set and Evaluation on Test Set with Extra Metrics
        final_results = {}
        for name, splits in split_datasets.items():
            logging.info(f"Final training on full training set for dataset: {name}")
            X_train, y_train = splits["train"]
            X_test, y_test = splits["test"]

            # Create full training and test DataLoaders.
            train_loader = create_dataloader(X_train, y_train, batch_size=1028, shuffle=True)
            test_loader = create_dataloader(X_test, y_test, batch_size=1028, shuffle=False)

            # Instantiate a new Lightning model.
            final_model = LitFCNN(input_dim=X_train.shape[1], lr=1e-3)
            trainer = pl.Trainer(
                max_epochs=20,
                accelerator="gpu" if device.type == "cuda" else "cpu",
                devices=1 if device.type == "cuda" else None,
                log_every_n_steps=10,
            )
            trainer.fit(final_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
            # Run testing via Lightning (this returns a list of metric dictionaries).
            test_results = trainer.test(final_model, dataloaders=test_loader)
            # Now, for additional metrics (Pearson and R2), we obtain predictions and compute them.
            all_preds = []
            all_targets = []
            final_model.freeze()  # Ensure the model is in evaluation mode.
            for batch in test_loader:
                X_batch, y_batch = batch
                preds = final_model(X_batch.to(device))
                all_preds.append(preds.detach().cpu())
                all_targets.append(y_batch)
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            mse, mae, r2, pearson_coef = evaluate_regression_metrics(all_targets, all_preds)
            # Combine Lightning test metrics with our custom metrics.
            final_results[name] = {
                "lightning_test": test_results[0],
                "MSE": mse,
                "MAE": mae,
                "R2": r2,
                "Pearson": pearson_coef,
            }
            logging.info(f"Final Evaluation Metrics for dataset {name}: {final_results[name]}")

        # Store all results for the current configuration.
        all_results[f"min_n_{min_n}_algo_{algorithm}"] = {
            "CV": cv_results,
            "Final": final_results,
        }

# %% Log and save the combined results
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
    names=["Dataset_and_Model", "min_n_and_algo"],
)
results_df.to_csv("results_all_min_n_and_algos.csv", index=True)
logging.info("All results saved to results_all_min_n_and_algos.csv.")
