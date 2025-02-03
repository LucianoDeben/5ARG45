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

from evaluation import evaluate_model
from models import FlexibleFCNN
from preprocess import run_tf_activity_inference, split_data
from training import train_model
from utils import create_dataloader, load_config, load_sampled_data, sanity_check

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join("..", "src")))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load Config
config = load_config("../config.yaml")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# %% Load and Preprocess Datasets with TF Activity Inference
logging.info("Loading datasets and running TF activity inference...")


# Global variables
SAMPLE_SIZE = None

# Load Collectri network
collectri_net = dc.get_collectri(organism="human", split_complexes=False)

# Load datasets
datasets = {
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

# Apply TF activity inference
datasets = {
    name: (
        run_tf_activity_inference(
            data,
            collectri_net,
            min_n=config.get("min_n", 1),
        ),
        target,
    )
    for name, (data, target) in datasets.items()
}

# Log the shapes of the data after TF activity inference
for name, (X, y_col) in datasets.items():
    logging.info(f"{name} shape: {X.shape}")

# %% Split datasets into train/val/test sets
split_datasets = {}
for name, (X, y_col) in datasets.items():
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X,
        target_name=y_col,
        config=config,
        stratify_by="cell_mfc_name",
    )
    split_datasets[name] = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

# Standardize all the data splits using standard scaler
from sklearn.preprocessing import StandardScaler

for name, splits in split_datasets.items():
    # Fit the scaler on the training set
    X_train, y_train = splits["train"]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(
        X_train.values
    )  # Fit and transform the training set
    X_train_scaled_df = pd.DataFrame(
        X_train_scaled, index=X_train.index, columns=X_train.columns
    )
    split_datasets[name]["train"] = (X_train_scaled_df, y_train)

    # Transform the validation and test sets using the same scaler
    for split in ["val", "test"]:
        X, y = splits[split]
        X_scaled = scaler.transform(X.values)  # Transform using the fitted scaler
        X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        split_datasets[name][split] = (X_scaled_df, y)

# %% Create dataloaders
dataloaders = {
    name: {
        split: create_dataloader(X, y, batch_size=256, shuffle=True)
        for split, (X, y) in splits.items()
    }
    for name, splits in split_datasets.items()
}

# %% Train FlexibleFCNN sequentially
results = {}

for name, loaders in dataloaders.items():
    train_loader, val_loader, test_loader = (
        loaders["train"],
        loaders["val"],
        loaders["test"],
    )

    # Train FlexibleFCNN
    logging.info(f"Training FlexibleFCNN on {name} feature set...")
    fcnn_model = FlexibleFCNN(
        input_dim=next(iter(train_loader))[0].shape[1],
        hidden_dims=[512, 256, 128, 64],
        output_dim=1,
        activation_fn="relu",
        dropout_prob=0.2,
        residual=False,
        norm_type="batchnorm",
        weight_init="kaiming",
    ).to(device)

    fcnn_criterion = nn.MSELoss()
    fcnn_optimizer = optim.AdamW(fcnn_model.parameters(), lr=0.001, weight_decay=1e-5)

    sanity_check(fcnn_model, train_loader, device)

    fcnn_train_losses, fcnn_val_losses = train_model(
        model=fcnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=fcnn_criterion,
        optimizer=fcnn_optimizer,
        scheduler=ReduceLROnPlateau(
            fcnn_optimizer, mode="min", patience=5, verbose=True
        ),
        epochs=20,
        device=device,
        gradient_clipping=5.0,
        early_stopping_patience=10,
        model_name=f"FlexibleFCNN_{name}",
    )

    fcnn_metrics = evaluate_model(
        fcnn_model, test_loader, fcnn_criterion, device=device, calculate_metrics=True
    )
    results[f"{name}_FlexibleFCNN"] = fcnn_metrics

# Log and save results
logging.info(f"Training and evaluation completed. Results: {results}")
results_df = pd.DataFrame.from_dict(results, orient="index")

# Split the keys into MultiIndex
results_df.index = pd.MultiIndex.from_tuples(
    [
        (key.split("_")[0], "FlexibleFCNN") for key in results_df.index
    ],  # Explicitly add model name
    names=["Feature Set", "Model"],  # Names for the MultiIndex levels
)

# Save results to CSV
results_df.to_csv("results.csv", index=True)
logging.info("Results saved to results.csv.")
