"""
Infrastructure tester for the multimodal drug response prediction framework.
This script demonstrates how to use the infrastructure layer components to:
1. Load and preprocess data
2. Train a Ridge regression model and an FCNN model
3. Log metrics and save checkpoints
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from data.datasets import DatasetFactory, MultimodalDrugDataset, TranscriptomicsDataset

# Import infrastructure components
from data.loaders import GCTXDataLoader
from data.preprocessing import LINCSCTRPDataProcessor
from data.transformers import create_transformations
from utils.logging import ExperimentLogger
from utils.storage import CacheManager, CheckpointManager, DatasetStorage

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class FCNN(nn.Module):
    """Simple fully-connected neural network for drug response prediction."""

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build layers one by one instead of using Sequential
        # First layer
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        # Create the rest of the layers
        self.layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            self.layers.append(nn.BatchNorm1d(hidden_dims[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))

        # Output layer
        self.output = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        # Debug the input shape
        batch_size = x.size(0)
        if batch_size == 0:
            raise ValueError(f"Empty batch received, shape: {x.shape}")

        # Print warning if batch size is 1 (BatchNorm will struggle)
        if batch_size == 1:
            logger.warning(
                f"Batch size is 1, which can cause BatchNorm issues. Shape: {x.shape}"
            )

        # First layer
        x = self.fc1(x)

        # Skip BatchNorm if batch size is 1
        if batch_size > 1:
            x = self.bn1(x)

        x = self.act1(x)
        x = self.drop1(x)

        # Process through the rest of the layers
        for i in range(0, len(self.layers), 4):
            x = self.layers[i](x)  # Linear

            # Skip BatchNorm if batch size is 1
            if batch_size > 1:
                x = self.layers[i + 1](x)  # BatchNorm

            x = self.layers[i + 2](x)  # ReLU
            x = self.layers[i + 3](x)  # Dropout

        # Output layer
        x = self.output(x)
        return x.squeeze(-1)


def train_ridge_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    logger: ExperimentLogger,
) -> Ridge:
    """Train a Ridge regression model and log results."""
    logger.log_metric("data_size/train", len(X_train), 0, "info")
    logger.log_metric("data_size/val", len(X_val), 0, "info")

    # Train model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Evaluate on training set
    y_pred_train = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)

    # Evaluate on validation set
    y_pred_val = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_pred_val)
    val_r2 = r2_score(y_val, y_pred_val)

    # Log metrics
    logger.log_metrics({"mse": train_mse, "r2": train_r2}, 0, "train")

    logger.log_metrics({"mse": val_mse, "r2": val_r2}, 0, "val")

    logger.logger.info(
        f"Ridge Regression Results - Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}"
    )
    logger.logger.info(
        f"Ridge Regression Results - Val MSE: {val_mse:.4f}, R²: {val_r2:.4f}"
    )

    return model


def train_fcnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    hidden_dims: list,
    logger: ExperimentLogger,
    checkpoint_mgr: CheckpointManager,
    device: torch.device,
    epochs: int = 50,
    lr: float = 0.001,
) -> nn.Module:
    """Train a fully-connected neural network and log results."""
    # Initialize model
    model = FCNN(input_dim, hidden_dims).to(device)
    logger.logger.info(
        f"Created FCNN model with input dim {input_dim} and hidden dims {hidden_dims}"
    )

    # Log model architecture
    logger.logger.info(f"Model architecture:\n{model}")

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Train model
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            # Extract inputs and targets
            if isinstance(batch, dict):  # Multimodal dataset
                inputs = batch["transcriptomics"].to(device)
                targets = batch["viability"].to(device)
            else:  # Tuple from regular dataset
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

            # Debug info
            if batch_count == 0:
                logger.logger.info(f"Batch shape: {inputs.shape}")

            # Skip batches with only one sample (BatchNorm problems)
            if inputs.size(0) <= 1:
                logger.logger.warning(f"Skipping batch with size {inputs.size(0)}")
                continue

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

        if batch_count > 0:
            train_loss /= batch_count

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        val_batch_count = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):  # Multimodal dataset
                    inputs = batch["transcriptomics"].to(device)
                    targets = batch["viability"].to(device)
                else:  # Tuple from regular dataset
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                # Skip batches with only one sample (BatchNorm problems)
                if inputs.size(0) <= 1:
                    continue

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                val_preds.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
                val_batch_count += 1

        if val_batch_count > 0:
            val_loss /= val_batch_count

        if val_preds:
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            val_r2 = r2_score(val_targets, val_preds)
        else:
            val_r2 = 0.0
            logger.logger.warning("No validation predictions - check batch sizes")

        # Log metrics
        logger.log_metrics(
            {
                "loss": train_loss,
            },
            epoch,
            "train",
        )

        logger.log_metrics({"loss": val_loss, "r2": val_r2}, epoch, "val")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        metrics = {"train_loss": train_loss, "val_loss": val_loss, "val_r2": val_r2}

        checkpoint_mgr.save(model, epoch, metrics, optimizer)

        logger.logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val R²: {val_r2:.4f}"
        )

    # Load best model
    best_path = checkpoint_mgr.get_best_model_path()
    if best_path:
        checkpoint = CheckpointManager.load(best_path, model)
        best_epoch = checkpoint["epoch"]
        logger.logger.info(f"Loaded best model from epoch {best_epoch+1}")

    return model


def create_dl_dataloaders(
    X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32
):
    """Create PyTorch DataLoaders from numpy arrays for deep learning."""
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate_models(
    ridge_model: Ridge,
    fcnn_model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_loader: DataLoader,
    logger: ExperimentLogger,
    device: torch.device,
):
    """Evaluate both models on the test set."""
    # Evaluate Ridge model
    y_pred_ridge = ridge_model.predict(X_test)
    ridge_mse = mean_squared_error(y_test, y_pred_ridge)
    ridge_r2 = r2_score(y_test, y_pred_ridge)
    ridge_pearson, ridge_p_value = stats.pearsonr(y_test, y_pred_ridge)

    logger.log_metrics(
        {
            "mse": ridge_mse,
            "r2": ridge_r2,
            "pearson_correlation": ridge_pearson,
            "pearson_pvalue": ridge_p_value,
        },
        0,
        "test_ridge",
    )

    logger.logger.info(
        f"Ridge Test Results - MSE: {ridge_mse:.4f}, R²: {ridge_r2:.4f}, "
        f"Pearson r: {ridge_pearson:.4f}, p-value: {ridge_p_value:.4f}"
    )

    # Evaluate FCNN model
    fcnn_model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch in test_loader:
            # Skip batches with only one sample
            if isinstance(batch, dict):
                inputs = batch["transcriptomics"].to(device)
                targets = batch["viability"].to(device)
            else:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

            if inputs.size(0) <= 1:
                continue

            outputs = fcnn_model(inputs)
            test_preds.append(outputs.cpu().numpy())
            test_targets.append(targets.cpu().numpy())

    # Check if we have predictions
    if len(test_preds) > 0:
        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)

        # Compute metrics
        fcnn_mse = mean_squared_error(test_targets, test_preds)
        fcnn_r2 = r2_score(test_targets, test_preds)
        fcnn_pearson, fcnn_p_value = stats.pearsonr(test_targets, test_preds)
    else:
        fcnn_mse = 0.0
        fcnn_r2 = 0.0
        fcnn_pearson = 0.0
        fcnn_p_value = 1.0
        logger.logger.warning("No test predictions - check batch sizes")

    # Log FCNN metrics
    logger.log_metrics(
        {
            "mse": fcnn_mse,
            "r2": fcnn_r2,
            "pearson_correlation": fcnn_pearson,
            "pearson_pvalue": fcnn_p_value,
        },
        0,
        "test_fcnn",
    )

    logger.logger.info(
        f"FCNN Test Results - MSE: {fcnn_mse:.4f}, R²: {fcnn_r2:.4f}, "
        f"Pearson r: {fcnn_pearson:.4f}, p-value: {fcnn_p_value:.4f}"
    )

    # Rest of the visualization code remains the same
    # Create comparison plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Ridge model predictions
    ax[0].scatter(y_test, y_pred_ridge, alpha=0.5)
    ax[0].plot([0, 1], [0, 1], "r--")
    ax[0].set_title(f"Ridge Regression (R² = {ridge_r2:.4f}, r = {ridge_pearson:.4f})")
    ax[0].set_xlabel("True Viability")
    ax[0].set_ylabel("Predicted Viability")

    # FCNN model predictions
    if len(test_preds) > 0:
        ax[1].scatter(test_targets, test_preds, alpha=0.5)
        ax[1].plot([0, 1], [0, 1], "r--")
        ax[1].set_title(f"FCNN (R² = {fcnn_r2:.4f}, r = {fcnn_pearson:.4f})")
        ax[1].set_xlabel("True Viability")
        ax[1].set_ylabel("Predicted Viability")
    else:
        ax[1].text(0.5, 0.5, "No predictions available", ha="center", va="center")
        ax[1].set_title("FCNN (No predictions)")

    plt.tight_layout()

    # Log figure
    logger.log_figure("model_comparison", fig, 0, "test")


def main():
    """Main function to test the infrastructure components."""
    parser = argparse.ArgumentParser(description="Test the infrastructure layer")
    parser.add_argument(
        "--gctx",
        type=str,
        default="data/processed/LINCS.gctx",
        help="Path to GCTX file",
    )
    parser.add_argument("--nrows", type=int, default=1000, help="Number of rows to use")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    args = parser.parse_args()

    # Create directory structure if it doesn't exist
    for dir_path in ["data/processed", "logs", "models/saved", "results"]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Hard-coded paths instead of loading from config
    data_dir = Path("data")
    model_dir = Path("models")
    log_dir = Path("logs")

    # Set up experiment logger
    experiment_logger = ExperimentLogger(
        experiment_name="infrastructure_test",
        log_dir=str(log_dir),
        use_tensorboard=True,
        use_wandb=False,
    )

    # Create a simple config dict for logging
    config = {
        "data": {
            "gctx_file": args.gctx,
            "feature_space": "landmark",
            "nrows": args.nrows,
        },
        "model": {"fcnn_hidden_dims": [256, 128, 64], "dropout": 0.3},
        "training": {
            "batch_size": args.batch_size,
            "epochs": 20,
            "learning_rate": 0.001,
            "test_size": 0.4,
            "val_size": 0.1,
        },
    }

    # Log config
    experiment_logger.log_config(config)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_logger.logger.info(f"Using device: {device}")

    # Set up data transformations
    transcriptomics_transform, molecular_transform = create_transformations(
        transcriptomics_transform_type="scale",
        molecular_transform_type="fingerprint",
        fingerprint_size=1024,
        fingerprint_radius=2,
    )

    # Create data processor
    data_processor = LINCSCTRPDataProcessor(
        gctx_file=args.gctx,
        feature_space="landmark",
        nrows=args.nrows,
        test_size=0.4,  # We'll split this into val+test later
        val_size=0.1 / 0.6,  # Adjusted to get 10% of total as validation
        random_state=42,
        batch_size=args.batch_size,
        transform_transcriptomics=transcriptomics_transform,
        transform_molecular=molecular_transform,
    )

    # Get transcriptomics datasets for Ridge regression
    train_ds_t, val_ds_t, test_ds_t = data_processor.get_transcriptomics_data()

    # Get training data for Ridge model
    X_train, y_train = train_ds_t.get_data()
    X_val, y_val = val_ds_t.get_data()
    X_test, y_test = test_ds_t.get_data()

    experiment_logger.logger.info(f"Training set: {X_train.shape}")
    experiment_logger.logger.info(f"Validation set: {X_val.shape}")
    experiment_logger.logger.info(f"Test set: {X_test.shape}")

    # Create custom dataloaders for FCNN to avoid BatchNorm issues
    train_loader, val_loader, test_loader = create_dl_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=args.batch_size
    )

    experiment_logger.logger.info(
        f"Created DataLoaders with batch size {args.batch_size}"
    )

    # Create checkpoint manager
    checkpoint_mgr = CheckpointManager(
        model_dir / "fcnn",
        filename="fcnn_epoch{epoch:03d}_loss{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    # Create cache manager
    cache_mgr = CacheManager(data_dir / "cache", max_memory_size=1.0, max_disk_size=5.0)

    # Create dataset storage
    dataset_storage = DatasetStorage(data_dir, compress=True)

    # Train Ridge regression model
    experiment_logger.logger.info("Training Ridge regression model...")
    ridge_model = train_ridge_regression(
        X_train, y_train, X_val, y_val, experiment_logger
    )

    # Save the Ridge model
    import joblib

    ridge_model_path = model_dir / "ridge_model.joblib"
    ridge_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge_model, ridge_model_path)
    experiment_logger.logger.info(f"Saved Ridge model to {ridge_model_path}")

    # Train FCNN model
    experiment_logger.logger.info("Training FCNN model...")
    fcnn_model = train_fcnn(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=X_train.shape[1],
        hidden_dims=[256, 128, 64],
        logger=experiment_logger,
        checkpoint_mgr=checkpoint_mgr,
        device=device,
        epochs=20,
        lr=0.001,
    )

    # Evaluate both models
    experiment_logger.logger.info("Evaluating models on test set...")
    evaluate_models(
        ridge_model=ridge_model,
        fcnn_model=fcnn_model,
        X_test=X_test,
        y_test=y_test,
        test_loader=test_loader,
        logger=experiment_logger,
        device=device,
    )

    # Generate metric plots
    experiment_logger.plot_metrics()

    # Clean up
    experiment_logger.close()
    cache_mgr.close()

    experiment_logger.logger.info("Finished infrastructure testing")


if __name__ == "__main__":
    main()
