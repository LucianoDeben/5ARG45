"""
Infrastructure tester for the multimodal drug response prediction framework.
This script demonstrates how to use the infrastructure layer components to:
1. Load and preprocess data using config-driven settings
2. Train a Ridge regression model and an FCNN model
3. Log metrics, save checkpoints, and cache data
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

from .data.loaders import GCTXDataLoader

# Import infrastructure components
from .config.config_utils import load_config, setup_logging
from .data.augmentation import create_augmentations
from .data.datasets import DatasetFactory, TranscriptomicsDataset
from .data.feature_transforms import create_feature_transform
from .data.preprocessing import LINCSCTRPDataProcessor
from .data.preprocessing_transforms import create_preprocessing_transform
from .utils.experiment_tracker import ExperimentTracker
from .utils.storage import CacheManager, CheckpointManager, DatasetStorage

# Setup logging using config module
setup_logging()
logger = logging.getLogger(__name__)


class FCNN(nn.Module):
    """Simple fully-connected neural network for drug response prediction."""

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        if batch_size == 0:
            raise ValueError(f"Empty batch received, shape: {x.shape}")
        if batch_size == 1:
            logger.warning(
                f"Batch size is 1, BatchNorm may be unstable. Shape: {x.shape}"
            )
        return self.network(x).squeeze(-1)


def train_ridge_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    logger: ExperimentTracker,
) -> Ridge:
    """Train a Ridge regression model and log results."""
    logger.log_metric("data_size/train", len(X_train), 0, "info")
    logger.log_metric("data_size/val", len(X_val), 0, "info")

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)

    y_pred_val = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_pred_val)
    val_r2 = r2_score(y_val, y_pred_val)

    logger.log_metrics({"mse": train_mse, "r2": train_r2}, 0, "train")
    logger.log_metrics({"mse": val_mse, "r2": val_r2}, 0, "val")

    logger.logger.info(f"Ridge - Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
    logger.logger.info(f"Ridge - Val MSE: {val_mse:.4f}, R²: {val_r2:.4f}")

    return model


def train_fcnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    hidden_dims: list,
    logger: ExperimentTracker,
    checkpoint_mgr: CheckpointManager,
    device: torch.device,
    epochs: int = 20,
    lr: float = 0.001,
) -> nn.Module:
    """Train a fully-connected neural network and log results."""
    model = FCNN(input_dim, hidden_dims).to(device)
    logger.logger.info(
        f"Created FCNN with input dim {input_dim}, hidden dims {hidden_dims}"
    )
    logger.log_model_summary(model, {"transcriptomics": (2, input_dim)})

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            inputs = batch["transcriptomics"].to(device)
            targets = batch["viability"].to(device)

            if inputs.size(0) <= 1:
                continue

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

        train_loss = train_loss / batch_count if batch_count > 0 else 0.0

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        val_batch_count = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["transcriptomics"].to(device)
                targets = batch["viability"].to(device)

                if inputs.size(0) <= 1:
                    continue

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                val_preds.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
                val_batch_count += 1

        val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0.0
        val_r2 = (
            r2_score(np.concatenate(val_targets), np.concatenate(val_preds))
            if val_preds
            else 0.0
        )

        logger.log_metrics({"loss": train_loss}, epoch, "train")
        logger.log_metrics({"loss": val_loss, "r2": val_r2}, epoch, "val")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        metrics = {"train_loss": train_loss, "val_loss": val_loss, "val_r2": val_r2}
        checkpoint_mgr.save(model, epoch, metrics, optimizer)

        logger.logger.info(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}"
        )

    best_path = checkpoint_mgr.get_best_model_path()
    if best_path:
        CheckpointManager.load(best_path, model)
        logger.logger.info(f"Loaded best model from {best_path}")

    return model


def evaluate_models(
    ridge_model: Ridge,
    fcnn_model: nn.Module,
    test_ds: TranscriptomicsDataset,
    logger: ExperimentTracker,
    device: torch.device,
    batch_size: int = 32,
):
    """Evaluate both models on the test set."""
    X_test, y_test = test_ds.get_data()
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Ridge evaluation
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
        f"Ridge Test - MSE: {ridge_mse:.4f}, R²: {ridge_r2:.4f}, Pearson r: {ridge_pearson:.4f}, p: {ridge_p_value:.4f}"
    )

    # FCNN evaluation
    fcnn_model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)  # TranscriptomicsDataset returns tuple
            targets = batch[1].to(device)
            if inputs.size(0) <= 1:
                continue
            outputs = fcnn_model(inputs)
            test_preds.append(outputs.cpu().numpy())
            test_targets.append(targets.cpu().numpy())

    test_preds = np.concatenate(test_preds) if test_preds else np.array([])
    test_targets = np.concatenate(test_targets) if test_targets else np.array([])

    fcnn_mse = mean_squared_error(test_targets, test_preds) if test_preds.size else 0.0
    fcnn_r2 = r2_score(test_targets, test_preds) if test_preds.size else 0.0
    fcnn_pearson, fcnn_p_value = (
        stats.pearsonr(test_targets, test_preds) if test_preds.size else (0.0, 1.0)
    )

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
        f"FCNN Test - MSE: {fcnn_mse:.4f}, R²: {fcnn_r2:.4f}, Pearson r: {fcnn_pearson:.4f}, p: {fcnn_p_value:.4f}"
    )

    # Visualization
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(y_test, y_pred_ridge, alpha=0.5)
    ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax[0].set_title(f"Ridge (R² = {ridge_r2:.4f}, r = {ridge_pearson:.4f})")
    ax[0].set_xlabel("True Viability")
    ax[0].set_ylabel("Predicted Viability")

    if test_preds.size:
        ax[1].scatter(test_targets, test_preds, alpha=0.5)
        ax[1].plot(
            [test_targets.min(), test_targets.max()],
            [test_targets.min(), test_targets.max()],
            "r--",
        )
        ax[1].set_title(f"FCNN (R² = {fcnn_r2:.4f}, r = {fcnn_pearson:.4f})")
    else:
        ax[1].text(0.5, 0.5, "No predictions", ha="center", va="center")
        ax[1].set_title("FCNN (No predictions)")
    ax[1].set_xlabel("True Viability")
    ax[1].set_ylabel("Predicted Viability")

    plt.tight_layout()
    logger.log_figure("model_comparison", fig, 0, "test")


def main():
    parser = argparse.ArgumentParser(description="Test the infrastructure layer")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--gctx",
        type=str,
        default="data/processed/LINCS.gctx",
        help="Path to GCTX file",
    )
    parser.add_argument("--nrows", type=int, default=1000, help="Number of rows to use")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    data_dir = Path(config["paths"]["data_dir"])
    model_dir = Path(config["paths"]["model_dir"])
    log_dir = Path(config["paths"]["log_dir"])

    for dir_path in [data_dir / "processed", log_dir, model_dir / "saved"]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Setup experiment logger
    experiment_logger = ExperimentTracker(
        experiment_name="infrastructure_test",
        config=config,
        log_dir=str(log_dir),
        use_tensorboard=True,
        use_wandb=config["experiment"]["track"],
        wandb_project=config["experiment"]["project_name"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup transformations
    transcriptomics_transform = create_preprocessing_transform(
        config["data"].get("normalize", "scale")
    )
    molecular_transform = create_feature_transform(
        config["molecular"]["representation"],
        fingerprint_size=config["molecular"]["fingerprint_size"],
        fingerprint_radius=config["molecular"]["radius"],
    )
    aug_transform, _ = create_augmentations(
        transcriptomics_augment_type="noise", noise_args={"std": 0.05}
    )

    # Preprocess data
    preprocessor = LINCSCTRPDataProcessor(
        gctx_file=args.gctx,
        feature_space=config["data"]["feature_space"],
        nrows=args.nrows or config["data"]["nrows"],
        transform_transcriptomics=transcriptomics_transform,
    )
    transcriptomics, metadata = preprocessor.preprocess()

    # Create and split datasets with chunking
    with GCTXDataLoader(args.gctx) as loader:
        train_ds, val_ds, test_ds = DatasetFactory.create_and_split_transcriptomics(
            gctx_loader=loader,
            feature_space=config["data"]["feature_space"],
            nrows=args.nrows or config["data"]["nrows"],
            test_size=config["training"]["test_size"],
            val_size=config["training"]["val_size"],
            random_state=config["training"]["random_state"],
            transform=aug_transform,
            chunk_size=10000,
        )

    X_train, y_train = train_ds.get_data()
    X_val, y_val = val_ds.get_data()
    logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

    # Initialize storage components
    cache_mgr = CacheManager(data_dir / "cache", max_memory_size=1.0, max_disk_size=5.0)
    dataset_storage = DatasetStorage(data_dir, compress=True)
    checkpoint_mgr = CheckpointManager(
        model_dir / "fcnn",
        filename="fcnn_epoch{epoch:03d}_loss{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    # Cache training data
    cache_mgr.put("train_data", {"X": X_train, "y": y_train}, to_disk=True)
    if cache_mgr.get("train_data"):
        logger.info("Successfully cached and retrieved training data")

    # Train Ridge model
    logger.info("Training Ridge regression model...")
    ridge_model = train_ridge_regression(
        X_train, y_train, X_val, y_val, experiment_logger
    )
    import joblib

    ridge_path = model_dir / "ridge_model.joblib"
    joblib.dump(ridge_model, ridge_path)
    logger.info(f"Saved Ridge model to {ridge_path}")

    # Train FCNN model
    logger.info("Training FCNN model...")
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["training"]["batch_size"], shuffle=False
    )
    fcnn_model = train_fcnn(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=X_train.shape[1],
        hidden_dims=config["model"]["fcnn_hidden_dims"],
        logger=experiment_logger,
        checkpoint_mgr=checkpoint_mgr,
        device=device,
        epochs=config["training"]["epochs"],
        lr=config["training"]["learning_rate"],
    )

    # Evaluate models
    logger.info("Evaluating models on test set...")
    evaluate_models(
        ridge_model,
        fcnn_model,
        test_ds,
        experiment_logger,
        device,
        config["training"]["batch_size"],
    )

    # Save processed dataset
    dataset_storage.save_processed(
        {"X": X_train, "y": y_train},
        "test_dataset",
        "v1",
        {"source": "infrastructure_test"},
    )
    logger.info("Saved processed dataset")

    # Cleanup
    experiment_logger.close()
    cache_mgr.close()
    logger.info("Infrastructure testing completed")


if __name__ == "__main__":
    main()
