import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

from src.config.config_utils import load_config, merge_configs, validate_config
from src.config.default_config import get_default_config
from src.data.preprocessing import LINCSCTRPDataProcessor
from src.data.feature_transforms import create_feature_transform

logger = logging.getLogger(__name__)


# Define activation and normalization options
ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "tanh": nn.Tanh,
}

NORM_LAYERS = {
    "batchnorm": nn.BatchNorm1d,
    "layernorm": nn.LayerNorm,
    "instancenorm": nn.InstanceNorm1d,
    "none": nn.Identity,
}


class FlexibleFCNN(nn.Module):
    """
    Flexible fully connected neural network for transcriptomics data.

    Supports various architectures, activation functions, normalization,
    and initialization methods.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=[512, 256, 128, 64],
        output_dim=1,
        activation_fn="relu",
        dropout_prob=0.2,
        residual=False,
        norm_type="batchnorm",
        weight_init="kaiming",
    ):
        super(FlexibleFCNN, self).__init__()
        self.residual = residual

        # Ensure hidden_dims are consistent for residual
        if residual:
            hidden_dims = [hidden_dims[0]] * len(hidden_dims)

        self.activation = ACTIVATIONS.get(activation_fn.lower(), nn.ReLU)()
        self.norm_type = norm_type.lower()

        # Build layers
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            norm = (
                NORM_LAYERS.get(self.norm_type, nn.Identity)(dims[i + 1])
                if norm_type != "none"
                else nn.Identity()
            )
            self.norms.append(norm)

        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.output = nn.Linear(hidden_dims[-1], output_dim)

        self._initialize_weights(weight_init)

    def _initialize_weights(self, method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = x
        for layer, norm in zip(self.layers, self.norms):
            identity = out  # Save for residual

            # Main path
            out = layer(out)
            out = norm(out)

            # Optional residual
            if self.residual and (out.shape == identity.shape):
                out = out + identity

            out = self.activation(out)
            out = self.dropout(out)

        return self.output(out)


def dataset_to_numpy(dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert MultimodalDataset to numpy arrays for scikit-learn.

    This function directly accesses dataset attributes instead of using method
    chaining to avoid potential issues with unimodal_type preservation.
    """
    # Direct access to transcriptomics data and viability
    X = dataset.transcriptomics_data
    y = dataset.metadata["viability"].values
    return X, y


def compute_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Pearson correlation coefficient."""
    correlation, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return correlation


def evaluate_ridge(config: Dict, num_runs: int = 20) -> Dict:
    """
    Evaluate Ridge regression model across multiple runs.

    Creates a fresh data split for each run to ensure proper variability.
    """
    results = {}

    for split_type in ["random", "group"]:
        logger.info(f"Evaluating Ridge regression with {split_type} split...")

        # Store metrics for each run
        train_r2_scores = []
        train_pearson_scores = []
        val_r2_scores = []
        val_pearson_scores = []
        test_r2_scores = []
        test_pearson_scores = []

        # Run multiple times with different random states AND different splits
        for run in range(num_runs):
            # Create new data split for each run
            random_state = run + config["training"]["random_state"]

            # Create processor with unique random state for this run
            processor = LINCSCTRPDataProcessor(
                gctx_file=config["data"]["gctx_file"],
                feature_space=config["data"]["feature_space"],
                nrows=config["data"].get("nrows"),
                test_size=config["training"]["test_size"],
                val_size=config["training"]["val_size"],
                random_state=random_state,  # Different for each run
                group_by="cell_mfc_name" if split_type == "group" else None,
                batch_size=config["training"]["batch_size"],
                additional_features=[],
            )

            # Get fresh data splits
            train_dataset, val_dataset, test_dataset = processor.process()

            # Convert to unimodal datasets
            train_dataset = train_dataset.to_unimodal("transcriptomics")
            val_dataset = val_dataset.to_unimodal("transcriptomics")
            test_dataset = test_dataset.to_unimodal("transcriptomics")

            # Get data
            train_X, train_y = dataset_to_numpy(train_dataset)
            val_X, val_y = dataset_to_numpy(val_dataset)
            test_X, test_y = dataset_to_numpy(test_dataset)

            # Train Ridge regression model
            model = Ridge(alpha=1.0)
            model.fit(train_X, train_y)

            # Predict on all sets
            train_pred = model.predict(train_X)
            val_pred = model.predict(val_X)
            test_pred = model.predict(test_X)

            # Compute metrics
            train_r2 = r2_score(train_y, train_pred)
            train_pearson = compute_pearson_correlation(train_y, train_pred)
            val_r2 = r2_score(val_y, val_pred)
            val_pearson = compute_pearson_correlation(val_y, val_pred)
            test_r2 = r2_score(test_y, test_pred)
            test_pearson = compute_pearson_correlation(test_y, test_pred)

            # Store metrics
            train_r2_scores.append(train_r2)
            train_pearson_scores.append(train_pearson)
            val_r2_scores.append(val_r2)
            val_pearson_scores.append(val_pearson)
            test_r2_scores.append(test_r2)
            test_pearson_scores.append(test_pearson)

        # Calculate statistics
        results[split_type] = {
            "train_r2_mean": np.mean(train_r2_scores),
            "train_r2_std": np.std(train_r2_scores),
            "train_pearson_mean": np.mean(train_pearson_scores),
            "train_pearson_std": np.std(train_pearson_scores),
            "val_r2_mean": np.mean(val_r2_scores),
            "val_r2_std": np.std(val_r2_scores),
            "val_pearson_mean": np.mean(val_pearson_scores),
            "val_pearson_std": np.std(val_pearson_scores),
            "test_r2_mean": np.mean(test_r2_scores),
            "test_r2_std": np.std(test_r2_scores),
            "test_pearson_mean": np.mean(test_pearson_scores),
            "test_pearson_std": np.std(test_pearson_scores),
        }

        # Log results once after all runs are complete
        logger.info(
            f"Ridge (Transcriptomics, {split_type.capitalize()} Split) - Train R2 (Mean ± Std): {results[split_type]['train_r2_mean']:.4f} ± {results[split_type]['train_r2_std']:.4f}"
        )
        logger.info(
            f"Ridge (Transcriptomics, {split_type.capitalize()} Split) - Train Pearson (Mean ± Std): {results[split_type]['train_pearson_mean']:.4f} ± {results[split_type]['train_pearson_std']:.4f}"
        )
        logger.info(
            f"Ridge (Transcriptomics, {split_type.capitalize()} Split) - Val R2 (Mean ± Std): {results[split_type]['val_r2_mean']:.4f} ± {results[split_type]['val_r2_std']:.4f}"
        )
        logger.info(
            f"Ridge (Transcriptomics, {split_type.capitalize()} Split) - Val Pearson (Mean ± Std): {results[split_type]['val_pearson_mean']:.4f} ± {results[split_type]['val_pearson_std']:.4f}"
        )
        logger.info(
            f"Ridge (Transcriptomics, {split_type.capitalize()} Split) - Test R2 (Mean ± Std): {results[split_type]['test_r2_mean']:.4f} ± {results[split_type]['test_r2_std']:.4f}"
        )
        logger.info(
            f"Ridge (Transcriptomics, {split_type.capitalize()} Split) - Test Pearson (Mean ± Std): {results[split_type]['test_pearson_mean']:.4f} ± {results[split_type]['test_pearson_std']:.4f}"
        )

    return results


def train_fcnn_epoch(model, dataloader, criterion, optimizer, device):
    """Train FCNN for one epoch."""
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


# Change this function name
def evaluate_model(model, dataloader, criterion, device):
    """Evaluate FCNN on a dataset."""
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            # Store predictions and targets for metrics
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    # Concatenate batches
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)

    # Calculate metrics
    r2 = r2_score(all_targets, all_outputs)
    pearson = compute_pearson_correlation(all_targets, all_outputs)
    mse = running_loss / len(dataloader.dataset)

    return mse, r2, pearson


def create_torch_dataloader(X, y, batch_size, shuffle=True):
    """Create a PyTorch DataLoader from numpy arrays."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def evaluate_fcnn(config: Dict, num_runs: int = 20) -> Dict:
    """
    Evaluate FCNN model across multiple runs.

    Creates a fresh data split for each run to ensure proper variability.
    """
    results = {}

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    for split_type in ["random", "group"]:
        logger.info(f"Evaluating FCNN with {split_type} split...")

        # Store metrics for each run
        train_mse_scores = []
        train_r2_scores = []
        train_pearson_scores = []
        val_mse_scores = []
        val_r2_scores = []
        val_pearson_scores = []
        test_mse_scores = []
        test_r2_scores = []
        test_pearson_scores = []

        # Run multiple times with different seeds and data splits
        for run in range(num_runs):
            # Create new data split for each run
            random_state = run + config["training"]["random_state"]

            # Set seed for reproducibility
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state)

            # Create processor with unique random state for this run
            processor = LINCSCTRPDataProcessor(
                gctx_file=config["data"]["gctx_file"],
                feature_space=config["data"]["feature_space"],
                nrows=config["data"].get("nrows"),
                test_size=config["training"]["test_size"],
                val_size=config["training"]["val_size"],
                random_state=random_state,  # Different for each run
                group_by="cell_mfc_name" if split_type == "group" else None,
                batch_size=config["training"]["batch_size"],
                additional_features=[],
            )

            # Get fresh data splits
            train_dataset, val_dataset, test_dataset = processor.process()

            # Convert to unimodal datasets
            train_dataset = train_dataset.to_unimodal("transcriptomics")
            val_dataset = val_dataset.to_unimodal("transcriptomics")
            test_dataset = test_dataset.to_unimodal("transcriptomics")

            # Get data
            train_X, train_y = dataset_to_numpy(train_dataset)
            val_X, val_y = dataset_to_numpy(val_dataset)
            test_X, test_y = dataset_to_numpy(test_dataset)

            # Create DataLoaders
            batch_size = config["training"]["batch_size"]
            train_loader = create_torch_dataloader(
                train_X, train_y, batch_size, shuffle=True
            )
            val_loader = create_torch_dataloader(
                val_X, val_y, batch_size, shuffle=False
            )
            test_loader = create_torch_dataloader(
                test_X, test_y, batch_size, shuffle=False
            )

            # Get input dimension
            input_dim = train_X.shape[1]

            # Initialize model with more complex architecture
            model = FlexibleFCNN(
                input_dim=input_dim,
                hidden_dims=[256, 128, 64, 32],  # More complex architecture
                output_dim=1,
                activation_fn="relu",
                dropout_prob=0.2,
                residual=False,
                norm_type="batchnorm",
                weight_init="kaiming",
            ).to(device)

            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Train for specified number of epochs
            epochs = 10
            for epoch in range(epochs):
                train_loss = train_fcnn_epoch(
                    model, train_loader, criterion, optimizer, device
                )

                # Evaluate on validation set every 5 epochs or on final epoch
                if epoch % 5 == 4 or epoch == epochs - 1:
                    val_loss, val_r2, val_pearson = evaluate_model(
                        model, val_loader, criterion, device
                    )
                    logger.debug(
                        f"Run {run+1}, Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}"
                    )

            # Final evaluation on all sets
            train_mse, train_r2, train_pearson = evaluate_model(
                model, train_loader, criterion, device
            )
            val_mse, val_r2, val_pearson = evaluate_model(
                model, val_loader, criterion, device
            )
            test_mse, test_r2, test_pearson = evaluate_model(
                model, test_loader, criterion, device
            )

            # Store metrics
            train_mse_scores.append(train_mse)
            train_r2_scores.append(train_r2)
            train_pearson_scores.append(train_pearson)
            val_mse_scores.append(val_mse)
            val_r2_scores.append(val_r2)
            val_pearson_scores.append(val_pearson)
            test_mse_scores.append(test_mse)
            test_r2_scores.append(test_r2)
            test_pearson_scores.append(test_pearson)

        # Calculate statistics
        results[split_type] = {
            "train_mse_mean": np.mean(train_mse_scores),
            "train_mse_std": np.std(train_mse_scores),
            "train_r2_mean": np.mean(train_r2_scores),
            "train_r2_std": np.std(train_r2_scores),
            "train_pearson_mean": np.mean(train_pearson_scores),
            "train_pearson_std": np.std(train_pearson_scores),
            "val_mse_mean": np.mean(val_mse_scores),
            "val_mse_std": np.std(val_mse_scores),
            "val_r2_mean": np.mean(val_r2_scores),
            "val_r2_std": np.std(val_r2_scores),
            "val_pearson_mean": np.mean(val_pearson_scores),
            "val_pearson_std": np.std(val_pearson_scores),
            "test_mse_mean": np.mean(test_mse_scores),
            "test_mse_std": np.std(test_mse_scores),
            "test_r2_mean": np.mean(test_r2_scores),
            "test_r2_std": np.std(test_r2_scores),
            "test_pearson_mean": np.mean(test_pearson_scores),
            "test_pearson_std": np.std(test_pearson_scores),
        }

        # Log results once after all runs are complete
        logger.info(
            f"FCNN (Transcriptomics, {split_type.capitalize()} Split) - Train MSE (Mean ± Std): {results[split_type]['train_mse_mean']:.4f} ± {results[split_type]['train_mse_std']:.4f}"
        )
        logger.info(
            f"FCNN (Transcriptomics, {split_type.capitalize()} Split) - Train R2 (Mean ± Std): {results[split_type]['train_r2_mean']:.4f} ± {results[split_type]['train_r2_std']:.4f}"
        )
        logger.info(
            f"FCNN (Transcriptomics, {split_type.capitalize()} Split) - Train Pearson (Mean ± Std): {results[split_type]['train_pearson_mean']:.4f} ± {results[split_type]['train_pearson_std']:.4f}"
        )
        logger.info(
            f"FCNN (Transcriptomics, {split_type.capitalize()} Split) - Val MSE (Mean ± Std): {results[split_type]['val_mse_mean']:.4f} ± {results[split_type]['val_mse_std']:.4f}"
        )
        logger.info(
            f"FCNN (Transcriptomics, {split_type.capitalize()} Split) - Val R2 (Mean ± Std): {results[split_type]['val_r2_mean']:.4f} ± {results[split_type]['val_r2_std']:.4f}"
        )
        logger.info(
            f"FCNN (Transcriptomics, {split_type.capitalize()} Split) - Val Pearson (Mean ± Std): {results[split_type]['val_pearson_mean']:.4f} ± {results[split_type]['val_pearson_std']:.4f}"
        )
        logger.info(
            f"FCNN (Transcriptomics, {split_type.capitalize()} Split) - Test MSE (Mean ± Std): {results[split_type]['test_mse_mean']:.4f} ± {results[split_type]['test_mse_std']:.4f}"
        )
        logger.info(
            f"FCNN (Transcriptomics, {split_type.capitalize()} Split) - Test R2 (Mean ± Std): {results[split_type]['test_r2_mean']:.4f} ± {results[split_type]['test_r2_std']:.4f}"
        )
        logger.info(
            f"FCNN (Transcriptomics, {split_type.capitalize()} Split) - Test Pearson (Mean ± Std): {results[split_type]['test_pearson_mean']:.4f} ± {results[split_type]['test_pearson_std']:.4f}"
        )

    return results


def main():
    """Main function to orchestrate the testing process."""
    try:
        logger.info("Starting unimodal transcriptomics model evaluation")

        # Parse command-line arguments
        parser = argparse.ArgumentParser(
            description="Evaluate unimodal transcriptomics models"
        )
        parser.add_argument(
            "--config",
            type=str,
            default="./config.yaml",
            help="Path to configuration file",
        )
        parser.add_argument(
            "--nrows",
            type=int,
            default=None,
            help="Number of rows to use (None for all)",
        )
        args = parser.parse_args()

        # Load and merge configurations
        default_config = get_default_config()
        custom_config = load_config(args.config)
        config = merge_configs(default_config, custom_config)

        # Validate configuration
        validate_config(config)

        # Update configuration for unimodal test
        config["data"]["feature_space"] = "landmark"  # Ensure using landmark genes
        config["training"]["test_size"] = 0.4  # 40% test split
        config["training"]["val_size"] = 1 / 6  # ~10% validation (1/6 of remaining 60%)

        # Override nrows if specified
        if args.nrows is not None:
            config["data"]["nrows"] = args.nrows

        logger.info(
            f"Using configuration: data.gctx_file={config['data']['gctx_file']}, "
            f"data.feature_space={config['data']['feature_space']}, "
            f"nrows={config['data'].get('nrows')}"
        )

        # # Evaluate Ridge regression (creates its own data splits)
        # logger.info("Evaluating Ridge regression model...")
        # ridge_results = evaluate_ridge(config)

        # Evaluate FCNN (creates its own data splits)
        logger.info("Evaluating FCNN model...")
        fcnn_results = evaluate_fcnn(config)

        logger.info("Evaluation completed successfully")

        return {"fcnn": fcnn_results}

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
