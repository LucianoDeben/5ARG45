# evaluation/evaluator.py
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

from utils.logging import ExperimentLogger

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Model evaluation service for comprehensive performance assessment.

    Features:
    - Multiple metric calculation
    - Result visualization
    - Performance reporting
    - Cross-validation support
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        exp_logger: Optional[ExperimentLogger] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            model: PyTorch model to evaluate
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            exp_logger: Optional experiment logger for tracking results
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_logger = exp_logger

        # Move model to device
        self.model.to(self.device)

        # Set model to evaluation mode
        self.model.eval()

    def evaluate(
        self,
        data_loader: DataLoader,
        metrics: Optional[List[str]] = None,
        criterion: Optional[Callable] = None,
        output_dir: Optional[str] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            data_loader: DataLoader with evaluation data
            metrics: List of metrics to calculate ('r2', 'rmse', 'mae', 'pearson', 'all')
            criterion: Loss function
            output_dir: Directory to save evaluation results and plots
            prefix: Prefix for saved files

        Returns:
            Dictionary of evaluation metrics
        """
        # Set default metrics if not provided
        if metrics is None:
            metrics = ["r2", "rmse", "mae", "pearson"]
        elif "all" in metrics:
            metrics = ["r2", "rmse", "mae", "pearson"]

        # Set default criterion if not provided
        if criterion is None:
            criterion = nn.MSELoss()

        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Initialize metrics
        all_targets = []
        all_outputs = []
        total_loss = 0.0
        num_samples = len(data_loader.dataset)

        # Evaluate model
        with torch.no_grad():
            for batch in data_loader:
                # Get batch data
                if isinstance(batch, dict):
                    # For MultimodalDataset returning dictionary
                    inputs = batch
                    targets = batch.get("viability")
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # For standard (X, y) tuple
                    inputs, targets = batch
                else:
                    raise ValueError(f"Unsupported batch format: {type(batch)}")

                # Move data to device
                if isinstance(inputs, dict):
                    inputs = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }
                else:
                    inputs = inputs.to(self.device)

                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Calculate loss
                loss = criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)

                # Store predictions and targets
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        # Calculate average loss
        avg_loss = total_loss / num_samples

        # Concatenate predictions and targets
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)

        # Calculate metrics
        results = {"loss": avg_loss}

        if "r2" in metrics:
            results["r2"] = r2_score(all_targets, all_outputs)

        if "rmse" in metrics:
            results["rmse"] = np.sqrt(mean_squared_error(all_targets, all_outputs))

        if "mae" in metrics:
            results["mae"] = mean_absolute_error(all_targets, all_outputs)

        if "pearson" in metrics:
            pearson_corr, _ = pearsonr(all_targets.flatten(), all_outputs.flatten())
            results["pearson"] = pearson_corr

        # Log results
        if self.exp_logger:
            self.exp_logger.log_metrics(results, step=0, phase=f"{prefix}eval")

        # Save results to file
        if output_dir:
            # Save metrics to JSON
            metrics_file = os.path.join(output_dir, f"{prefix}metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(results, f, indent=2)

            # Generate and save scatter plot
            self._plot_predictions(
                all_targets, all_outputs, results, output_dir, prefix
            )

            # Save raw predictions
            predictions_df = pd.DataFrame(
                {
                    "target": all_targets.flatten(),
                    "prediction": all_outputs.flatten(),
                }
            )
            predictions_file = os.path.join(output_dir, f"{prefix}predictions.csv")
            predictions_df.to_csv(predictions_file, index=False)

        return results

    def evaluate_by_group(
        self,
        data_loader: DataLoader,
        group_column: str,
        metrics: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        prefix: str = "",
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance by group (e.g., cell line, drug type).

        Args:
            data_loader: DataLoader with evaluation data
            group_column: Column name to group by
            metrics: List of metrics to calculate
            output_dir: Directory to save evaluation results
            prefix: Prefix for saved files

        Returns:
            Dictionary of evaluation metrics per group
        """
        # Set default metrics if not provided
        if metrics is None:
            metrics = ["r2", "rmse", "mae", "pearson"]
        elif "all" in metrics:
            metrics = ["r2", "rmse", "mae", "pearson"]

        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Initialize metrics
        groups = {}

        # Collect all predictions
        all_predictions = []

        # Evaluate model
        with torch.no_grad():
            for batch in data_loader:
                # Get batch data
                if isinstance(batch, dict):
                    # For MultimodalDataset returning dictionary
                    inputs = batch
                    targets = batch.get("viability")
                    group_values = batch.get(group_column)
                elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                    # For custom tuple with group information
                    inputs, targets, group_values = batch
                else:
                    raise ValueError(
                        f"Unsupported batch format or missing group column: {type(batch)}"
                    )

                # Move data to device
                if isinstance(inputs, dict):
                    inputs = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }
                else:
                    inputs = inputs.to(self.device)

                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Store predictions, targets, and groups
                for i in range(len(targets)):
                    group = (
                        group_values[i].item()
                        if isinstance(group_values, torch.Tensor)
                        else group_values[i]
                    )

                    if group not in groups:
                        groups[group] = {
                            "targets": [],
                            "outputs": [],
                        }

                    groups[group]["targets"].append(targets[i].cpu().numpy())
                    groups[group]["outputs"].append(outputs[i].cpu().numpy())

                    all_predictions.append(
                        {
                            "group": group,
                            "target": targets[i].cpu().numpy().item(),
                            "prediction": outputs[i].cpu().numpy().item(),
                        }
                    )

        # Calculate metrics per group
        results = {}

        for group, data in groups.items():
            group_targets = np.array(data["targets"])
            group_outputs = np.array(data["outputs"])

            group_results = {}

            if "r2" in metrics:
                group_results["r2"] = r2_score(group_targets, group_outputs)

            if "rmse" in metrics:
                group_results["rmse"] = np.sqrt(
                    mean_squared_error(group_targets, group_outputs)
                )

            if "mae" in metrics:
                group_results["mae"] = mean_absolute_error(group_targets, group_outputs)

            if "pearson" in metrics:
                pearson_corr, _ = pearsonr(
                    group_targets.flatten(), group_outputs.flatten()
                )
                group_results["pearson"] = pearson_corr

            results[group] = group_results

        # Save results to file
        if output_dir:
            # Save metrics to JSON
            metrics_file = os.path.join(output_dir, f"{prefix}group_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(results, f, indent=2)

            # Save raw predictions
            predictions_df = pd.DataFrame(all_predictions)
            predictions_file = os.path.join(
                output_dir, f"{prefix}group_predictions.csv"
            )
            predictions_df.to_csv(predictions_file, index=False)

            # Generate and save group comparison plot
            self._plot_group_comparison(results, metrics, output_dir, prefix)

        return results

    def cross_validate(
        self,
        dataset: Dataset,
        n_splits: int = 5,
        metrics: Optional[List[str]] = None,
        batch_size: int = 32,
        output_dir: Optional[str] = None,
        prefix: str = "",
        stratify_column: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation to evaluate model stability.

        Args:
            dataset: PyTorch dataset
            n_splits: Number of cross-validation folds
            metrics: List of metrics to calculate
            batch_size: Batch size for DataLoader
            output_dir: Directory to save evaluation results
            prefix: Prefix for saved files
            stratify_column: Column name to use for stratified sampling

        Returns:
            Dictionary of cross-validation results
        """
        try:
            from sklearn.model_selection import KFold, StratifiedKFold
        except ImportError:
            logger.error("scikit-learn is required for cross-validation")
            raise

        # Set default metrics if not provided
        if metrics is None:
            metrics = ["r2", "rmse", "mae", "pearson"]
        elif "all" in metrics:
            metrics = ["r2", "rmse", "mae", "pearson"]

        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Initialize results dictionary
        results = {metric: [] for metric in metrics}
        results["loss"] = []

        # Create cross-validation splitter
        if stratify_column is not None:
            if (
                not hasattr(dataset, "metadata")
                or stratify_column not in dataset.metadata.columns
            ):
                logger.warning(
                    f"Stratify column '{stratify_column}' not found, falling back to regular KFold"
                )
                splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                y = None
            else:
                stratify_values = dataset.metadata[stratify_column].values
                splitter = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=42
                )
                y = stratify_values
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            y = None

        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(range(len(dataset)), y)
        ):
            logger.info(f"Evaluating fold {fold+1}/{n_splits}")

            # Create train and validation subdatasets
            if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
                val_dataset = torch.utils.data.Subset(dataset, val_idx)
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False
                )
            else:
                # Handle special case for custom datasets
                val_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    sampler=torch.utils.data.SubsetRandomSampler(val_idx),
                )

            # Evaluate on validation set
            fold_results = self.evaluate(
                val_loader,
                metrics=metrics,
                output_dir=output_dir if output_dir else None,
                prefix=f"{prefix}fold{fold+1}_",
            )

            # Store results
            for metric, value in fold_results.items():
                results[metric].append(value)

        # Calculate aggregate statistics
        aggregate_results = {}

        for metric, values in results.items():
            aggregate_results[f"{metric}_mean"] = np.mean(values)
            aggregate_results[f"{metric}_std"] = np.std(values)
            aggregate_results[f"{metric}_min"] = np.min(values)
            aggregate_results[f"{metric}_max"] = np.max(values)

        # Save aggregate results
        if output_dir:
            # Save metrics to JSON
            metrics_file = os.path.join(output_dir, f"{prefix}cv_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(aggregate_results, f, indent=2)

            # Generate and save boxplot for each metric
            self._plot_cv_results(results, output_dir, prefix)

        # Log results
        if self.exp_logger:
            self.exp_logger.log_metrics(aggregate_results, step=0, phase=f"{prefix}cv")

        return aggregate_results

    def _plot_predictions(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        metrics: Dict[str, float],
        output_dir: str,
        prefix: str = "",
    ) -> None:
        """Generate and save scatter plot of predictions vs. targets."""
        plt.figure(figsize=(10, 8))

        # Create scatter plot
        plt.scatter(targets, predictions, alpha=0.5)

        # Add perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")

        # Add metrics text
        metrics_str = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        plt.text(
            0.05,
            0.95,
            metrics_str,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Set labels and title
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("Predicted vs. True Values")

        # Add grid
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save figure
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{prefix}predictions_scatter.png"), dpi=300
        )
        plt.close()

    def _plot_group_comparison(
        self,
        group_results: Dict[str, Dict[str, float]],
        metrics: List[str],
        output_dir: str,
        prefix: str = "",
    ) -> None:
        """Generate and save bar charts comparing performance across groups."""
        for metric in metrics:
            plt.figure(figsize=(12, 6))

            # Extract values for this metric across groups
            groups = list(group_results.keys())
            values = [group_results[group].get(metric, 0) for group in groups]

            # Sort groups by performance
            sorted_indices = np.argsort(values)
            sorted_groups = [groups[i] for i in sorted_indices]
            sorted_values = [values[i] for i in sorted_indices]

            # Create bar chart
            plt.bar(range(len(sorted_groups)), sorted_values)

            # Set labels and title
            plt.xlabel("Group")
            plt.ylabel(metric)
            plt.title(f"{metric.upper()} by Group")

            # Add group labels
            plt.xticks(range(len(sorted_groups)), sorted_groups, rotation=90)

            # Add grid
            plt.grid(True, axis="y", linestyle="--", alpha=0.6)

            # Save figure
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{prefix}group_{metric}_comparison.png"),
                dpi=300,
            )
            plt.close()

    def _plot_cv_results(
        self,
        cv_results: Dict[str, List[float]],
        output_dir: str,
        prefix: str = "",
    ) -> None:
        """Generate and save boxplots of cross-validation results."""
        plt.figure(figsize=(10, 6))

        # Extract metrics and values
        metrics = list(cv_results.keys())
        values = [cv_results[metric] for metric in metrics]

        # Create boxplot
        plt.boxplot(values, labels=metrics)

        # Set labels and title
        plt.xlabel("Metric")
        plt.ylabel("Value")
        plt.title("Cross-Validation Results")

        # Add grid
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}cv_results.png"), dpi=300)
        plt.close()
