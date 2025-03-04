# evaluation/evaluator.py
import json
import logging
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

from ..config.config_utils import load_config
from ..utils.logging import ExperimentLogger

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates model performance for multimodal drug response prediction.

    Supports comprehensive performance assessment with metrics, visualizations,
    and cross-validation, configured via a configuration dictionary. Designed
    for multimodal data (e.g., transcriptomics and chemical features) from
    MultimodalDrugDataset.

    Attributes:
        model: PyTorch model instance.
        device: Device for computation ('cuda' or 'cpu').
        exp_logger: ExperimentLogger for tracking results.
        config: Configuration dictionary for evaluation parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        exp_logger: Optional[ExperimentLogger] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the Evaluator.

        Args:
            model: PyTorch model to evaluate.
            device: Device to use ('cuda', 'cpu', or None for auto-detection).
            exp_logger: Optional ExperimentLogger for tracking results.
            config: Configuration dictionary (loads default if None).
        """
        self.model = model
        self.config = config or load_config("config.yaml")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_logger = exp_logger or ExperimentLogger()

        # Move model to device and set to evaluation mode
        self.model.to(self.device).eval()

    def evaluate(
        self,
        data_loader: DataLoader,
        metrics: Optional[List[str]] = None,
        criterion: Optional[Callable] = None,
        output_dir: Optional[str] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Evaluate model performance on a dataset.

        Args:
            data_loader: DataLoader with evaluation data from MultimodalDrugDataset.
            metrics: List of metrics to calculate ('r2', 'rmse', 'mae', 'pearson', 'all').
                     Defaults to config if None.
            criterion: Loss function (defaults to config if None).
            output_dir: Directory to save results and plots (defaults to config if None).
            prefix: Prefix for saved files.

        Returns:
            Dictionary of evaluation metrics.
        """
        eval_cfg = self.config.get("evaluation", {})
        metrics = metrics or eval_cfg.get("metrics", ["r2", "rmse", "mae", "pearson"])
        if "all" in metrics:
            metrics = ["r2", "rmse", "mae", "pearson"]

        criterion = criterion or self._create_criterion()
        output_dir = output_dir or eval_cfg.get("output_dir", "results/eval")

        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        all_targets, all_outputs = [], []
        total_loss = 0.0
        num_samples = len(data_loader.dataset)

        with torch.no_grad():
            for batch in data_loader:
                if not isinstance(batch, dict) or "viability" not in batch:
                    raise ValueError(
                        "Batch must be a dict with 'viability' key and multimodal data"
                    )

                inputs = {
                    k: v.to(self.device) for k, v in batch.items() if k != "viability"
                }
                targets = batch["viability"].to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)

                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        avg_loss = total_loss / num_samples
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)

        results = {"loss": avg_loss}
        for metric in metrics:
            if metric == "r2":
                results["r2"] = r2_score(all_targets, all_outputs)
            elif metric == "rmse":
                results["rmse"] = np.sqrt(mean_squared_error(all_targets, all_outputs))
            elif metric == "mae":
                results["mae"] = mean_absolute_error(all_targets, all_outputs)
            elif metric == "pearson":
                results["pearson"] = pearsonr(
                    all_targets.flatten(), all_outputs.flatten()
                )[0]

        if self.exp_logger:
            self.exp_logger.log_metrics(results, step=0, phase=f"{prefix}eval")

        if output_dir:
            self._save_results(results, all_targets, all_outputs, output_dir, prefix)

        return results

    def _create_criterion(self) -> Callable:
        """Create loss function from config."""
        eval_cfg = self.config.get("evaluation", {})
        loss_type = eval_cfg.get("loss", "mse").lower()
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "mae":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss: {loss_type}")

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
            data_loader: DataLoader with evaluation data from MultimodalDrugDataset.
            group_column: Key in batch dict for grouping (e.g., 'cell_line', 'drug').
            metrics: List of metrics to calculate (defaults to config if None).
            output_dir: Directory to save results (defaults to config if None).
            prefix: Prefix for saved files.

        Returns:
            Dictionary of evaluation metrics per group.
        """
        eval_cfg = self.config.get("evaluation", {})
        metrics = metrics or eval_cfg.get("metrics", ["r2", "rmse", "mae", "pearson"])
        if "all" in metrics:
            metrics = ["r2", "rmse", "mae", "pearson"]

        output_dir = output_dir or eval_cfg.get("output_dir", "results/eval")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        groups = {}
        all_predictions = []

        with torch.no_grad():
            for batch in data_loader:
                if (
                    not isinstance(batch, dict)
                    or "viability" not in batch
                    or group_column not in batch
                ):
                    raise ValueError(
                        f"Batch must include 'viability' and '{group_column}' keys"
                    )

                inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k not in ["viability", group_column]
                }
                targets = batch["viability"].to(self.device)
                group_values = batch[group_column]

                outputs = self.model(inputs)

                for i in range(len(targets)):
                    group = (
                        group_values[i].item()
                        if isinstance(group_values, torch.Tensor)
                        else group_values[i]
                    )
                    if group not in groups:
                        groups[group] = {"targets": [], "outputs": []}

                    groups[group]["targets"].append(targets[i].cpu().numpy())
                    groups[group]["outputs"].append(outputs[i].cpu().numpy())
                    all_predictions.append(
                        {
                            "group": group,
                            "target": targets[i].cpu().numpy().item(),
                            "prediction": outputs[i].cpu().numpy().item(),
                        }
                    )

        results = {}
        for group, data in groups.items():
            group_targets = np.array(data["targets"])
            group_outputs = np.array(data["outputs"])

            group_results = {}
            for metric in metrics:
                if metric == "r2":
                    group_results["r2"] = r2_score(group_targets, group_outputs)
                elif metric == "rmse":
                    group_results["rmse"] = np.sqrt(
                        mean_squared_error(group_targets, group_outputs)
                    )
                elif metric == "mae":
                    group_results["mae"] = mean_absolute_error(
                        group_targets, group_outputs
                    )
                elif metric == "pearson":
                    group_results["pearson"] = pearsonr(
                        group_targets.flatten(), group_outputs.flatten()
                    )[0]

            results[group] = group_results

        if output_dir:
            self._save_group_results(
                results, all_predictions, metrics, output_dir, prefix
            )

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
        Perform k-fold cross-validation for model stability.

        Args:
            dataset: PyTorch dataset (e.g., MultimodalDrugDataset).
            n_splits: Number of cross-validation folds.
            metrics: List of metrics to calculate (defaults to config if None).
            batch_size: Batch size for DataLoader.
            output_dir: Directory to save results (defaults to config if None).
            prefix: Prefix for saved files.
            stratify_column: Key in dataset metadata for stratified sampling.

        Returns:
            Dictionary of aggregate cross-validation metrics.
        """
        eval_cfg = self.config.get("evaluation", {})
        metrics = metrics or eval_cfg.get("metrics", ["r2", "rmse", "mae", "pearson"])
        if "all" in metrics:
            metrics = ["r2", "rmse", "mae", "pearson"]

        output_dir = output_dir or eval_cfg.get("output_dir", "results/eval")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            from sklearn.model_selection import KFold, StratifiedKFold
        except ImportError:
            logger.error("scikit-learn is required for cross-validation")
            raise

        results = {metric: [] for metric in metrics}
        results["loss"] = []

        if stratify_column:
            if (
                not hasattr(dataset, "metadata")
                or stratify_column not in dataset.metadata.columns
            ):
                logger.warning(
                    f"Stratify column '{stratify_column}' not found, using regular KFold"
                )
                splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                y = None
            else:
                y = dataset.metadata[stratify_column].values
                splitter = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=42
                )
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            y = None

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(range(len(dataset)), y)
        ):
            logger.info(f"Evaluating fold {fold + 1}/{n_splits}")

            val_dataset = torch.utils.data.Subset(dataset, val_idx)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            fold_results = self.evaluate(
                val_loader,
                metrics=metrics,
                output_dir=(
                    output_dir if fold == 0 else None
                ),  # Save plots only for first fold
                prefix=f"{prefix}fold{fold+1}_",
            )

            for metric, value in fold_results.items():
                results[metric].append(value)

        aggregate_results = {}
        for metric, values in results.items():
            aggregate_results[f"{metric}_mean"] = np.mean(values)
            aggregate_results[f"{metric}_std"] = np.std(values)
            aggregate_results[f"{metric}_min"] = np.min(values)
            aggregate_results[f"{metric}_max"] = np.max(values)

        if self.exp_logger:
            self.exp_logger.log_metrics(aggregate_results, step=0, phase=f"{prefix}cv")

        if output_dir:
            self._save_cv_results(aggregate_results, results, output_dir, prefix)

        return aggregate_results

    def _save_results(
        self,
        metrics: Dict[str, float],
        targets: np.ndarray,
        predictions: np.ndarray,
        output_dir: str,
        prefix: str = "",
    ) -> None:
        """Save evaluation results, including metrics, predictions, and visualizations."""
        # Save metrics to JSON
        metrics_file = os.path.join(output_dir, f"{prefix}metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save predictions to CSV
        predictions_df = pd.DataFrame(
            {
                "target": targets.flatten(),
                "prediction": predictions.flatten(),
            }
        )
        predictions_file = os.path.join(output_dir, f"{prefix}predictions.csv")
        predictions_df.to_csv(predictions_file, index=False)

        # Generate and save scatter plot
        self._plot_predictions(targets, predictions, metrics, output_dir, prefix)

    def _save_group_results(
        self,
        group_results: Dict[str, Dict[str, float]],
        predictions: List[Dict],
        metrics: List[str],
        output_dir: str,
        prefix: str = "",
    ) -> None:
        """Save group-wise evaluation results and visualizations."""
        # Save metrics to JSON
        metrics_file = os.path.join(output_dir, f"{prefix}group_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(group_results, f, indent=2)

        # Save predictions to CSV
        predictions_df = pd.DataFrame(predictions)
        predictions_file = os.path.join(output_dir, f"{prefix}group_predictions.csv")
        predictions_df.to_csv(predictions_file, index=False)

        # Generate and save group comparison plots
        self._plot_group_comparison(group_results, metrics, output_dir, prefix)

    def _save_cv_results(
        self,
        aggregate_results: Dict[str, float],
        cv_results: Dict[str, List[float]],
        output_dir: str,
        prefix: str = "",
    ) -> None:
        """Save cross-validation results and visualizations."""
        # Save metrics to JSON
        metrics_file = os.path.join(output_dir, f"{prefix}cv_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(aggregate_results, f, indent=2)

        # Generate and save boxplot
        self._plot_cv_results(cv_results, output_dir, prefix)

    def _plot_predictions(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        metrics: Dict[str, float],
        output_dir: str,
        prefix: str = "",
    ) -> None:
        """Generate and save a scatter plot of predictions vs. true values."""
        plt.figure(figsize=(10, 8))
        plt.scatter(targets, predictions, alpha=0.5, label="Predictions")

        # Perfect prediction line
        min_val, max_val = min(targets.min(), predictions.min()), max(
            targets.max(), predictions.max()
        )
        plt.plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction"
        )

        # Metrics text
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

        plt.xlabel("True Viability")
        plt.ylabel("Predicted Viability")
        plt.title("Predicted vs. True Viability")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
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
        """Generate and save bar charts comparing performance across groups for each metric."""
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            groups = list(group_results.keys())
            values = [group_results[group].get(metric, 0) for group in groups]

            # Sort by value for better visualization
            sorted_indices = np.argsort(values)
            sorted_groups = [groups[i] for i in sorted_indices]
            sorted_values = [values[i] for i in sorted_indices]

            plt.bar(range(len(sorted_groups)), sorted_values)
            plt.xticks(range(len(sorted_groups)), sorted_groups, rotation=90)
            plt.xlabel("Group")
            plt.ylabel(metric.upper())
            plt.title(f"{metric.upper()} by Group")
            plt.grid(True, axis="y", linestyle="--", alpha=0.6)
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
        metrics = list(cv_results.keys())
        values = [cv_results[metric] for metric in metrics]

        plt.boxplot(values, labels=metrics)
        plt.xlabel("Metric")
        plt.ylabel("Value")
        plt.title("Cross-Validation Performance")
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}cv_results.png"), dpi=300)
        plt.close()

    def multi_run_evaluate(
        self,
        models: List[nn.Module],
        data_loader: DataLoader,
        metrics: Optional[List[str]] = None,
        criterion: Optional[Callable] = None,
        output_dir: Optional[str] = None,
        prefix: str = "",
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple model runs for statistical analysis.

        Args:
            models: List of models from different training runs
            data_loader: DataLoader with evaluation data
            metrics: List of metrics to calculate
            criterion: Loss function
            output_dir: Directory to save results and plots
            prefix: Prefix for saved files

        Returns:
            Dictionary with aggregate statistics across runs
        """
        eval_cfg = self.config.get("evaluation", {})
        metrics = metrics or eval_cfg.get("metrics", ["r2", "rmse", "mae", "pearson"])
        if "all" in metrics:
            metrics = ["r2", "rmse", "mae", "pearson"]

        output_dir = output_dir or eval_cfg.get("output_dir", "results/eval")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Dictionary to store metrics for each run
        run_metrics = {f"run_{i}": {} for i in range(len(models))}
        all_predictions = []

        # Evaluate each model
        for i, model in enumerate(models):
            model.to(self.device).eval()

            run_prefix = f"{prefix}run_{i}_"

            # Collect predictions and targets
            all_targets, all_outputs = [], []
            with torch.no_grad():
                for batch in data_loader:
                    if not isinstance(batch, dict) or "viability" not in batch:
                        raise ValueError(
                            "Batch must be a dict with 'viability' key and multimodal data"
                        )

                    inputs = {
                        k: v.to(self.device)
                        for k, v in batch.items()
                        if k != "viability"
                    }
                    targets = batch["viability"].to(self.device)

                    outputs = model(inputs)
                    all_targets.append(targets.cpu().numpy())
                    all_outputs.append(outputs.cpu().numpy())

            # Concatenate all predictions
            targets = np.concatenate(all_targets)
            outputs = np.concatenate(all_outputs)

            # Store for later analysis
            for j in range(len(targets)):
                all_predictions.append(
                    {
                        "run": i,
                        "target": targets[j].item(),
                        "prediction": outputs[j].item(),
                    }
                )

            # Calculate metrics
            for metric in metrics:
                if metric == "r2":
                    run_metrics[f"run_{i}"]["r2"] = r2_score(targets, outputs)
                elif metric == "rmse":
                    run_metrics[f"run_{i}"]["rmse"] = np.sqrt(
                        mean_squared_error(targets, outputs)
                    )
                elif metric == "mae":
                    run_metrics[f"run_{i}"]["mae"] = mean_absolute_error(
                        targets, outputs
                    )
                elif metric == "pearson":
                    run_metrics[f"run_{i}"]["pearson"] = pearsonr(
                        targets.flatten(), outputs.flatten()
                    )[0]

            # Generate individual run plots if requested for first run
            if i == 0 and output_dir:
                self._save_results(
                    run_metrics[f"run_{i}"], targets, outputs, output_dir, run_prefix
                )

        # Calculate aggregate statistics
        aggregate_metrics = {}
        metrics_data = defaultdict(list)

        # Collect all metrics across runs
        for run_id, metrics_dict in run_metrics.items():
            for metric, value in metrics_dict.items():
                metrics_data[metric].append(value)

        # Calculate statistics
        for metric, values in metrics_data.items():
            aggregate_metrics[f"{metric}_mean"] = float(np.mean(values))
            aggregate_metrics[f"{metric}_std"] = float(np.std(values))
            aggregate_metrics[f"{metric}_min"] = float(np.min(values))
            aggregate_metrics[f"{metric}_max"] = float(np.max(values))

        # Save aggregate results
        if output_dir:
            # Save metrics to JSON
            metrics_file = os.path.join(output_dir, f"{prefix}multi_run_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(
                    {
                        "runs": run_metrics,
                        "aggregate": aggregate_metrics,
                        "num_runs": len(models),
                    },
                    f,
                    indent=2,
                )

            # Save all predictions to CSV
            predictions_df = pd.DataFrame(all_predictions)
            predictions_file = os.path.join(
                output_dir, f"{prefix}multi_run_predictions.csv"
            )
            predictions_df.to_csv(predictions_file, index=False)

            # Generate boxplot visualization
            self._plot_multi_run_metrics(metrics_data, output_dir, prefix)

        if self.exp_logger:
            self.exp_logger.log_metrics(
                aggregate_metrics, step=0, phase=f"{prefix}multi_run"
            )

        return {"runs": run_metrics, "aggregate": aggregate_metrics}

    def _plot_multi_run_metrics(
        self,
        metrics_data: Dict[str, List[float]],
        output_dir: str,
        prefix: str = "",
    ) -> None:
        """Generate boxplot visualization for metrics across multiple runs."""
        plt.figure(figsize=(12, 6))

        # Prepare data for boxplot
        metrics = list(metrics_data.keys())
        data = [metrics_data[m] for m in metrics]

        # Create boxplot
        box = plt.boxplot(data, labels=metrics, patch_artist=True)

        # Add individual run points
        for i, metric in enumerate(metrics):
            x = np.random.normal(i + 1, 0.04, size=len(metrics_data[metric]))
            plt.plot(x, metrics_data[metric], "r.", alpha=0.5)

        # Add mean and std annotations
        for i, metric in enumerate(metrics):
            values = metrics_data[metric]
            mean_val = np.mean(values)
            std_val = np.std(values)
            plt.annotate(
                f"μ={mean_val:.4f}\nσ={std_val:.4f}",
                xy=(i + 1, max(values) + 0.05 * (max(values) - min(values))),
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
            )

        # Set plot attributes
        plt.title(f"Metrics Across {len(next(iter(metrics_data.values())))} Runs")
        plt.ylabel("Value")
        plt.grid(True, linestyle="--", alpha=0.7)

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}multi_run_metrics.png"), dpi=300)
        plt.close()


class MultiDatasetEvaluator:
    """
    Evaluates model performance across multiple datasets with standardized metrics.

    This class facilitates performance comparison across different datasets,
    supporting both in-domain and cross-dataset evaluation with appropriate
    normalization and transformation when needed.

    Attributes:
        model: PyTorch model instance.
        device: Device for computation ('cuda' or 'cpu').
        exp_logger: ExperimentLogger for tracking results.
        config: Configuration dictionary for evaluation parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        exp_logger: Optional[ExperimentLogger] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the MultiDatasetEvaluator.

        Args:
            model: PyTorch model to evaluate.
            device: Device to use ('cuda', 'cpu', or None for auto-detection).
            exp_logger: Optional ExperimentLogger for tracking results.
            config: Configuration dictionary (loads default if None).
        """
        self.model = model
        self.config = config or load_config("config.yaml")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_logger = exp_logger or ExperimentLogger()

        # Move model to device and set to evaluation mode
        self.model.to(self.device).eval()

        # Create standard evaluator
        self.evaluator = Evaluator(model, device, exp_logger, config)

    def evaluate_dataset(
        self,
        data_loader: DataLoader,
        dataset_name: str,
        metrics: Optional[List[str]] = None,
        criterion: Optional[Callable] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a specific dataset.

        Args:
            data_loader: DataLoader with evaluation data.
            dataset_name: Name of the dataset for logging and output files.
            metrics: List of metrics to calculate.
            criterion: Loss function.
            output_dir: Directory to save results and plots.

        Returns:
            Dictionary of evaluation metrics.
        """
        # Create dataset-specific output directory
        if output_dir:
            dataset_output_dir = os.path.join(output_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)
        else:
            dataset_output_dir = None

        # Evaluate using standard evaluator
        results = self.evaluator.evaluate(
            data_loader=data_loader,
            metrics=metrics,
            criterion=criterion,
            output_dir=dataset_output_dir,
            prefix=f"{dataset_name}_",
        )

        return results

    def evaluate_multiple_datasets(
        self,
        datasets: Dict[str, DataLoader],
        metrics: Optional[List[str]] = None,
        criterion: Optional[Callable] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on multiple datasets.

        Args:
            datasets: Dictionary mapping dataset names to their DataLoaders.
            metrics: List of metrics to calculate.
            criterion: Loss function.
            output_dir: Directory to save results and plots.

        Returns:
            Dictionary mapping dataset names to their evaluation metrics.
        """
        all_results = {}

        for dataset_name, data_loader in datasets.items():
            logger.info(f"Evaluating on dataset: {dataset_name}")

            # Evaluate on this dataset
            results = self.evaluate_dataset(
                data_loader=data_loader,
                dataset_name=dataset_name,
                metrics=metrics,
                criterion=criterion,
                output_dir=output_dir,
            )

            all_results[dataset_name] = results

        # Generate comparison visualization across datasets
        if output_dir:
            self._visualize_dataset_comparison(all_results, metrics, output_dir)

        return all_results

    def cross_dataset_evaluation(
        self,
        train_datasets: Dict[str, DataLoader],
        test_datasets: Dict[str, DataLoader],
        metrics: Optional[List[str]] = None,
        criterion: Optional[Callable] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Perform cross-dataset evaluation to assess transfer performance.

        Args:
            train_datasets: Dictionary mapping train dataset names to their loaders.
            test_datasets: Dictionary mapping test dataset names to their loaders.
            metrics: List of metrics to calculate.
            criterion: Loss function.
            output_dir: Directory to save results and plots.

        Returns:
            Nested dictionary: train_dataset -> test_dataset -> metrics
        """
        # This method assumes the model has a method to adapt to different datasets
        # If your model doesn't have such a method, you'll need to add dataset
        # adaptation logic here

        cross_results = {}

        for train_name, train_loader in train_datasets.items():
            cross_results[train_name] = {}

            for test_name, test_loader in test_datasets.items():
                logger.info(f"Cross-evaluation: {train_name} → {test_name}")

                # Create specific output directory
                if output_dir:
                    cross_output_dir = os.path.join(
                        output_dir, f"{train_name}_to_{test_name}"
                    )
                    os.makedirs(cross_output_dir, exist_ok=True)
                else:
                    cross_output_dir = None

                # Evaluate performance
                results = self.evaluator.evaluate(
                    data_loader=test_loader,
                    metrics=metrics,
                    criterion=criterion,
                    output_dir=cross_output_dir,
                    prefix=f"{train_name}_to_{test_name}_",
                )

                cross_results[train_name][test_name] = results

        # Generate heatmap visualization for cross-dataset performance
        if output_dir:
            self._visualize_cross_dataset_performance(
                cross_results, metrics, output_dir
            )

        return cross_results

    def _visualize_dataset_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        output_dir: str,
    ) -> None:
        """Generate comparative visualization across datasets."""
        # Create one plot per metric
        for metric in metrics:
            if not all(
                metric in dataset_results for dataset_results in results.values()
            ):
                continue

            plt.figure(figsize=(10, 6))

            # Extract values for this metric across datasets
            datasets = list(results.keys())
            values = [results[dataset][metric] for dataset in datasets]

            # Create bar chart
            plt.bar(datasets, values)
            plt.xlabel("Dataset")
            plt.ylabel(metric.upper())
            plt.title(f"{metric.upper()} Across Datasets")

            # Add value labels on top of bars
            for i, v in enumerate(values):
                plt.text(i, v + 0.01, f"{v:.4f}", ha="center")

            plt.grid(True, axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()

            # Save figure
            plt.savefig(
                os.path.join(output_dir, f"dataset_comparison_{metric}.png"), dpi=300
            )
            plt.close()

    def _visualize_cross_dataset_performance(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        metrics: List[str],
        output_dir: str,
    ) -> None:
        """Generate heatmap visualization for cross-dataset performance."""
        # Create one heatmap per metric
        for metric in metrics:
            # Check if this metric exists in all results
            if not all(
                metric in test_results
                for train_results in results.values()
                for test_results in train_results.values()
            ):
                continue

            # Extract values for this metric
            train_datasets = list(results.keys())
            test_datasets = list(next(iter(results.values())).keys())

            # Create matrix of values
            matrix = np.zeros((len(train_datasets), len(test_datasets)))
            for i, train_dataset in enumerate(train_datasets):
                for j, test_dataset in enumerate(test_datasets):
                    matrix[i, j] = results[train_dataset][test_dataset][metric]

            # Create heatmap
            plt.figure(
                figsize=(8 + len(test_datasets) * 0.5, 6 + len(train_datasets) * 0.5)
            )
            plt.imshow(matrix, cmap="viridis")

            # Add value annotations
            for i in range(len(train_datasets)):
                for j in range(len(test_datasets)):
                    plt.text(
                        j,
                        i,
                        f"{matrix[i, j]:.4f}",
                        ha="center",
                        va="center",
                        color=(
                            "white" if matrix[i, j] < np.max(matrix) * 0.7 else "black"
                        ),
                    )

            # Set labels
            plt.xticks(range(len(test_datasets)), test_datasets, rotation=45)
            plt.yticks(range(len(train_datasets)), train_datasets)
            plt.xlabel("Test Dataset")
            plt.ylabel("Train Dataset")
            plt.title(f"Cross-Dataset {metric.upper()} Performance")
            plt.colorbar(label=metric.upper())

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"cross_dataset_{metric}_heatmap.png"), dpi=300
            )
            plt.close()
