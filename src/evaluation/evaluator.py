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
from torch.utils.data import DataLoader, Dataset

from ..config.config_utils import load_config
from ..utils.data_validation import validate_batch
from ..utils.logging import ExperimentLogger
from ..utils.loss import create_criterion
from ..utils.metrics import compute_metrics
from ..utils.visualization import plot_boxplot

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates model performance for multimodal drug response prediction.

    Supports comprehensive performance assessment with metrics, visualizations,
    and cross-validation, configured via a configuration dictionary. Designed
    for multimodal data (e.g., transcriptomics and chemical features).

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
            data_loader: DataLoader with evaluation data.
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

        criterion = criterion or create_criterion(self.config, "evaluation")
        output_dir = output_dir or eval_cfg.get("output_dir", "results/eval")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        all_targets, all_outputs = [], []
        total_loss = 0.0
        num_samples = len(data_loader.dataset)

        with torch.no_grad():
            for batch in data_loader:
                validate_batch(batch)
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
        results.update(compute_metrics(all_targets, all_outputs, metrics))

        if self.exp_logger:
            self.exp_logger.log_metrics(results, step=0, phase=f"{prefix}eval")

        if output_dir:
            self._save_results(results, all_targets, all_outputs, output_dir, prefix)

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
            data_loader: DataLoader with evaluation data.
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
                validate_batch(batch, required_keys=["viability", group_column])
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
            group_results = compute_metrics(group_targets, group_outputs, metrics)
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
    ) -> Dict[str, float]:
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

        splitter = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            if stratify_column
            and hasattr(dataset, "metadata")
            and stratify_column in dataset.metadata.columns
            else KFold(n_splits=n_splits, shuffle=True, random_state=42)
        )
        y = (
            dataset.metadata[stratify_column].values
            if stratify_column and hasattr(dataset, "metadata")
            else None
        )

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(range(len(dataset)), y)
        ):
            logger.info(f"Evaluating fold {fold + 1}/{n_splits}")
            val_dataset = torch.utils.data.Subset(dataset, val_idx)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            fold_results = self.evaluate(
                val_loader,
                metrics=metrics,
                output_dir=output_dir if fold == 0 else None,
                prefix=f"{prefix}fold{fold+1}_",
            )

            for metric, value in fold_results.items():
                results[metric].append(value)

        aggregate_results = {}
        for metric, values in results.items():
            aggregate_results[f"{metric}_mean"] = float(np.mean(values))
            aggregate_results[f"{metric}_std"] = float(np.std(values))
            aggregate_results[f"{metric}_min"] = float(np.min(values))
            aggregate_results[f"{metric}_max"] = float(np.max(values))

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
        metrics_file = os.path.join(output_dir, f"{prefix}metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        predictions_df = pd.DataFrame(
            {"target": targets.flatten(), "prediction": predictions.flatten()}
        )
        predictions_file = os.path.join(output_dir, f"{prefix}predictions.csv")
        predictions_df.to_csv(predictions_file, index=False)

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
        metrics_file = os.path.join(output_dir, f"{prefix}group_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(group_results, f, indent=2)

        predictions_df = pd.DataFrame(predictions)
        predictions_file = os.path.join(output_dir, f"{prefix}group_predictions.csv")
        predictions_df.to_csv(predictions_file, index=False)

        self._plot_group_comparison(group_results, metrics, output_dir, prefix)

    def _save_cv_results(
        self,
        aggregate_results: Dict[str, float],
        cv_results: Dict[str, List[float]],
        output_dir: str,
        prefix: str = "",
    ) -> None:
        """Save cross-validation results and visualizations."""
        metrics_file = os.path.join(output_dir, f"{prefix}cv_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(aggregate_results, f, indent=2)

        plot_boxplot(
            cv_results,
            "Cross-Validation Performance",
            "Value",
            os.path.join(output_dir, f"{prefix}cv_results.png"),
        )

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

        min_val, max_val = min(targets.min(), predictions.min()), max(
            targets.max(), predictions.max()
        )
        plt.plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction"
        )

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
        """Generate and save bar charts comparing performance across groups."""
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            groups = list(group_results.keys())
            values = [group_results[group].get(metric, 0) for group in groups]

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
            models: List of models from different training runs.
            data_loader: DataLoader with evaluation data.
            metrics: List of metrics to calculate.
            criterion: Loss function.
            output_dir: Directory to save results and plots.
            prefix: Prefix for saved files.

        Returns:
            Dictionary with aggregate statistics across runs.
        """
        eval_cfg = self.config.get("evaluation", {})
        metrics = metrics or eval_cfg.get("metrics", ["r2", "rmse", "mae", "pearson"])
        if "all" in metrics:
            metrics = ["r2", "rmse", "mae", "pearson"]

        output_dir = output_dir or eval_cfg.get("output_dir", "results/eval")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        run_metrics = {f"run_{i}": {} for i in range(len(models))}
        all_predictions = []

        for i, model in enumerate(models):
            model.to(self.device).eval()
            run_prefix = f"{prefix}run_{i}_"

            all_targets, all_outputs = [], []
            with torch.no_grad():
                for batch in data_loader:
                    validate_batch(batch)
                    inputs = {
                        k: v.to(self.device)
                        for k, v in batch.items()
                        if k != "viability"
                    }
                    targets = batch["viability"].to(self.device)

                    outputs = model(inputs)
                    all_targets.append(targets.cpu().numpy())
                    all_outputs.append(outputs.cpu().numpy())

            targets = np.concatenate(all_targets)
            outputs = np.concatenate(all_outputs)

            for j in range(len(targets)):
                all_predictions.append(
                    {
                        "run": i,
                        "target": targets[j].item(),
                        "prediction": outputs[j].item(),
                    }
                )

            run_metrics[f"run_{i}"] = compute_metrics(targets, outputs, metrics)

            if i == 0 and output_dir:
                self._save_results(
                    run_metrics[f"run_{i}"], targets, outputs, output_dir, run_prefix
                )

        aggregate_metrics = {}
        metrics_data = defaultdict(list)
        for run_id, metrics_dict in run_metrics.items():
            for metric, value in metrics_dict.items():
                metrics_data[metric].append(value)

        for metric, values in metrics_data.items():
            aggregate_metrics[f"{metric}_mean"] = float(np.mean(values))
            aggregate_metrics[f"{metric}_std"] = float(np.std(values))
            aggregate_metrics[f"{metric}_min"] = float(np.min(values))
            aggregate_metrics[f"{metric}_max"] = float(np.max(values))

        if output_dir:
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

            predictions_df = pd.DataFrame(all_predictions)
            predictions_file = os.path.join(
                output_dir, f"{prefix}multi_run_predictions.csv"
            )
            predictions_df.to_csv(predictions_file, index=False)

            plot_boxplot(
                metrics_data,
                f"Metrics Across {len(models)} Runs",
                "Value",
                os.path.join(output_dir, f"{prefix}multi_run_metrics.png"),
            )

        if self.exp_logger:
            self.exp_logger.log_metrics(
                aggregate_metrics, step=0, phase=f"{prefix}multi_run"
            )

        return {"runs": run_metrics, "aggregate": aggregate_metrics}


class MultiDatasetEvaluator:
    """
    Evaluates model performance across multiple datasets with standardized metrics.
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

        self.model.to(self.device).eval()
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
        dataset_output_dir = (
            os.path.join(output_dir, dataset_name) if output_dir else None
        )
        if dataset_output_dir:
            os.makedirs(dataset_output_dir, exist_ok=True)

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
            all_results[dataset_name] = self.evaluate_dataset(
                data_loader, dataset_name, metrics, criterion, output_dir
            )

        if output_dir:
            self._visualize_dataset_comparison(all_results, metrics or [], output_dir)

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
        cross_results = {}
        for train_name, train_loader in train_datasets.items():
            cross_results[train_name] = {}
            for test_name, test_loader in test_datasets.items():
                logger.info(f"Cross-evaluation: {train_name} → {test_name}")
                cross_output_dir = (
                    os.path.join(output_dir, f"{train_name}_to_{test_name}")
                    if output_dir
                    else None
                )
                if cross_output_dir:
                    os.makedirs(cross_output_dir, exist_ok=True)

                results = self.evaluator.evaluate(
                    data_loader=test_loader,
                    metrics=metrics,
                    criterion=criterion,
                    output_dir=cross_output_dir,
                    prefix=f"{train_name}_to_{test_name}_",
                )
                cross_results[train_name][test_name] = results

        if output_dir:
            self._visualize_cross_dataset_performance(
                cross_results, metrics or [], output_dir
            )

        return cross_results

    def _visualize_dataset_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        output_dir: str,
    ) -> None:
        """Generate comparative visualization across datasets."""
        for metric in metrics:
            if not all(
                metric in dataset_results for dataset_results in results.values()
            ):
                continue

            plt.figure(figsize=(10, 6))
            datasets = list(results.keys())
            values = [results[dataset][metric] for dataset in datasets]

            plt.bar(datasets, values)
            plt.xlabel("Dataset")
            plt.ylabel(metric.upper())
            plt.title(f"{metric.upper()} Across Datasets")
            for i, v in enumerate(values):
                plt.text(i, v + 0.01, f"{v:.4f}", ha="center")
            plt.grid(True, axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
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
        for metric in metrics:
            if not all(
                metric in test_results
                for train_results in results.values()
                for test_results in train_results.values()
            ):
                continue

            train_datasets = list(results.keys())
            test_datasets = list(next(iter(results.values())).keys())
            matrix = np.zeros((len(train_datasets), len(test_datasets)))
            for i, train_dataset in enumerate(train_datasets):
                for j, test_dataset in enumerate(test_datasets):
                    matrix[i, j] = results[train_dataset][test_dataset][metric]

            plt.figure(
                figsize=(8 + len(test_datasets) * 0.5, 6 + len(train_datasets) * 0.5)
            )
            plt.imshow(matrix, cmap="viridis")
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
