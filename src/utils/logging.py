# utils/logging.py
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

import wandb


class MetricAggregator:
    """Aggregates and computes statistics for metrics."""

    def __init__(self, window_size: int = 100):
        """
        Initialize the metric aggregator.

        Args:
            window_size: Number of recent values to keep for computing statistics
        """
        self.metrics = {}
        self.window_size = window_size

    def update(self, name: str, value: float) -> Dict[str, float]:
        """
        Update a metric and compute its statistics.

        Args:
            name: Metric name
            value: New value to add

        Returns:
            Dictionary of statistics for the metric
        """
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append(value)
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name].pop(0)

        return self.compute_stats(name)

    def compute_stats(self, name: str) -> Dict[str, float]:
        """
        Compute statistics for a metric.

        Args:
            name: Metric name

        Returns:
            Dictionary of statistics (mean, std, min, max, last)
        """
        values = np.array(self.metrics[name])
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "last": float(values[-1]),
        }

    def reset(self, name: Optional[str] = None):
        """
        Reset metrics.

        Args:
            name: If provided, reset only this metric; otherwise, reset all metrics
        """
        if name:
            if name in self.metrics:
                self.metrics[name] = []
        else:
            self.metrics = {}


class SystemMonitor:
    """Monitors system resource usage."""

    def __init__(self):
        """Initialize the system monitor."""
        self.process = psutil.Process()

    def get_stats(self) -> Dict[str, float]:
        """
        Get current system resource usage statistics.

        Returns:
            Dictionary of system metrics
        """
        stats = {
            "cpu_percent": self.process.cpu_percent(),
            "memory_percent": self.process.memory_percent(),
        }

        # GPU metrics if available
        if torch.cuda.is_available():
            stats["gpu_memory_allocated_gb"] = self._get_gpu_memory_allocated()
            stats["gpu_memory_reserved_gb"] = self._get_gpu_memory_reserved()
            stats["gpu_utilization"] = self._get_gpu_utilization()

        return stats

    def _get_gpu_memory_allocated(self) -> float:
        """Get allocated GPU memory in GB."""
        return torch.cuda.memory_allocated() / 1024**3

    def _get_gpu_memory_reserved(self) -> float:
        """Get reserved GPU memory in GB."""
        return torch.cuda.memory_reserved() / 1024**3

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            # This requires nvidia-smi access, might fail in some environments
            free_memory = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            return 100 * (1 - free_memory / total_memory)
        except:
            return 0.0


class ExperimentLogger:
    """Unified logging system supporting multiple backends."""

    def __init__(
        self,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None,
        log_dir: str = "logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        log_to_file: bool = True,
        log_level: int = logging.INFO,
        monitor_system: bool = True,
        metric_window_size: int = 100,
    ):
        # Set up deterministic timestamp to match previous implementation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create experiment directory with robust path handling
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.experiment_dir = log_dir / f"{experiment_name}_{timestamp}"

        # Ensure all necessary subdirectories are created
        for subdir in ["figures", "config", "checkpoints", "images"]:
            (self.experiment_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Rest of the initialization remains the same as previous implementation
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.config = config

        # Initialize loggers
        self._setup_python_logger(log_to_file, log_level)
        if self.use_tensorboard:
            self._setup_tensorboard()
        if self.use_wandb:
            self._setup_wandb(wandb_project, wandb_entity)

        # Initialize metric tracking
        self.metrics = {"train": {}, "val": {}, "test": {}}
        self.metric_aggregator = MetricAggregator(window_size=metric_window_size)

        # Initialize system monitoring
        self.system_monitor = SystemMonitor() if monitor_system else None

        self.logger.info(f"Initialized experiment logger for '{experiment_name}'")

        # Log config if provided
        if config:
            self.log_config(config)

    def _setup_python_logger(self, log_to_file: bool, log_level: int) -> None:
        """
        Set up Python's built-in logger.

        Args:
            log_to_file: Whether to log to a file
            log_level: Logging level
        """
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(log_level)

        # Remove any existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            log_file = self.experiment_dir / "experiment.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def _setup_tensorboard(self) -> None:
        """Set up TensorBoard writer."""
        self.tensorboard_dir = self.experiment_dir / "tensorboard"
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

    def _setup_wandb(self, project: Optional[str], entity: Optional[str]) -> None:
        """
        Set up Weights & Biases.

        Args:
            project: W&B project name
            entity: W&B entity (organization or username)
        """
        wandb_config = {}
        if self.config:
            wandb_config = self.config

        wandb.init(
            project=project,
            entity=entity,
            name=f"{self.experiment_name}_{self.timestamp}",
            dir=str(self.experiment_dir),
            config=wandb_config,
            settings=wandb.Settings(start_method="thread"),
        )

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log configuration parameters.

        Args:
            config: Configuration dictionary
        """
        self.logger.info("Configuration parameters:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

        # Save config as JSON and YAML
        import json

        import yaml

        config_dir = self.experiment_dir / "config"
        config_dir.mkdir(exist_ok=True)

        # JSON
        config_file_json = config_dir / "config.json"
        with open(config_file_json, "w") as f:
            json.dump(config, f, indent=2)

        # YAML
        config_file_yaml = config_dir / "config.yaml"
        with open(config_file_yaml, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Log to W&B
        if self.use_wandb:
            wandb.config.update(config)

    def log_model_summary(
        self, model: torch.nn.Module, input_shapes: Optional[Dict[str, tuple]] = None
    ) -> None:
        """
        Log model summary and graph.

        Args:
            model: PyTorch model
            input_shapes: Dictionary mapping input names to shapes for tracing
        """
        # Get the base model if using DataParallel or DistributedDataParallel
        if isinstance(
            model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ):
            model_to_log = model.module
        else:
            model_to_log = model

        self.logger.info(f"Model architecture:\n{model_to_log}")

        # Parameter counts
        total_params = sum(p.numel() for p in model_to_log.parameters())
        trainable_params = sum(
            p.numel() for p in model_to_log.parameters() if p.requires_grad
        )
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        # Log model details to file
        model_info_path = self.experiment_dir / "model_summary.txt"
        with open(model_info_path, "w") as f:
            f.write(f"Model: {model_to_log.__class__.__name__}\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n\n")
            f.write(str(model_to_log))

        # Log model graph
        if input_shapes and self.use_tensorboard:
            try:
                # Create example inputs
                example_inputs = {}
                for name, shape in input_shapes.items():
                    example_inputs[name] = torch.randn(shape)

                # Create a forward method to handle dictionary inputs
                def forward_wrapper(model, inputs_dict):
                    return model(**inputs_dict)

                self.writer.add_graph(
                    model_to_log, forward_wrapper(model_to_log, example_inputs)
                )
                self.logger.info("Added model graph to TensorBoard")
            except Exception as e:
                self.logger.warning(f"Failed to add model graph to TensorBoard: {e}")

    def log_metric(
        self, name: str, value: float, step: int, phase: str = "train"
    ) -> None:
        """
        Log a metric value to all active backends.

        Args:
            name: Metric name
            value: Metric value
            step: Training step or epoch
            phase: Training phase (train, val, test)
        """
        # Update metrics storage
        if phase not in self.metrics:
            self.metrics[phase] = {}
        if name not in self.metrics[phase]:
            self.metrics[phase][name] = []
        self.metrics[phase][name].append((step, value))

        # Get aggregated stats
        stats = self.metric_aggregator.update(f"{phase}/{name}", value)

        # Log to all backends
        self.logger.info(
            f"{phase.capitalize()} {name}: {value:.6f} "
            f"(Step {step}, Moving avg: {stats['mean']:.6f})"
        )

        if self.use_tensorboard:
            self.writer.add_scalar(f"{phase}/{name}", value, step)
            for stat_name, stat_value in stats.items():
                self.writer.add_scalar(f"{phase}/{name}/{stat_name}", stat_value, step)

        if self.use_wandb:
            wandb.log(
                {
                    f"{phase}/{name}": value,
                    **{f"{phase}/{name}/{k}": v for k, v in stats.items()},
                    "step": step,
                }
            )

        # Log system metrics if enabled
        if self.system_monitor:
            system_stats = self.system_monitor.get_stats()
            if self.use_tensorboard:
                for k, v in system_stats.items():
                    self.writer.add_scalar(f"system/{k}", v, step)
            if self.use_wandb:
                wandb.log({f"system/{k}": v for k, v in system_stats.items()})

        # Calculate and log Pearson correlation if metric is numeric
        if isinstance(value, (int, float, np.number)):
            try:
                # After several data points, calculate Pearson correlation
                if len(self.metrics[phase][name]) > 1:
                    steps, values = zip(*self.metrics[phase][name])

                    # Compute Pearson correlation with step
                    pearson_corr, p_value = stats.pearsonr(steps, values)

                    # Log Pearson correlation
                    self.logger.info(
                        f"{phase.capitalize()} Pearson Correlation for {name}: "
                        f"r = {pearson_corr:.4f}, p = {p_value:.4f}"
                    )

                    # Log to backends if available
                    if self.use_tensorboard:
                        self.writer.add_scalar(
                            f"{phase}/{name}/pearson_correlation", pearson_corr, step
                        )

                    if self.use_wandb:
                        wandb.log(
                            {
                                f"{phase}/{name}/pearson_correlation": pearson_corr,
                                f"{phase}/{name}/pearson_pvalue": p_value,
                            }
                        )
            except Exception as corr_error:
                self.logger.warning(
                    f"Error calculating Pearson correlation: {corr_error}"
                )

    def log_metrics(
        self, metrics_dict: Dict[str, float], step: int, phase: str = "train"
    ) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics_dict: Dictionary of metric names and values
            step: Training step or epoch
            phase: Training phase (train, val, test)
        """
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step, phase)

    def log_nested_metrics(
        self,
        metrics_dict: Dict[str, Any],
        step: int,
        phase: str = "train",
        parent_key: str = "",
    ) -> None:
        """
        Log nested metrics dictionary with support for hierarchical structure.

        Args:
            metrics_dict: Nested dictionary of metrics
            step: Training step or epoch
            phase: Training phase (train, val, test)
            parent_key: Parent key for nested metrics
        """
        for key, value in metrics_dict.items():
            # Build the metric path
            metric_path = f"{parent_key}/{key}" if parent_key else key

            if isinstance(value, dict):
                # Recursively process nested dictionaries
                self.log_nested_metrics(value, step, phase, metric_path)
            elif isinstance(value, (int, float)):
                # Log scalar values
                self.log_metric(metric_path, float(value), step, phase)
            elif isinstance(value, (np.ndarray, torch.Tensor)) and value.size == 1:
                # Handle single-element arrays/tensors
                self.log_metric(metric_path, float(value.item()), step, phase)

    def log_figure(
        self, name: str, figure: plt.Figure, step: int, phase: str = "train"
    ) -> None:
        """
        Log a matplotlib figure.

        Args:
            name: Figure name
            figure: Matplotlib figure
            step: Training step or epoch
            phase: Training phase (train, val, test)
        """
        if self.use_tensorboard:
            self.writer.add_figure(f"{phase}/{name}", figure, step)
        if self.use_wandb:
            wandb.log({f"{phase}/{name}": wandb.Image(figure)}, step=step)

        # Save figure
        figure_dir = self.experiment_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        figure_path = figure_dir / f"{phase}_{name}_{step}.png"
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")

    def log_image(
        self, name: str, image: np.ndarray, step: int, phase: str = "train"
    ) -> None:
        """
        Log an image.

        Args:
            name: Image name
            image: Image as numpy array (HWC format)
            step: Training step or epoch
            phase: Training phase (train, val, test)
        """
        if self.use_tensorboard:
            self.writer.add_image(f"{phase}/{name}", image, step, dataformats="HWC")
        if self.use_wandb:
            wandb.log({f"{phase}/{name}": wandb.Image(image)}, step=step)

        # Save image
        try:
            from PIL import Image

            image_dir = self.experiment_dir / "images"
            image_dir.mkdir(exist_ok=True)
            image_path = image_dir / f"{phase}_{name}_{step}.png"

            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            Image.fromarray(image).save(image_path)
        except Exception as e:
            self.logger.warning(f"Failed to save image: {e}")

    def log_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        scheduler=None,
        is_best: bool = False,
        keep_n_checkpoints: int = 3,
    ) -> str:
        """
        Save model checkpoint and log to tracking backends.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            step: Current step
            metrics: Dictionary of metrics
            scheduler: Optional learning rate scheduler
            is_best: Whether this is the best model so far
            keep_n_checkpoints: Number of checkpoints to keep

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # Generate checkpoint filename
        filename = f"checkpoint_e{epoch}_s{step}.pt"
        path = checkpoint_dir / filename

        # Get base model if using DataParallel
        if isinstance(
            model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        # Create checkpoint
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        # Add scheduler if provided
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Save checkpoint
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

        # If best model, create a copy
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")

            if self.use_wandb:
                wandb.save(str(best_path))

        # Clean up old checkpoints if needed
        if keep_n_checkpoints > 0:
            checkpoints = sorted([f for f in checkpoint_dir.glob("checkpoint_e*.pt")])
            for old_checkpoint in checkpoints[:-keep_n_checkpoints]:
                old_checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {old_checkpoint}")

        return str(path)

    def plot_metrics(self, save: bool = True) -> Dict[str, plt.Figure]:
        """
        Generate and save metric plots with robust error handling.

        Args:
            save: Whether to save the plots

        Returns:
            Dictionary of matplotlib figures
        """
        plots = {}
        try:
            for phase in self.metrics:
                for metric_name in self.metrics[phase]:
                    # Only plot if there are actual measurements
                    phase_metrics = self.metrics[phase][metric_name]
                    if not phase_metrics:
                        continue

                    steps, values = zip(*phase_metrics)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(steps, values, marker="o", linestyle="-", label=phase)
                    ax.set_title(f"{metric_name} vs Steps ({phase})")
                    ax.set_xlabel("Steps")
                    ax.set_ylabel(metric_name)
                    ax.grid(True)

                    # Ensure metrics directory exists before saving
                    figures_dir = self.experiment_dir / "figures"
                    figures_dir.mkdir(parents=True, exist_ok=True)

                    if save:
                        filename = f"{phase}_{metric_name}_history_{max(steps)}.png"
                        figure_path = figures_dir / filename
                        try:
                            fig.savefig(figure_path, dpi=300, bbox_inches="tight")
                            self.logger.info(f"Saved metric plot to {figure_path}")
                        except Exception as save_error:
                            self.logger.warning(f"Failed to save plot: {save_error}")

                    plots[f"{phase}_{metric_name}"] = fig
                    plt.close(fig)  # Close to prevent memory leaks

        except Exception as e:
            self.logger.error(f"Error in plot_metrics: {e}")

        return plots

    def plot_comparison(
        self, metric_name: str, save: bool = True
    ) -> Optional[plt.Figure]:
        """
        Generate a comparison plot for a metric across different phases.

        Args:
            metric_name: Metric name to compare
            save: Whether to save the plot

        Returns:
            Matplotlib figure if successful, None otherwise
        """
        try:
            # Find phases that have the metric
            phases_with_metric = [
                phase
                for phase in self.metrics
                if metric_name in self.metrics[phase]
                and self.metrics[phase][metric_name]
            ]

            if not phases_with_metric:
                self.logger.warning(
                    f"No data found for metric '{metric_name}' in any phase"
                )
                return None

            # Ensure we have valid data for plotting
            fig, ax = plt.subplots(figsize=(12, 6))

            max_global_step = 0
            for phase in phases_with_metric:
                # Safely unpack steps and values
                metric_data = self.metrics[phase][metric_name]
                steps, values = zip(*metric_data)

                # Update global max step
                max_global_step = max(max_global_step, max(steps))

                ax.plot(
                    steps, values, marker="o", linestyle="-", label=phase.capitalize()
                )

            ax.set_title(f"{metric_name} Comparison")
            ax.set_xlabel("Steps")
            ax.set_ylabel(metric_name)
            ax.grid(True)
            ax.legend()

            if save:
                # Ensure figures directory exists
                figures_dir = self.experiment_dir / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)

                # Save the comparison plot
                filename = f"{metric_name}_comparison_{max_global_step}.png"
                figure_path = figures_dir / filename
                try:
                    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
                    self.logger.info(f"Saved comparison plot to {figure_path}")
                except Exception as save_error:
                    self.logger.warning(f"Failed to save comparison plot: {save_error}")

            plt.close(fig)  # Close to prevent memory leaks
            return fig

        except Exception as e:
            self.logger.error(f"Error in plot_comparison: {e}")
            return None

    def close(self) -> None:
        """Clean up resources and generate final outputs."""
        # Generate final metric plots
        self.plot_metrics()

        # Generate comparison plots for common metrics
        all_metric_names = set()
        for phase in self.metrics:
            all_metric_names.update(self.metrics[phase].keys())

        for metric_name in all_metric_names:
            self.plot_comparison(metric_name)

        # Close TensorBoard writer
        if hasattr(self, "writer"):
            self.writer.close()

        # Finish wandb run
        if self.use_wandb:
            wandb.finish()

        self.logger.info(f"Experiment '{self.experiment_name}' completed")
