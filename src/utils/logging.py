# utils/logging.py
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """
    Unified logging system for experiments that supports multiple backends.

    Features:
    - Console and file logging
    - TensorBoard support
    - Metric tracking and visualization
    - Model architecture logging
    - Hardware monitoring
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        use_tensorboard: bool = True,
        log_to_file: bool = True,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Base directory for logs
            use_tensorboard: Whether to use TensorBoard
            log_to_file: Whether to log to a file
            log_level: Logging level
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            log_dir, f"{experiment_name}_{self.timestamp}"
        )

        # Create log directory
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Set up Python logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(log_level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            log_file = os.path.join(self.experiment_dir, "experiment.log")
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # Set up TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.tensorboard_dir = os.path.join(self.experiment_dir, "tensorboard")
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

        # Metrics storage
        self.metrics = {"train": {}, "val": {}, "test": {}}

        self.logger.info(f"Initialized experiment logger for '{experiment_name}'")

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration parameters."""
        self.logger.info("Configuration parameters:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

        # Save config to file
        import json

        config_file = os.path.join(self.experiment_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    def log_model_summary(self, model: torch.nn.Module) -> None:
        """Log model summary information."""
        self.logger.info(f"Model architecture:\n{model}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

    def log_metric(
        self, name: str, value: float, step: int, phase: str = "train"
    ) -> None:
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Current step (epoch or iteration)
            phase: Training phase ('train', 'val', or 'test')
        """
        if phase not in self.metrics:
            self.metrics[phase] = {}

        if name not in self.metrics[phase]:
            self.metrics[phase][name] = []

        self.metrics[phase][name].append((step, value))

        # Log to console
        self.logger.info(f"{phase.capitalize()} {name}: {value:.6f} (Step {step})")

        # Log to TensorBoard
        if self.use_tensorboard:
            self.writer.add_scalar(f"{phase}/{name}", value, step)

    def log_metrics(
        self, metrics_dict: Dict[str, float], step: int, phase: str = "train"
    ) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics_dict: Dictionary of metric names and values
            step: Current step (epoch or iteration)
            phase: Training phase ('train', 'val', or 'test')
        """
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step, phase)

    def plot_metrics(self, save: bool = True) -> Dict[str, plt.Figure]:
        """
        Plot all tracked metrics.

        Args:
            save: Whether to save plots to disk

        Returns:
            Dictionary of metric plots
        """
        plots = {}

        # Plot each metric
        for phase in self.metrics:
            for metric_name in self.metrics[phase]:
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))

                # Extract x and y values
                steps, values = zip(*self.metrics[phase][metric_name])

                # Plot
                ax.plot(steps, values, marker="o", linestyle="-", label=f"{phase}")

                # Add title and labels
                ax.set_title(f"{metric_name.capitalize()} vs. Steps ({phase})")
                ax.set_xlabel("Steps")
                ax.set_ylabel(metric_name)
                ax.grid(True)

                # Save figure
                if save:
                    plot_path = os.path.join(
                        self.experiment_dir, f"{phase}_{metric_name}.png"
                    )
                    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

                plots[f"{phase}_{metric_name}"] = fig

        return plots

    def close(self) -> None:
        """Close the logger and clean up resources."""
        if self.use_tensorboard:
            self.writer.close()

        # Generate final metric plots
        self.plot_metrics()

        self.logger.info(f"Experiment '{self.experiment_name}' completed")
