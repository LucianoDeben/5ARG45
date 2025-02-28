"""Unified logging system supporting multiple backends."""

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

    def __init__(self):
        self.metrics = {}
        self.window_size = 100

    def update(self, name: str, value: float) -> Dict[str, float]:
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append(value)
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name].pop(0)

        return self.compute_stats(name)

    def compute_stats(self, name: str) -> Dict[str, float]:
        values = np.array(self.metrics[name])
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "last": float(values[-1]),
        }


class SystemMonitor:
    """Monitors system resource usage."""

    def __init__(self):
        self.process = psutil.Process()

    def get_stats(self) -> Dict[str, float]:
        return {
            "cpu_percent": self.process.cpu_percent(),
            "memory_percent": self.process.memory_percent(),
            "gpu_memory": self._get_gpu_memory() if torch.cuda.is_available() else 0,
        }

    def _get_gpu_memory(self) -> float:
        return torch.cuda.memory_allocated() / 1024**3  # GB


class ExperimentLogger:
    """Unified logging system supporting multiple backends."""

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        log_to_file: bool = True,
        log_level: int = logging.INFO,
        monitor_system: bool = True,
    ):
        """Initialize experiment logger with specified backends."""
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(log_dir) / f"{experiment_name}_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Set flags for logging backends - IMPORTANT to initialize these first
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        # Initialize loggers
        self._setup_python_logger(log_to_file, log_level)
        if self.use_tensorboard:
            self._setup_tensorboard()
        if self.use_wandb:
            self._setup_wandb(wandb_project, wandb_entity)

        # Initialize metric tracking
        self.metrics = {"train": {}, "val": {}, "test": {}}
        self.metric_aggregator = MetricAggregator()

        # Initialize system monitoring
        self.system_monitor = SystemMonitor() if monitor_system else None

        self.logger.info(f"Initialized experiment logger for '{experiment_name}'")

    def _setup_python_logger(self, log_to_file: bool, log_level: int) -> None:
        """Set up Python's built-in logger."""
        self.logger = logging.getLogger(self.experiment_name)
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
        """Set up Weights & Biases."""
        wandb.init(
            project=project,
            entity=entity,
            name=self.experiment_name,
            dir=str(self.experiment_dir),
        )

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration parameters."""
        self.logger.info("Configuration parameters:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

        # Save config
        config_file = self.experiment_dir / "config.json"
        config_file.write_text(str(config))

        # Log to W&B
        if self.use_wandb:
            wandb.config.update(config)

    def log_model_summary(
        self, model: torch.nn.Module, input_shapes: Optional[Dict[str, tuple]] = None
    ) -> None:
        """Log model summary and graph."""
        self.logger.info(f"Model architecture:\n{model}")

        # Parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        # Log model graph
        if input_shapes and self.use_tensorboard:
            example_inputs = {
                name: torch.randn(shape) for name, shape in input_shapes.items()
            }
            self.writer.add_graph(model, example_inputs)

    def log_metric(
        self, name: str, value: float, step: int, phase: str = "train"
    ) -> None:
        """Log a metric value to all active backends."""
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

    def log_metrics(
        self, metrics_dict: Dict[str, float], step: int, phase: str = "train"
    ) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step, phase)

    def log_figure(
        self, name: str, figure: plt.Figure, step: int, phase: str = "train"
    ) -> None:
        """Log a matplotlib figure."""
        if self.use_tensorboard:
            self.writer.add_figure(f"{phase}/{name}", figure, step)
        if self.use_wandb:
            wandb.log({f"{phase}/{name}": wandb.Image(figure)}, step=step)

        # Save figure
        figure_path = self.experiment_dir / "figures" / f"{phase}_{name}_{step}.png"
        figure_path.parent.mkdir(exist_ok=True)
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")

    def plot_metrics(self, save: bool = True) -> Dict[str, plt.Figure]:
        """Generate and save metric plots."""
        plots = {}
        for phase in self.metrics:
            for metric_name in self.metrics[phase]:
                fig, ax = plt.subplots(figsize=(10, 6))
                steps, values = zip(*self.metrics[phase][metric_name])

                ax.plot(steps, values, marker="o", linestyle="-", label=phase)
                ax.set_title(f"{metric_name} vs Steps ({phase})")
                ax.set_xlabel("Steps")
                ax.set_ylabel(metric_name)
                ax.grid(True)

                if save:
                    self.log_figure(f"{metric_name}_history", fig, max(steps), phase)
                plots[f"{phase}_{metric_name}"] = fig

        return plots

    def close(self) -> None:
        """Clean up resources and generate final outputs."""
        if hasattr(self, "writer"):
            self.writer.close()
        if self.use_wandb:
            wandb.finish()

        # Generate final metric plots
        self.plot_metrics()
        self.logger.info(f"Experiment '{self.experiment_name}' completed")
