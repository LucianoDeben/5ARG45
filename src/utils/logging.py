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
from scipy.stats import pearsonr  # Added import for Pearson correlation
from torch.utils.tensorboard import SummaryWriter

import wandb


class MetricAggregator:
    """Aggregates and computes statistics for metrics."""

    def __init__(self, window_size: int = 100):
        self.metrics = {}
        self.window_size = window_size

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

    def reset(self, name: Optional[str] = None):
        if name and name in self.metrics:
            self.metrics[name] = []
        elif not name:
            self.metrics = {}


class SystemMonitor:
    """Monitors system resource usage."""

    def __init__(self):
        self.process = psutil.Process()

    def get_stats(self) -> Dict[str, float]:
        stats = {
            "cpu_percent": self.process.cpu_percent(),
            "memory_percent": self.process.memory_percent(),
        }
        if torch.cuda.is_available():
            stats["gpu_memory_allocated_gb"] = self._get_gpu_memory_allocated()
            stats["gpu_memory_reserved_gb"] = self._get_gpu_memory_reserved()
            stats["gpu_utilization"] = self._get_gpu_utilization()
        return stats

    def _get_gpu_memory_allocated(self) -> float:
        return torch.cuda.memory_allocated() / 1024**3

    def _get_gpu_memory_reserved(self) -> float:
        return torch.cuda.memory_reserved() / 1024**3

    def _get_gpu_utilization(self) -> float:
        try:
            free_memory = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            return 100 * (1 - free_memory / total_memory)
        except:
            return 0.0


class ExperimentLogger:
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
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tracking: bool = True,
    ):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project_name = project_name
        self.run_name = run_name
        self.tracking = tracking

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.experiment_dir = log_dir / f"{experiment_name}_{self.timestamp}"
        for subdir in ["figures", "config", "checkpoints", "images"]:
            (self.experiment_dir / subdir).mkdir(parents=True, exist_ok=True)

        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.config = config

        self._setup_python_logger(log_to_file, log_level)
        if self.use_tensorboard:
            self._setup_tensorboard()
        if self.use_wandb:
            self._setup_wandb(wandb_project, wandb_entity)

        self.metrics = {"train": {}, "val": {}, "test": {}}
        self.metric_aggregator = MetricAggregator(window_size=metric_window_size)
        self.system_monitor = SystemMonitor() if monitor_system else None

        self.logger.info(f"Initialized experiment logger for '{experiment_name}'")
        if config:
            self.log_config(config)

    def _setup_python_logger(self, log_to_file: bool, log_level: int) -> None:
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(log_level)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        if log_to_file:
            log_file = self.experiment_dir / "experiment.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            self.logger.addHandler(file_handler)

    def _setup_tensorboard(self) -> None:
        self.tensorboard_dir = self.experiment_dir / "tensorboard"
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

    def _setup_wandb(self, project: Optional[str], entity: Optional[str]) -> None:
        wandb_config = self.config or {}
        wandb.init(
            project=project,
            entity=entity,
            name=f"{self.experiment_name}_{self.timestamp}",
            dir=str(self.experiment_dir),
            config=wandb_config,
            settings=wandb.Settings(start_method="thread"),
        )

    def log_config(self, config: Dict[str, Any]) -> None:
        self.logger.info("Configuration parameters:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        import json

        import yaml

        config_dir = self.experiment_dir / "config"
        config_dir.mkdir(exist_ok=True)
        with open(config_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        with open(config_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        if self.use_wandb:
            wandb.config.update(config)

    def log_model_summary(
        self, model: torch.nn.Module, input_shapes: Optional[Dict[str, tuple]] = None
    ) -> None:
        if isinstance(
            model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ):
            model_to_log = model.module
        else:
            model_to_log = model

        self.logger.info(f"Model architecture:\n{model_to_log}")
        total_params = sum(p.numel() for p in model_to_log.parameters())
        trainable_params = sum(
            p.numel() for p in model_to_log.parameters() if p.requires_grad
        )
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        model_info_path = self.experiment_dir / "model_summary.txt"
        with open(model_info_path, "w") as f:
            f.write(f"Model: {model_to_log.__class__.__name__}\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n\n")
            f.write(str(model_to_log))

        if input_shapes and self.use_tensorboard:
            try:
                example_inputs = {
                    name: torch.randn(shape) for name, shape in input_shapes.items()
                }
                self.writer.add_graph(model_to_log, example_inputs)
                self.logger.info("Added model graph to TensorBoard")
            except Exception as e:
                self.logger.warning(f"Failed to add model graph to TensorBoard: {e}")

    def log_metric(
        self, name: str, value: float, step: int, phase: str = "train"
    ) -> None:
        if phase not in self.metrics:
            self.metrics[phase] = {}
        if name not in self.metrics[phase]:
            self.metrics[phase][name] = []
        self.metrics[phase][name].append((step, value))

        stats = self.metric_aggregator.update(f"{phase}/{name}", value)
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

        if self.system_monitor:
            system_stats = self.system_monitor.get_stats()
            if self.use_tensorboard:
                for k, v in system_stats.items():
                    self.writer.add_scalar(f"system/{k}", v, step)
            if self.use_wandb:
                wandb.log({f"system/{k}": v for k, v in system_stats.items()})

        if isinstance(value, (int, float, np.number)):
            try:
                if len(self.metrics[phase][name]) > 1:
                    steps, values = zip(*self.metrics[phase][name])
                    pearson_corr, p_value = pearsonr(steps, values)
                    self.logger.info(
                        f"{phase.capitalize()} Pearson Correlation for {name}: "
                        f"r = {pearson_corr:.4f}, p = {p_value:.4f}"
                    )
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
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step, phase)

    def log_nested_metrics(
        self,
        metrics_dict: Dict[str, Any],
        step: int,
        phase: str = "train",
        parent_key: str = "",
    ) -> None:
        for key, value in metrics_dict.items():
            metric_path = f"{parent_key}/{key}" if parent_key else key
            if isinstance(value, dict):
                self.log_nested_metrics(value, step, phase, metric_path)
            elif isinstance(value, (int, float)):
                self.log_metric(metric_path, float(value), step, phase)
            elif isinstance(value, (np.ndarray, torch.Tensor)) and value.size == 1:
                self.log_metric(metric_path, float(value.item()), step, phase)

    def log_figure(
        self, name: str, figure: plt.Figure, step: int, phase: str = "train"
    ) -> None:
        if self.use_tensorboard:
            self.writer.add_figure(f"{phase}/{name}", figure, step)
        if self.use_wandb:
            wandb.log({f"{phase}/{name}": wandb.Image(figure)}, step=step)
        figure_dir = self.experiment_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        figure_path = figure_dir / f"{phase}_{name}_{step}.png"
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")

    def log_image(
        self, name: str, image: np.ndarray, step: int, phase: str = "train"
    ) -> None:
        if self.use_tensorboard:
            self.writer.add_image(f"{phase}/{name}", image, step, dataformats="HWC")
        if self.use_wandb:
            wandb.log({f"{phase}/{name}": wandb.Image(image)}, step=step)
        try:
            from PIL import Image

            image_dir = self.experiment_dir / "images"
            image_dir.mkdir(exist_ok=True)
            image_path = image_dir / f"{phase}_{name}_{step}.png"
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
        checkpoint_dir = self.experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        filename = f"checkpoint_e{epoch}_s{step}.pt"
        path = checkpoint_dir / filename

        if isinstance(
            model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
            if self.use_wandb:
                wandb.save(str(best_path))

        if keep_n_checkpoints > 0:
            checkpoints = sorted([f for f in checkpoint_dir.glob("checkpoint_e*.pt")])
            for old_checkpoint in checkpoints[:-keep_n_checkpoints]:
                old_checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {old_checkpoint}")

        return str(path)

    def plot_metrics(self, save: bool = True) -> Dict[str, plt.Figure]:
        plots = {}
        try:
            for phase in self.metrics:
                for metric_name in self.metrics[phase]:
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
                    if save:
                        figures_dir = self.experiment_dir / "figures"
                        figures_dir.mkdir(exist_ok=True)
                        filename = f"{phase}_{metric_name}_history_{max(steps)}.png"
                        figure_path = figures_dir / filename
                        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
                        self.logger.info(f"Saved metric plot to {figure_path}")
                    plots[f"{phase}_{metric_name}"] = fig
                    plt.close(fig)
        except Exception as e:
            self.logger.error(f"Error in plot_metrics: {e}")
        return plots

    def plot_comparison(
        self, metric_name: str, save: bool = True
    ) -> Optional[plt.Figure]:
        try:
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

            fig, ax = plt.subplots(figsize=(12, 6))
            max_global_step = 0
            for phase in phases_with_metric:
                steps, values = zip(*self.metrics[phase][metric_name])
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
                figures_dir = self.experiment_dir / "figures"
                figures_dir.mkdir(exist_ok=True)
                filename = f"{metric_name}_comparison_{max_global_step}.png"
                figure_path = figures_dir / filename
                fig.savefig(figure_path, dpi=300, bbox_inches="tight")
                self.logger.info(f"Saved comparison plot to {figure_path}")
            plt.close(fig)
            return fig
        except Exception as e:
            self.logger.error(f"Error in plot_comparison: {e}")
            return None

    def clone_with_run_name(self, run_name: str) -> "ExperimentLogger":
        return ExperimentLogger(
            experiment_name=f"{self.experiment_name}_{run_name}",
            config=self.config,
            log_dir=str(self.experiment_dir.parent),
            use_tensorboard=self.use_tensorboard,
            use_wandb=self.use_wandb,
            wandb_project=self.project_name,
            wandb_entity=None,
            log_to_file=False,
            log_level=self.logger.level,
            monitor_system=self.system_monitor is not None,
            metric_window_size=self.metric_aggregator.window_size,
        )

    def close(self) -> None:
        self.plot_metrics()
        all_metric_names = set()
        for phase in self.metrics:
            all_metric_names.update(self.metrics[phase].keys())
        for metric_name in all_metric_names:
            self.plot_comparison(metric_name)
        if hasattr(self, "writer"):
            self.writer.close()
        if self.use_wandb:
            wandb.finish()
        self.logger.info(f"Experiment '{self.experiment_name}' completed")
