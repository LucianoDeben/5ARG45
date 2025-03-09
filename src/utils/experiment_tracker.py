# utils/experiment_tracker.py
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

import wandb


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


class ExperimentTracker:
    """
    Tracks experiments, metrics, and artifacts.
    Handles TensorBoard and W&B integration, checkpointing, and visualization.
    """

    def __init__(
        self,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None,
        log_dir: str = "logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        log_to_file: bool = False,
        monitor_system: bool = True,
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tracking: bool = True,
    ):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project_name = project_name
        self.run_name = run_name
        self.tracking = tracking
        
        self.logger = logging.getLogger(f"experiment.{experiment_name}")

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.experiment_dir = log_dir / f"{experiment_name}_{self.timestamp}"
        for subdir in ["figures", "config", "checkpoints", "images"]:
            (self.experiment_dir / subdir).mkdir(parents=True, exist_ok=True)

        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.config = config

        # Setup experiment-specific file logging if requested
        if log_to_file:
            log_file = self.experiment_dir / "experiment.log"
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        if self.use_tensorboard:
            self._setup_tensorboard()
        if self.use_wandb:
            self._setup_wandb(wandb_project, wandb_entity)

        self.metrics = {"train": {}, "val": {}, "test": {}}
        self.system_monitor = SystemMonitor() if monitor_system else None

        self.logger.info(f"Initialized experiment tracker for '{experiment_name}'")
        if config:
            self.log_config(config)

    def _setup_tensorboard(self) -> None:
        self.tensorboard_dir = self.experiment_dir / "tensorboard"
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

    #TODO: Refactor this with the main implementation that also allows for multiple runs
    def _setup_wandb(self, project: Optional[str], entity: Optional[str]) -> None:
        """Set up Weights & Biases tracking."""
        wandb_config = self.config or {}
        
        # Check if wandb is already initialized to avoid re-initialization
        if wandb.run is None:
            wandb.init(
                project=project or self.project_name,
                entity=entity,
                name=self.run_name or f"{self.experiment_name}_{self.timestamp}",
                dir=str(self.experiment_dir),
                config=wandb_config,
                settings=wandb.Settings(start_method="thread"),
            )
        else:
            # If already initialized, just update the config
            wandb.config.update(wandb_config)
            self.logger.info("Using existing wandb run")

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

        self.logger.info(
            f"{phase.capitalize()} {name}: {value:.6f} (Step {step})"
        )

        if self.use_tensorboard:
            self.writer.add_scalar(f"{phase}/{name}", value, step)

        if self.use_wandb:
            wandb.log(
                {
                    f"{phase}/{name}": value,
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
    
    def clone_with_run_name(self, run_name: str) -> "ExperimentTracker":
        """
        Create a new ExperimentTracker with the same configuration but a different run name.
        
        Args:
            run_name: The run name for the new tracker
            
        Returns:
            A new ExperimentTracker instance
        """
        return ExperimentTracker(
            experiment_name=f"{self.experiment_name}_{run_name}",
            config=self.config,
            log_dir=str(self.experiment_dir.parent),
            use_tensorboard=self.use_tensorboard,
            use_wandb=self.use_wandb,
            wandb_project=self.project_name,
            log_to_file=True,
            monitor_system=self.system_monitor is not None,
            project_name=self.project_name,
            run_name=run_name,
            tracking=self.tracking
        )

    def close(self) -> None:
        """Close the experiment tracker, cleaning up resources."""
        if hasattr(self, "writer"):
            try:
                self.writer.close()
                self.logger.info("TensorBoard writer closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing TensorBoard writer: {e}")
                
        if self.use_wandb and wandb.run is not None:
            try:
                wandb.finish()
                self.logger.info("wandb run finished successfully")
            except Exception as e:
                self.logger.error(f"Error finishing wandb: {e}")
                
        self.logger.info(f"Experiment '{self.experiment_name}' completed")