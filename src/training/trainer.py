# training/trainer.py
import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..config.config_utils import load_config
from ..utils.data_validation import validate_batch
from ..utils.logging import ExperimentLogger
from ..utils.loss import create_criterion
from ..utils.metrics import compute_metrics
from ..utils.storage import CheckpointManager

logger = logging.getLogger(__name__)


class TrainingCallback:
    """Base class for training callbacks."""

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called when training begins."""
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called when training ends."""
        pass

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        """Called when epoch begins."""
        pass

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float]
    ) -> None:
        """Called when epoch ends."""
        pass

    def on_batch_begin(self, trainer: "Trainer", batch: int) -> None:
        """Called when batch begins."""
        pass

    def on_batch_end(
        self, trainer: "Trainer", batch: int, logs: Dict[str, float]
    ) -> None:
        """Called when batch ends."""
        pass


class EarlyStopping(TrainingCallback):
    """Early stopping callback to halt training when a monitored metric stops improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 5,
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Initialize early stopping callback.

        Args:
            monitor: Metric to monitor (e.g., 'val_loss', 'val_r2').
            min_delta: Minimum change to qualify as improvement.
            patience: Epochs with no improvement before stopping.
            mode: 'min' (minimize) or 'max' (maximize) for metric direction.
            verbose: Whether to log early stopping events.
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Reset state at training start."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = float("inf") if self.mode == "min" else float("-inf")

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float]
    ) -> None:
        """Check early stopping condition."""
        if self.monitor not in logs:
            if self.verbose:
                logger.warning(f"Early stopping metric '{self.monitor}' not in logs")
            return

        current = logs[self.monitor]
        if self.mode == "min":
            if current < self.best_value - self.min_delta:
                self.best_value = current
                self.wait = 0
            else:
                self.wait += 1
        else:
            if current > self.best_value + self.min_delta:
                self.best_value = current
                self.wait = 0
            else:
                self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            trainer.stop_training = True
            if self.verbose:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")


class LearningRateSchedulerCallback(TrainingCallback):
    """Callback for adjusting learning rate during training."""

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        monitor: Optional[str] = None,
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Initialize learning rate scheduler callback.

        Args:
            scheduler: PyTorch LR scheduler instance.
            monitor: Metric to monitor (e.g., 'val_loss') if scheduler requires it.
            mode: 'min' or 'max' for metric direction.
            verbose: Whether to log LR changes.
        """
        self.scheduler = scheduler
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float]
    ) -> None:
        """Update learning rate at epoch end."""
        if self.monitor:
            if self.monitor not in logs:
                if self.verbose:
                    logger.warning(f"LR scheduler metric '{self.monitor}' not in logs")
                return
            self.scheduler.step(logs[self.monitor])
        else:
            self.scheduler.step()

        if self.verbose:
            lrs = [group["lr"] for group in trainer.optimizer.param_groups]
            logger.info(
                f"Learning rate set to: {lrs[0]:.6f}"
                if len(lrs) == 1
                else f"Learning rates: {lrs}"
            )


class ModelCheckpointCallback(TrainingCallback):
    """Callback for saving model checkpoints during training."""

    def __init__(
        self,
        dirpath: str = "checkpoints",
        filename: str = "model_{epoch:02d}_{val_loss:.4f}",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        save_last: bool = True,
        save_weights_only: bool = False,
        period: int = 1,
        **kwargs,
    ):
        """
        Initialize checkpoint callback.

        Args:
            dirpath: Directory to save checkpoints.
            filename: Filename pattern with placeholders (e.g., '{epoch}', '{val_loss}').
            monitor: Metric to determine best model.
            mode: 'min' or 'max' for metric direction.
            save_top_k: Number of best checkpoints to keep.
            save_last: Whether to save the last epoch’s checkpoint.
            save_weights_only: Save only model weights (excludes optimizer/scheduler).
            period: Save every this many epochs.
        """
        self.checkpoint_handler = CheckpointManager(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
        )
        self.save_weights_only = save_weights_only
        self.period = period

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float]
    ) -> None:
        """Save checkpoint if conditions are met."""
        if (epoch + 1) % self.period != 0:
            return

        additional_data = {"epoch": epoch, "config": trainer.config}
        if self.save_weights_only:
            self.checkpoint_handler.save(
                model=trainer.model.state_dict(),
                epoch=epoch,
                metrics=logs,
                additional_data=additional_data,
            )
        else:
            self.checkpoint_handler.save(
                model=trainer.model,
                epoch=epoch,
                metrics=logs,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                additional_data=additional_data,
            )


class Trainer:
    """
    Manages training, validation, evaluation, and inference for PyTorch models.

    Features:
    - Config-driven training with support for multimodal data.
    - Mixed precision training for efficiency.
    - Extensible callbacks (e.g., early stopping, checkpointing).
    - Comprehensive metrics logging and checkpointing.
    - Robust error handling for batch processing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[Callable] = None,
        device: Optional[str] = None,
        exp_logger: Optional[ExperimentLogger] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        config: Optional[Dict] = None,
        mixed_precision: bool = False,
    ):
        """
        Initialize Trainer with config-driven parameters.

        Args:
            model: PyTorch model instance.
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation data.
            optimizer: Optional optimizer (defaults to config if None).
            scheduler: Optional LR scheduler (defaults to config if None).
            criterion: Optional loss function (defaults to config if None).
            device: Device ('cuda', 'cpu', or None for auto-detection).
            exp_logger: ExperimentLogger instance for tracking.
            callbacks: List of TrainingCallback instances.
            config: Configuration dictionary (loads default if None).
            mixed_precision: Enable mixed precision training if True.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or load_config("config.yaml")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_logger = exp_logger or ExperimentLogger()
        self.callbacks = callbacks or []
        self.mixed_precision = mixed_precision

        self.model.to(self.device)
        self.optimizer = optimizer or self._create_optimizer()
        self.scheduler = scheduler or self._create_scheduler()
        self.criterion = criterion or create_criterion(self.config, "training")

        self.scaler = (
            torch.cuda.amp.GradScaler()
            if self.mixed_precision and torch.cuda.is_available()
            else None
        )
        self.stop_training = False
        self.current_epoch = 0
        self.history = {"train": {}, "val": {}}

        if self.exp_logger:
            self.exp_logger.log_model_summary(self.model)
            self.exp_logger.log_config(self.config)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        train_cfg = self.config.get("training", {})
        opt_type = train_cfg.get("optimizer", "adam").lower()
        lr = train_cfg.get("learning_rate", 0.001)
        wd = train_cfg.get("weight_decay", 1e-5)

        if opt_type == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_type == "sgd":
            return optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create scheduler from config."""
        train_cfg = self.config.get("training", {})
        sch_type = train_cfg.get("scheduler", None)
        if not sch_type:
            return None

        sch_type = sch_type.lower()
        if sch_type == "step":
            step_size = train_cfg.get("step_size", 10)
            gamma = train_cfg.get("gamma", 0.1)
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif sch_type == "reduce_on_plateau":
            patience = train_cfg.get("patience", 5)
            factor = train_cfg.get("factor", 0.1)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=patience, factor=factor
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sch_type}")

    def _train_epoch(self) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        train_loss = 0.0
        num_samples = len(self.train_loader.dataset)
        all_targets, all_outputs = [], []

        for batch_idx, batch in enumerate(self.train_loader):
            validate_batch(batch)  # Assuming this checks batch validity
            inputs = {
                k: v.to(self.device) for k, v in batch.items() if k != "viability"
            }
            targets = batch["viability"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * targets.size(0)
            all_targets.append(targets.detach().cpu().numpy())  # Detach targets
            all_outputs.append(outputs.detach().cpu().numpy())  # Detach outputs

            # Optional: Log batch-level loss separately if desired
            # if self.exp_logger and batch_idx % 10 == 0:  # Log every 10 batches
            #     self.exp_logger.log_metric("batch_loss", loss.item(), step=batch_idx, phase="train_batch")

        train_loss /= num_samples
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        metrics = compute_metrics(all_targets, all_outputs, ["r2", "pearson", "rmse"])
        metrics["loss"] = train_loss

        return metrics

    def _validate_epoch(self) -> Dict[str, float]:
        """Validate model on validation set."""
        if not self.val_loader:
            return {}

        self.model.eval()
        val_loss = 0.0
        num_samples = len(self.val_loader.dataset)
        all_targets, all_outputs = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                validate_batch(batch)
                inputs = {
                    k: v.to(self.device) for k, v in batch.items() if k != "viability"
                }
                targets = batch["viability"].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)

                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        val_loss /= num_samples
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        metrics = compute_metrics(all_targets, all_outputs, ["r2", "pearson", "rmse"])
        metrics["loss"] = val_loss

        return metrics

    def evaluate(
        self,
        test_loader: Optional[DataLoader] = None,
        verbose: int = 1,
        use_evaluator: bool = False,
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        test_loader = test_loader or self.val_loader
        if not test_loader:
            raise ValueError("No test_loader or val_loader provided")

        if use_evaluator:
            from ..evaluation.evaluator import Evaluator

            evaluator = Evaluator(self.model, self.device, self.exp_logger, self.config)
            return evaluator.evaluate(test_loader, prefix="test_")

        self.model.eval()
        test_loss = 0.0
        num_samples = len(test_loader.dataset)
        all_targets, all_outputs = [], []

        with torch.no_grad():
            for batch in test_loader:
                validate_batch(batch)
                inputs = {
                    k: v.to(self.device) for k, v in batch.items() if k != "viability"
                }
                targets = batch["viability"].to(self.device).float()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)

                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        test_loss /= num_samples
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        metrics = compute_metrics(
            all_targets, all_outputs, ["r2", "rmse", "mae", "pearson"]
        )
        metrics["loss"] = test_loss

        if self.exp_logger:
            self.exp_logger.log_metrics(metrics, step=0, phase="test")

        if verbose > 0:
            logger.info(
                "Evaluation: " + " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            )

        return metrics

    def fit(self, epochs: int = 50) -> None:
        """
        Train the model for a specified number of epochs.

        Args:
            epochs: Number of epochs to train
        """
        for epoch in range(epochs):
            # Train for one epoch
            train_metrics = self._train_epoch()

            # Validate for one epoch
            val_metrics = self._validate_epoch()

            # Log metrics with epoch as step
            if self.exp_logger:
                self.exp_logger.log_metrics(train_metrics, step=epoch, phase="train")
                self.exp_logger.log_metrics(val_metrics, step=epoch, phase="val")

            # Combine metrics for logging and callbacks
            logs = {"train": train_metrics, "val": val_metrics}

            # Log epoch summary
            log_str = f"Epoch {epoch + 1}/{epochs}"
            for phase, phase_metrics in logs.items():
                log_str += f" - {phase}_loss: {phase_metrics['loss']:.4f}"
                for metric, value in phase_metrics.items():
                    if metric != "loss":
                        log_str += f" - {phase}_{metric}: {value:.4f}"
            logger.info(log_str)

            # Update scheduler if present
            if self.scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Callbacks
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_epoch_end(self, epoch, logs)

        # Final evaluation on test set if provided
        if hasattr(self, "test_loader") and self.test_loader:
            test_metrics = self.evaluate(self.test_loader, verbose=1)
            if self.exp_logger:
                self.exp_logger.log_metrics(test_metrics, step=epochs, phase="test")

    def predict(
        self, data_loader: DataLoader, return_targets: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate predictions."""
        self.model.eval()
        all_outputs, all_targets = [], [] if return_targets else None

        with torch.no_grad():
            for batch in data_loader:
                validate_batch(batch)
                inputs = {
                    k: v.to(self.device) for k, v in batch.items() if k != "viability"
                }
                outputs = self.model(inputs)
                all_outputs.append(outputs.cpu().numpy())

                if return_targets:
                    all_targets.append(batch["viability"].cpu().numpy())

        predictions = np.concatenate(all_outputs)
        return (
            (predictions, np.concatenate(all_targets))
            if return_targets
            else predictions
        )

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        map_location: Optional[str] = None,
        **kwargs,
    ) -> "Trainer":
        """Load Trainer from a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer = kwargs.get("optimizer") or optim.Adam(model.parameters(), lr=0.001)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler = kwargs.get("scheduler")
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        config = checkpoint.get("additional_data", {}).get("config", {})
        trainer = cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            **kwargs,
        )
        trainer.current_epoch = checkpoint.get("epoch", 0) + 1
        return trainer


class MultiRunTrainer:
    """
    Manages multiple training runs of the same model architecture for statistical analysis.
    """

    def __init__(
        self,
        model_class: type,
        model_kwargs: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        optimizer_class: Optional[type] = None,
        scheduler_class: Optional[type] = None,
        criterion: Optional[Callable] = None,
        device: Optional[str] = None,
        exp_logger: Optional[ExperimentLogger] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        config: Optional[Dict] = None,
        mixed_precision: bool = False,
        num_runs: int = 5,
        save_models: bool = True,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize MultiRunTrainer for statistical performance assessment.

        Args:
            model_class: PyTorch model class to instantiate
            model_kwargs: Keyword arguments to pass to model_class
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            test_loader: Optional DataLoader for testing (final evaluation)
            optimizer_class: Optional optimizer class (defaults to Adam if None)
            scheduler_class: Optional LR scheduler class
            criterion: Optional loss function (defaults to config if None)
            device: Device ('cuda', 'cpu', or None for auto-detection)
            exp_logger: ExperimentLogger instance for tracking
            callbacks: List of TrainingCallback instances
            config: Configuration dictionary (loads default if None)
            mixed_precision: Enable mixed precision training if True
            num_runs: Number of training runs to perform
            save_models: Whether to save all trained models
            output_dir: Directory to save results and models
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer_class = optimizer_class or torch.optim.Adam
        self.scheduler_class = scheduler_class
        self.criterion = criterion
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_logger = exp_logger
        self.callbacks = callbacks or []
        self.config = config or load_config("config.yaml")
        self.mixed_precision = mixed_precision
        self.num_runs = num_runs
        self.save_models = save_models

        self.output_dir = output_dir or self.config.get("paths", {}).get(
            "results_dir", "results"
        )
        self.models_dir = os.path.join(self.output_dir, "models")
        if self.save_models:
            os.makedirs(self.models_dir, exist_ok=True)

        self.run_results = []
        self.trained_models = []
        self.aggregate_results = {}

        logger.info(f"Initialized MultiRunTrainer for {num_runs} runs")

    def _get_run_trainer(self, run_id: int) -> Trainer:
        """Create a Trainer instance for a specific run."""
        model = self.model_class(**self.model_kwargs)

        seed = 42 + run_id
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        optimizer = self.optimizer_class(
            model.parameters(),
            lr=self.config.get("training", {}).get("learning_rate", 0.001),
        )

        scheduler = None
        if self.scheduler_class:
            scheduler = self.scheduler_class(optimizer)

        run_callbacks = self.callbacks.copy()
        if self.save_models:
            run_model_dir = os.path.join(self.models_dir, f"run_{run_id}")
            checkpoint_callback = ModelCheckpointCallback(
                dirpath=run_model_dir, save_top_k=1, verbose=True
            )
            run_callbacks.append(checkpoint_callback)

        run_logger = (
            self.exp_logger.clone_with_run_name(f"run_{run_id}")
            if self.exp_logger and hasattr(self.exp_logger, "clone_with_run_name")
            else self.exp_logger
        )
        if run_logger == self.exp_logger and self.exp_logger:
            logger.warning(
                "ExperimentLogger does not support cloning. Using default logger."
            )

        return Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=self.criterion,
            device=self.device,
            exp_logger=run_logger,
            callbacks=run_callbacks,
            config=self.config,
            mixed_precision=self.mixed_precision,
        )

    def run_training(
        self, epochs: Optional[int] = None, use_evaluator: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Run multiple training instances and collect results.

        Args:
            epochs: Number of epochs per run (defaults to config)
            use_evaluator: Use Evaluator for test set evaluation

        Returns:
            Dictionary with aggregate statistics of metrics across runs
        """
        from ..evaluation.evaluator import Evaluator

        epochs = epochs or self.config.get("training", {}).get("epochs", 50)
        evaluator = (
            Evaluator(
                self.model_class(**self.model_kwargs),
                self.device,
                self.exp_logger,
                self.config,
            )
            if use_evaluator
            else None
        )

        for run_id in range(self.num_runs):
            logger.info(f"Starting training run {run_id+1}/{self.num_runs}")
            trainer = self._get_run_trainer(run_id)
            trainer.fit(epochs=epochs)

            val_metrics = {}
            if self.val_loader:
                val_metrics = trainer.evaluate(self.val_loader, verbose=1)

            test_metrics = {}
            if self.test_loader:
                test_metrics = (
                    evaluator.evaluate(self.test_loader, prefix=f"run_{run_id}_")
                    if use_evaluator
                    else trainer.evaluate(self.test_loader, verbose=1)
                )

            run_result = {
                "run_id": run_id,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
            self.run_results.append(run_result)
            self.trained_models.append(trainer.model)

            logger.info(f"Completed training run {run_id+1}/{self.num_runs}")

        self._compute_aggregate_stats()
        self._save_results()

        return self.aggregate_results

    def _compute_aggregate_stats(self) -> None:
        """Compute statistical measures across runs."""
        val_metrics = defaultdict(list)
        test_metrics = defaultdict(list)

        for result in self.run_results:
            for k, v in result.get("val_metrics", {}).items():
                val_metrics[k].append(v)
            for k, v in result.get("test_metrics", {}).items():
                test_metrics[k].append(v)

        val_stats = {}
        for metric, values in val_metrics.items():
            if values:  # Only compute stats if we have values
                val_stats[f"{metric}_mean"] = float(np.mean(values))
                val_stats[f"{metric}_std"] = float(np.std(values))
                val_stats[f"{metric}_min"] = float(np.min(values))
                val_stats[f"{metric}_max"] = float(np.max(values))

        test_stats = {}
        for metric, values in test_metrics.items():
            if values:  # Only compute stats if we have values
                test_stats[f"{metric}_mean"] = float(np.mean(values))
                test_stats[f"{metric}_std"] = float(np.std(values))
                test_stats[f"{metric}_min"] = float(np.min(values))
                test_stats[f"{metric}_max"] = float(np.max(values))

        self.aggregate_results = {
            "val": val_stats,
            "test": test_stats,
            "num_runs": self.num_runs,
        }

    def _save_results(self) -> None:
        """Save aggregate results and generate visualizations."""
        from ..utils.visualization import plot_boxplot

        if not self.output_dir:
            return

        results_dir = os.path.join(self.output_dir, "multi_run_results")
        os.makedirs(results_dir, exist_ok=True)

        # Save aggregate metrics
        metrics_file = os.path.join(results_dir, "aggregate_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(self.aggregate_results, f, indent=2)

        # Save individual run results
        runs_file = os.path.join(results_dir, "individual_runs.json")
        with open(runs_file, "w") as f:
            json.dump(self.run_results, f, indent=2)

        # Create validation metrics visualization
        try:
            if self.aggregate_results.get("val"):
                # Prepare data for boxplot - gather values by metric
                val_metrics_data = {}
                for run_result in self.run_results:
                    for metric, value in run_result.get("val_metrics", {}).items():
                        if metric not in val_metrics_data:
                            val_metrics_data[metric] = []
                        val_metrics_data[metric].append(value)
                
                # Only plot if we have valid data
                if val_metrics_data:
                    plot_boxplot(
                        val_metrics_data,
                        "Validation Metrics Across Runs",
                        "Value",
                        os.path.join(results_dir, "val_metrics_distribution.png"),
                    )
                    logger.info("Generated validation metrics boxplot")
        except Exception as e:
            logger.error(f"Error creating validation metrics visualization: {e}")

        # Create test metrics visualization
        try:
            if self.aggregate_results.get("test"):
                # Prepare data for boxplot - gather values by metric
                test_metrics_data = {}
                for run_result in self.run_results:
                    for metric, value in run_result.get("test_metrics", {}).items():
                        if metric not in test_metrics_data:
                            test_metrics_data[metric] = []
                        test_metrics_data[metric].append(value)
                
                # Only plot if we have valid data
                if test_metrics_data:
                    plot_boxplot(
                        test_metrics_data,
                        "Test Metrics Across Runs",
                        "Value",
                        os.path.join(results_dir, "test_metrics_distribution.png"),
                    )
                    logger.info("Generated test metrics boxplot")
        except Exception as e:
            logger.error(f"Error creating test metrics visualization: {e}")

        logger.info(f"Saved multi-run results to {results_dir}")
