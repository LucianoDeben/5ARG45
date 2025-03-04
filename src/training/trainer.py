# training/trainer.py
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.logging import ExperimentLogger
from utils.storage import CheckpointManager

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
    """
    Early stopping callback to halt training when a monitored metric stops improving.
    """

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
            monitor: Metric to monitor
            min_delta: Minimum change to qualify as improvement
            patience: Number of epochs with no improvement after which training will stop
            mode: 'min' or 'max' for metric monitoring direction
            verbose: Whether to print early stopping information
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.verbose = verbose

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Reset early stopping state when training begins."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.should_stop = False

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float]
    ) -> None:
        """Check for early stopping condition at end of epoch."""
        if self.monitor not in logs:
            if self.verbose:
                logger.warning(
                    f"Early stopping metric '{self.monitor}' not found in logs"
                )
            return

        current = logs[self.monitor]

        if self.mode == "min":
            # For minimizing metrics like loss
            if current < self.best_value - self.min_delta:
                self.best_value = current
                self.wait = 0
            else:
                self.wait += 1
        else:
            # For maximizing metrics like accuracy
            if current > self.best_value + self.min_delta:
                self.best_value = current
                self.wait = 0
            else:
                self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.should_stop = True
            trainer.stop_training = True
            if self.verbose:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")


class LearningRateSchedulerCallback(TrainingCallback):
    """
    Callback for learning rate scheduling.
    """

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        monitor: Optional[str] = None,
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Initialize LR scheduler callback.

        Args:
            scheduler: PyTorch learning rate scheduler
            monitor: Optional metric to monitor for schedulers that require a metric value
            mode: 'min' or 'max' for metric monitoring direction
            verbose: Whether to print scheduling information
        """
        self.scheduler = scheduler
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float]
    ) -> None:
        """Update learning rate at end of epoch."""
        if self.monitor is not None:
            if self.monitor not in logs:
                if self.verbose:
                    logger.warning(
                        f"Learning rate scheduler metric '{self.monitor}' not found in logs"
                    )
                return

            # Get metric value for schedulers that require it
            current = logs[self.monitor]
            self.scheduler.step(current)
        else:
            # For schedulers that just need epoch number
            self.scheduler.step()

        if self.verbose:
            lrs = [group["lr"] for group in trainer.optimizer.param_groups]
            if len(lrs) == 1:
                logger.info(f"Learning rate set to: {lrs[0]:.6f}")
            else:
                logger.info(f"Learning rates set to: {lrs}")


class ModelCheckpointCallback(TrainingCallback):
    """
    Callback for model checkpointing during training.
    """

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
        verbose: bool = True,
    ):
        """
        Initialize model checkpoint callback.

        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename pattern
            monitor: Metric to monitor for best checkpoint determination
            mode: 'min' or 'max' for determining best checkpoint
            save_top_k: Number of best checkpoints to keep
            save_last: Whether to always save the last checkpoint
            save_weights_only: Whether to save only model weights (not optimizer state)
            period: Save checkpoint every this many epochs
            verbose: Whether to print checkpoint information
        """
        self.checkpoint_handler = CheckpointManager(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            verbose=verbose,
        )
        self.save_weights_only = save_weights_only
        self.period = period

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float]
    ) -> None:
        """Save checkpoint at end of epoch if conditions are met."""
        if (epoch + 1) % self.period != 0:
            return

        # Prepare additional data
        additional_data = {
            "epoch": epoch,
            "model_config": trainer.model_config,
            "train_config": trainer.train_config,
        }

        # Save checkpoint
        if self.save_weights_only:
            self.checkpoint_handler.save_checkpoint(
                model=trainer.model.state_dict(),
                epoch=epoch,
                metrics=logs,
                additional_data=additional_data,
            )
        else:
            self.checkpoint_handler.save_checkpoint(
                model=trainer.model,
                epoch=epoch,
                metrics=logs,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                additional_data=additional_data,
            )


class Trainer:
    """
    Trainer class for managing model training, validation, and testing.

    Features:
    - PyTorch-based training loop
    - Distributed training support
    - Callback mechanism for extensibility
    - Metrics tracking and logging
    - Mixed precision training
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
        model_config: Optional[Dict] = None,
        train_config: Optional[Dict] = None,
        mixed_precision: bool = False,
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            optimizer: PyTorch optimizer
            scheduler: Optional learning rate scheduler
            criterion: Loss function
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            exp_logger: Optional experiment logger
            callbacks: List of training callbacks
            model_config: Configuration dictionary for model architecture
            train_config: Configuration dictionary for training parameters
            mixed_precision: Whether to use mixed precision training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = scheduler
        self.criterion = criterion or nn.MSELoss()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_logger = exp_logger
        self.callbacks = callbacks or []
        self.model_config = model_config or {}
        self.train_config = train_config or {}
        self.mixed_precision = mixed_precision

        # Move model to device
        self.model.to(self.device)

        # Set up mixed precision training if requested and available
        self.scaler = None
        if self.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()

        # Internal state
        self.stop_training = False
        self.current_epoch = 0
        self.current_batch = 0
        self.history = {
            "train": {},
            "val": {},
        }

        # Log initialization
        if self.exp_logger:
            self.exp_logger.log_model_summary(self.model)

            if self.model_config:
                self.exp_logger.log_config({"model": self.model_config})

            if self.train_config:
                self.exp_logger.log_config({"training": self.train_config})

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch and return metrics."""
        self.model.train()
        epoch_loss = 0.0
        batch_metrics = {}

        # Get batch size for progress calculation
        batch_size = self.train_loader.batch_size or 1
        num_samples = len(self.train_loader.dataset)
        num_batches = len(self.train_loader)

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            self.current_batch = batch_idx

            # Call on_batch_begin callbacks
            for callback in self.callbacks:
                callback.on_batch_begin(self, batch_idx)

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

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision if enabled
            if self.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                # Backward and optimize with scaled gradients
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward and optimize
                loss.backward()
                self.optimizer.step()

            # Update metrics
            batch_loss = loss.item()
            epoch_loss += batch_loss * targets.size(0)  # Weight by batch size

            # Calculate batch metrics
            batch_metrics = {
                "loss": batch_loss,
                "progress": (batch_idx + 1) / num_batches * 100,
            }

            # Call on_batch_end callbacks
            for callback in self.callbacks:
                callback.on_batch_end(self, batch_idx, batch_metrics)

            # Log batch metrics
            if self.exp_logger and (batch_idx + 1) % 10 == 0:  # Log every 10 batches
                self.exp_logger.log_metrics(
                    batch_metrics,
                    step=self.current_epoch * num_batches + batch_idx,
                    phase="train_batch",
                )

        # Calculate epoch metrics
        train_loss = epoch_loss / num_samples
        epoch_metrics = {"loss": train_loss}

        # Add training time
        epoch_time = time.time() - start_time
        epoch_metrics["time"] = epoch_time

        # Add metrics to history
        for metric_name, metric_value in epoch_metrics.items():
            if metric_name not in self.history["train"]:
                self.history["train"][metric_name] = []
            self.history["train"][metric_name].append(metric_value)

        return epoch_metrics

    def _validate_epoch(self) -> Dict[str, float]:
        """Validate model on validation set and return metrics."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_loss = 0.0
        num_samples = len(self.val_loader.dataset)

        # For calculating metrics
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for batch in self.val_loader:
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
                loss = self.criterion(outputs, targets)

                # Update metrics
                val_loss += loss.item() * targets.size(0)  # Weight by batch size

                # Store outputs and targets for additional metrics
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        # Calculate epoch metrics
        val_loss = val_loss / num_samples
        epoch_metrics = {"loss": val_loss}

        # Concatenate all outputs and targets
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)

        # Calculate additional metrics like R2 and Pearson correlation
        try:
            from scipy.stats import pearsonr
            from sklearn.metrics import r2_score

            r2 = r2_score(all_targets, all_outputs)
            pearson_corr, _ = pearsonr(all_targets.flatten(), all_outputs.flatten())

            epoch_metrics["r2"] = r2
            epoch_metrics["pearson"] = pearson_corr
        except ImportError:
            logger.warning("scikit-learn or scipy not available for additional metrics")

        # Add metrics to history
        for metric_name, metric_value in epoch_metrics.items():
            if metric_name not in self.history["val"]:
                self.history["val"][metric_name] = []
            self.history["val"][metric_name].append(metric_value)

        return epoch_metrics

    def fit(
        self,
        epochs: int,
        validation_freq: int = 1,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Train the model for a fixed number of epochs.

        Args:
            epochs: Number of epochs to train
            validation_freq: Validate every this many epochs
            verbose: Verbosity mode (0: silent, 1: progress bar, 2: one line per epoch)

        Returns:
            Training history dictionary
        """
        # Call on_train_begin callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)

        # Main training loop
        for epoch in range(epochs):
            if self.stop_training:
                break

            self.current_epoch = epoch

            # Call on_epoch_begin callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)

            # Train for one epoch
            train_metrics = self._train_epoch()

            # Add 'train_' prefix to train metrics
            train_logs = {f"train_{k}": v for k, v in train_metrics.items()}

            # Validate if it's time
            val_metrics = {}
            if self.val_loader is not None and (epoch + 1) % validation_freq == 0:
                val_metrics = self._validate_epoch()
                # Add 'val_' prefix to validation metrics
                val_logs = {f"val_{k}": v for k, v in val_metrics.items()}
                train_logs.update(val_logs)

            # Log metrics
            if self.exp_logger:
                # Log train metrics
                self.exp_logger.log_metrics(train_metrics, step=epoch, phase="train")

                # Log validation metrics
                if val_metrics:
                    self.exp_logger.log_metrics(val_metrics, step=epoch, phase="val")

            # Print progress
            if verbose > 0:
                metrics_str = f"Epoch {epoch+1}/{epochs}"
                for name, value in train_logs.items():
                    metrics_str += f" - {name}: {value:.4f}"
                logger.info(metrics_str)

            # Call on_epoch_end callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, train_logs)

        # Call on_train_end callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)

        return self.history

    def evaluate(
        self,
        test_loader: Optional[DataLoader] = None,
        verbose: int = 1,
    ) -> Dict[str, float]:
        """
        Evaluate the model on the test dataset.

        Args:
            test_loader: DataLoader for test data (uses val_loader if None)
            verbose: Verbosity mode

        Returns:
            Dictionary of metrics
        """
        if test_loader is None:
            if self.val_loader is None:
                raise ValueError("No test_loader or val_loader provided for evaluation")
            test_loader = self.val_loader

        self.model.eval()
        test_loss = 0.0
        num_samples = len(test_loader.dataset)

        # For calculating metrics
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for batch in test_loader:
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
                loss = self.criterion(outputs, targets)

                # Update metrics
                test_loss += loss.item() * targets.size(0)  # Weight by batch size

                # Store outputs and targets for additional metrics
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        # Calculate metrics
        test_loss = test_loss / num_samples
        test_metrics = {"loss": test_loss}

        # Concatenate all outputs and targets
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)

        # Calculate additional metrics
        try:
            from scipy.stats import pearsonr
            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )

            r2 = r2_score(all_targets, all_outputs)
            rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
            mae = mean_absolute_error(all_targets, all_outputs)
            pearson_corr, _ = pearsonr(all_targets.flatten(), all_outputs.flatten())

            test_metrics["r2"] = r2
            test_metrics["rmse"] = rmse
            test_metrics["mae"] = mae
            test_metrics["pearson"] = pearson_corr
        except ImportError:
            logger.warning("scikit-learn or scipy not available for additional metrics")

        # Log metrics
        if self.exp_logger:
            self.exp_logger.log_metrics(test_metrics, step=0, phase="test")

        # Print results
        if verbose > 0:
            metrics_str = "Evaluation results:"
            for name, value in test_metrics.items():
                metrics_str += f" - {name}: {value:.4f}"
            logger.info(metrics_str)

        return test_metrics

    def predict(
        self,
        data_loader: DataLoader,
        return_targets: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions for the input samples.

        Args:
            data_loader: DataLoader for input data
            return_targets: Whether to return targets along with predictions

        Returns:
            Numpy array of predictions, or tuple of (predictions, targets)
        """
        self.model.eval()
        all_outputs = []
        all_targets = [] if return_targets else None

        with torch.no_grad():
            for batch in data_loader:
                # Get batch data
                if isinstance(batch, dict):
                    # For MultimodalDataset returning dictionary
                    inputs = batch
                    if return_targets:
                        targets = batch.get("viability")
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # For standard (X, y) tuple
                    inputs, targets = batch
                else:
                    inputs = batch
                    targets = None

                # Move data to device
                if isinstance(inputs, dict):
                    inputs = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }
                else:
                    inputs = inputs.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Store outputs
                all_outputs.append(outputs.cpu().numpy())

                # Store targets if requested
                if return_targets and targets is not None:
                    all_targets.append(targets.cpu().numpy())

        # Concatenate predictions
        predictions = np.concatenate(all_outputs)

        if return_targets:
            targets = np.concatenate(all_targets)
            return predictions, targets
        else:
            return predictions

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
        """
        Create a trainer from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model instance (architecture must match saved weights)
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            map_location: Device to map model to
            **kwargs: Additional arguments for Trainer initialization

        Returns:
            Initialized Trainer instance with loaded weights
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Load model weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        # Create optimizer
        optimizer = kwargs.get("optimizer", None)
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Load optimizer state if available
        if "optimizer_state_dict" in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Create scheduler
        scheduler = kwargs.get("scheduler", None)

        # Load scheduler state if available
        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Get model and training config from checkpoint
        model_config = checkpoint.get("additional_data", {}).get("model_config", {})
        train_config = checkpoint.get("additional_data", {}).get("train_config", {})

        # Create trainer
        trainer = cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            train_config=train_config,
            **kwargs,
        )

        # Set current epoch
        if "epoch" in checkpoint:
            trainer.current_epoch = checkpoint["epoch"] + 1

        return trainer
