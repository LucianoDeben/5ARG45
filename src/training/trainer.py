# multimodal_drug_response/training/trainer.py

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for multimodal drug response model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        experiment_name: str = None,
        metrics: List[str] = ["mse", "mae", "r2"],
        early_stopping_patience: int = 10,
        grad_clip_value: Optional[float] = None,
        mixed_precision: bool = False,
        checkpoint_interval: int = 1,
        log_interval: int = 10,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer for training
            loss_fn: Loss function
            scheduler: Learning rate scheduler
            device: Device to use for training ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name of experiment (for checkpointing)
            metrics: List of evaluation metrics to compute
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            grad_clip_value: Value for gradient clipping (None for no clipping)
            mixed_precision: Whether to use mixed precision training
            checkpoint_interval: How often to save checkpoints (in epochs)
            log_interval: How often to log training progress (in batches)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.metrics = metrics
        self.early_stopping_patience = early_stopping_patience
        self.grad_clip_value = grad_clip_value
        self.mixed_precision = mixed_precision
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval

        # Create checkpoint directory
        self.experiment_name = experiment_name or time.strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = os.path.join(checkpoint_dir, self.experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize tracking variables
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "train_metrics": {},
            "val_metrics": {},
        }

        # Initialize metrics tracking
        for metric in metrics:
            self.history["train_metrics"][metric] = []
            self.history["val_metrics"][metric] = []

        # Set up mixed precision if requested
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Move model to device
        self.model.to(self.device)

        logger.info(f"Trainer initialized with device: {self.device}")
        logger.info(f"Mixed precision training: {mixed_precision}")

    def train(self, num_epochs: int) -> nn.Module:
        """
        Train the model.

        Args:
            num_epochs: Number of epochs to train for

        Returns:
            Trained model
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_metrics = self._train_epoch(epoch)

            # Validation phase
            val_loss, val_metrics = self._validate_epoch(epoch)

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]["lr"]
                self.history["learning_rate"].append(current_lr)
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - Learning rate: {current_lr:.6f}"
                )

            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            for metric, value in train_metrics.items():
                self.history["train_metrics"][metric].append(value)

            for metric, value in val_metrics.items():
                self.history["val_metrics"][metric].append(value)

            # Check if this is the best model so far
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0

                # Save best model
                self._save_checkpoint(epoch, is_best=True)
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - New best model with val_loss: {val_loss:.6f}"
                )
            else:
                self.epochs_without_improvement += 1

                # Save regular checkpoint if interval reached
                if (epoch + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(epoch, is_best=False)

            # Print epoch summary
            summary = f"Epoch {epoch+1}/{num_epochs} - "
            summary += f"Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}"

            for metric in self.metrics:
                summary += f", Train {metric}: {train_metrics[metric]:.6f}"
                summary += f", Val {metric}: {val_metrics[metric]:.6f}"

            logger.info(summary)

            # Early stopping check
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Save training history
        self._save_history()

        # Plot training curves
        self._plot_training_curves()

        # Load best model
        self._load_best_model()

        logger.info(
            f"Training completed. Best model from epoch {self.best_epoch+1} with val_loss: {self.best_val_loss:.6f}"
        )
        return self.model

    def _train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.train()
        total_loss = 0.0
        all_targets = []
        all_predictions = []

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            transcriptomics = batch["transcriptomics"].to(self.device)
            drug = batch["drug"].to(self.device)
            targets = batch["viability"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass (with mixed precision if enabled)
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    predictions = self.model(transcriptomics, drug)
                    loss = self.loss_fn(predictions, targets)
            else:
                predictions = self.model(transcriptomics, drug)
                loss = self.loss_fn(predictions, targets)

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.grad_clip_value is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_value
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                if self.grad_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_value
                    )

                self.optimizer.step()

            # Update statistics
            total_loss += loss.item()

            # Store predictions and targets for metric computation
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

            # Update progress bar
            if (batch_idx + 1) % self.log_interval == 0 or batch_idx == len(
                self.train_loader
            ) - 1:
                pbar.set_postfix({"loss": total_loss / (batch_idx + 1)})

        # Calculate average loss and metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = self._compute_metrics(all_targets, all_predictions)

        return avg_loss, metrics

    def _validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_predictions = []

        # Progress bar
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                transcriptomics = batch["transcriptomics"].to(self.device)
                drug = batch["drug"].to(self.device)
                targets = batch["viability"].to(self.device)

                # Forward pass
                predictions = self.model(transcriptomics, drug)
                loss = self.loss_fn(predictions, targets)

                # Update statistics
                total_loss += loss.item()

                # Store predictions and targets for metric computation
                all_predictions.extend(predictions.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())

                # Update progress bar
                pbar.set_postfix({"loss": total_loss / (batch_idx + 1)})

        # Calculate average loss and metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self._compute_metrics(all_targets, all_predictions)

        return avg_loss, metrics

    def _compute_metrics(self, targets: List, predictions: List) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            targets: List of target values
            predictions: List of predicted values

        Returns:
            Dictionary of metric values
        """
        targets = np.array(targets).flatten()
        predictions = np.array(predictions).flatten()

        metric_values = {}

        if "mse" in self.metrics:
            metric_values["mse"] = mean_squared_error(targets, predictions)

        if "rmse" in self.metrics:
            metric_values["rmse"] = np.sqrt(mean_squared_error(targets, predictions))

        if "mae" in self.metrics:
            metric_values["mae"] = mean_absolute_error(targets, predictions)

        if "r2" in self.metrics:
            metric_values["r2"] = r2_score(targets, predictions)

        if "pearson" in self.metrics:
            pearson_corr = np.corrcoef(targets, predictions)[0, 1]
            metric_values["pearson"] = (
                pearson_corr if not np.isnan(pearson_corr) else 0.0
            )

        return metric_values

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)

    def _load_best_model(self) -> None:
        """Load the best model."""
        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")

        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            logger.warning(
                "Best model checkpoint not found. Using current model state."
            )

    def _save_history(self) -> None:
        """Save training history to JSON file."""
        history_path = os.path.join(self.checkpoint_dir, "training_history.json")

        # Convert numpy values to Python types for JSON serialization
        history_serializable = {}

        for key, value in self.history.items():
            if isinstance(value, dict):
                history_serializable[key] = {}
                for sub_key, sub_value in value.items():
                    history_serializable[key][sub_key] = [float(v) for v in sub_value]
            else:
                history_serializable[key] = [float(v) for v in value]

        with open(history_path, "w") as f:
            json.dump(history_serializable, f, indent=4)

    def _plot_training_curves(self) -> None:
        """Plot and save training curves."""
        # Create plots directory
        plots_dir = os.path.join(self.checkpoint_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(plots_dir, "loss_curves.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Plot learning rate
        if self.history["learning_rate"]:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history["learning_rate"])
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.grid(True, alpha=0.3)
            plt.savefig(
                os.path.join(plots_dir, "learning_rate.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


class TrainerFactory:
    """Factory for creating trainers."""

    @staticmethod
    def create_trainer(
        config: Dict[str, Any],
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Trainer:
        """
        Create a trainer from configuration.

        Args:
            config: Training configuration
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Configured Trainer instance
        """
        # Configure optimizer
        optimizer_config = config.get("optimizer", {})
        optimizer_type = optimizer_config.get("type", "adam").lower()
        lr = optimizer_config.get("learning_rate", 1e-3)
        weight_decay = optimizer_config.get("weight_decay", 0.0)

        if optimizer_type == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "sgd":
            momentum = optimizer_config.get("momentum", 0.9)
            optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        elif optimizer_type == "adamw":
            optimizer = optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # Configure loss function
        loss_config = config.get("loss", {})
        loss_type = loss_config.get("type", "mse").lower()

        if loss_type == "mse":
            loss_fn = nn.MSELoss()
        elif loss_type == "mae":
            loss_fn = nn.L1Loss()
        elif loss_type == "huber":
            delta = loss_config.get("delta", 1.0)
            loss_fn = nn.SmoothL1Loss(beta=delta)
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

        # Configure scheduler
        scheduler_config = config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", None)
        scheduler = None

        if scheduler_type == "step":
            step_size = scheduler_config.get("step_size", 10)
            gamma = scheduler_config.get("gamma", 0.1)
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == "cosine":
            T_max = scheduler_config.get("T_max", 100)
            eta_min = scheduler_config.get("eta_min", 0)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )
        elif scheduler_type == "plateau":
            patience = scheduler_config.get("patience", 5)
            factor = scheduler_config.get("factor", 0.1)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=factor, patience=patience, verbose=True
            )

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            checkpoint_dir=config.get("checkpoint_dir", "checkpoints"),
            experiment_name=config.get("experiment_name", None),
            metrics=config.get("metrics", ["mse", "mae", "r2"]),
            early_stopping_patience=config.get("early_stopping_patience", 10),
            grad_clip_value=config.get("grad_clip_value", None),
            mixed_precision=config.get("mixed_precision", False),
            checkpoint_interval=config.get("checkpoint_interval", 1),
            log_interval=config.get("log_interval", 10),
        )

        return trainer


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    from src.data.loaders import LINCSCTRPDataProcessor, create_data_loaders
    from src.models.architecture import ModelFactory

    # Configuration
    model_config = {
        "transcriptomics": {
            "input_dim": 12328,
            "hidden_dims": [1024, 512, 256],
            "dropout_rate": 0.3,
        },
        "chemical": {
            "input_type": "fingerprints",
            "hidden_dims": [128, 64],
            "dropout_rate": 0.3,
        },
        "fusion": {
            "fusion_type": "attention",
            "hidden_dims": [256, 128],
            "dropout_rate": 0.3,
        },
        "prediction": {"hidden_dims": [64, 32], "dropout_rate": 0.3},
    }

    training_config = {
        "optimizer": {"type": "adamw", "learning_rate": 1e-3, "weight_decay": 1e-4},
        "loss": {"type": "mse"},
        "scheduler": {"type": "cosine", "T_max": 100, "eta_min": 1e-6},
        "metrics": ["mse", "mae", "r2", "pearson"],
        "early_stopping_patience": 10,
        "grad_clip_value": 1.0,
        "mixed_precision": True,
        "experiment_name": "multimodal_drug_response_v1",
        "checkpoint_dir": "checkpoints",
    }

    # Create model
    model = ModelFactory.create_model(model_config)

    # Load data
    processor = LINCSCTRPDataProcessor(
        lincs_file="../data/processed/LINCS.gctx",
        ctrp_file="../data/raw/CTRP_viability.csv",
    )

    train_dataset, val_dataset, test_dataset = processor.process()
    train_loader, val_loader, _ = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=32
    )

    # Create trainer
    trainer = TrainerFactory.create_trainer(
        training_config, model, train_loader, val_loader
    )

    # Train model
    model = trainer.train(num_epochs=100)

    # # Plot metrics
    # for metric in self.metrics:
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(self.history["train_metrics"][metric], label=f"Train {metric}")
    #     plt.plot(self.history["val_metrics"][metric], label=f"Validation {metric}")
    #     plt.xlabel("Epoch")
    #     plt.ylabel(metric.upper())
    #     plt.title(f"Training and Validation {metric.upper()}")
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.savefig(os.path.join(plots_dir, f"{metric}_curves.png"), dpi=300, bbox_inches="tight")
    #     plt.close()
