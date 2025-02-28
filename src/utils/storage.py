# utils/storage.py
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


class ModelCheckpoint:
    """
    Handles model checkpointing operations, including saving, loading,
    and managing best model checkpoints.
    """

    def __init__(
        self,
        dirpath: str,
        filename: str = "model_{epoch:02d}_{val_loss:.4f}",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        save_last: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize the model checkpoint handler.

        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename pattern
            monitor: Metric to monitor for best checkpoint determination
            mode: 'min' or 'max' for determining best checkpoint
            save_top_k: Number of best checkpoints to keep
            save_last: Whether to save the most recent checkpoint
            verbose: Whether to print checkpoint information
        """
        self.dirpath = dirpath
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.verbose = verbose

        # Create checkpoint directory
        os.makedirs(dirpath, exist_ok=True)

        # Initialize state
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_checkpoint_path = None
        self.saved_checkpoints = []

        # Ensure mode is valid
        if mode not in ["min", "max"]:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

    def _format_filename(self, epoch: int, metrics: Dict[str, float]) -> str:
        """Format checkpoint filename with current epoch and metrics."""
        filename = self.filename

        # Replace {epoch} and metric values in filename
        filename = filename.replace("{epoch:02d}", f"{epoch:02d}")

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                pattern = f"{{{key}:.4f}}"
                if pattern in filename:
                    filename = filename.replace(pattern, f"{value:.4f}")

        return filename + ".pt"

    def _is_better(self, current_value: float) -> bool:
        """Check if current value is better than best value."""
        if self.mode == "min":
            return current_value < self.best_value
        return current_value > self.best_value

    def save_checkpoint(
        self,
        model: Union[torch.nn.Module, Dict],
        epoch: int,
        metrics: Dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save model checkpoint.

        Args:
            model: Model to save or state dict
            epoch: Current epoch
            metrics: Dictionary of metrics
            optimizer: Optional optimizer to save
            scheduler: Optional scheduler to save
            additional_data: Additional data to save with checkpoint

        Returns:
            Path to saved checkpoint
        """
        # Format filename
        filename = self._format_filename(epoch, metrics)
        filepath = os.path.join(self.dirpath, filename)

        # Prepare checkpoint
        checkpoint = {
            "epoch": epoch,
            "metrics": metrics,
            "model_state_dict": (
                model.state_dict() if isinstance(model, torch.nn.Module) else model
            ),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if additional_data is not None:
            checkpoint["additional_data"] = additional_data

        # Save checkpoint
        torch.save(checkpoint, filepath)

        # Save as last checkpoint if requested
        if self.save_last:
            last_filepath = os.path.join(self.dirpath, "last.pt")
            torch.save(checkpoint, last_filepath)

        # Check if this is the best checkpoint
        if self.monitor in metrics:
            current_value = metrics[self.monitor]

            if self._is_better(current_value):
                self.best_value = current_value
                self.best_checkpoint_path = filepath

                # Save as best checkpoint
                best_filepath = os.path.join(self.dirpath, "best.pt")
                torch.save(checkpoint, best_filepath)

                if self.verbose:
                    print(
                        f"New best checkpoint: {filepath} with {self.monitor}={current_value:.6f}"
                    )

        # Add to saved checkpoints
        self.saved_checkpoints.append(filepath)

        # Delete oldest checkpoints if we have too many
        if self.save_top_k > 0 and len(self.saved_checkpoints) > self.save_top_k:
            # Sort checkpoints by metric value
            if self.monitor in metrics:
                # We'll need to load the metrics for all checkpoints
                checkpoint_metrics = []
                for path in self.saved_checkpoints:
                    try:
                        ckpt = torch.load(path, map_location="cpu")
                        metric_value = ckpt["metrics"].get(
                            self.monitor,
                            float("inf") if self.mode == "min" else float("-inf"),
                        )
                        checkpoint_metrics.append((path, metric_value))
                    except Exception:
                        # If we can't load a checkpoint, assume it's the worst
                        checkpoint_metrics.append(
                            (
                                path,
                                float("inf") if self.mode == "min" else float("-inf"),
                            )
                        )

                # Sort by metric value (ascending for min, descending for max)
                checkpoint_metrics.sort(
                    key=lambda x: x[1], reverse=(self.mode == "max")
                )

                # Keep top k and remove the rest
                checkpoints_to_keep = [
                    path for path, _ in checkpoint_metrics[: self.save_top_k]
                ]
                checkpoints_to_remove = [
                    path
                    for path in self.saved_checkpoints
                    if path not in checkpoints_to_keep
                ]

                for path in checkpoints_to_remove:
                    if path != self.best_checkpoint_path and os.path.exists(path):
                        os.remove(path)
                        if self.verbose:
                            print(f"Removed checkpoint: {path}")

                self.saved_checkpoints = checkpoints_to_keep

        return filepath

    @staticmethod
    def load_checkpoint(
        filepath: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
            model: Optional model to load weights into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            map_location: Device to map tensors to

        Returns:
            Loaded checkpoint dictionary
        """
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=map_location)

        # Load model weights
        if model is not None and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint


class DataStorage:
    """
    Handles dataset storage, caching, and versioning.
    """

    def __init__(self, base_dir: str = "data"):
        """
        Initialize the data storage.

        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = base_dir

        # Create directories
        self.raw_dir = os.path.join(base_dir, "raw")
        self.processed_dir = os.path.join(base_dir, "processed")
        self.cached_dir = os.path.join(base_dir, "cached")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.cached_dir, exist_ok=True)

    def save_processed_data(self, data: Any, name: str, version: str = "v1") -> str:
        """
        Save processed data.

        Args:
            data: Data to save
            name: Data name
            version: Data version

        Returns:
            Path to saved data
        """
        # Create version directory
        version_dir = os.path.join(self.processed_dir, version)
        os.makedirs(version_dir, exist_ok=True)

        # Save data
        filepath = os.path.join(version_dir, f"{name}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        return filepath

    def load_processed_data(self, name: str, version: str = "v1") -> Any:
        """
        Load processed data.

        Args:
            name: Data name
            version: Data version

        Returns:
            Loaded data
        """
        filepath = os.path.join(self.processed_dir, version, f"{name}.pkl")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Processed data not found: {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        return data

    def cache_data(self, data: Any, name: str, metadata: Optional[Dict] = None) -> str:
        """
        Cache data for faster access.

        Args:
            data: Data to cache
            name: Cache name
            metadata: Optional metadata to store with cached data

        Returns:
            Path to cached data
        """
        # Create cache file
        cache_file = os.path.join(self.cached_dir, f"{name}.cache")

        # Save data and metadata
        with open(cache_file, "wb") as f:
            pickle.dump({"data": data, "metadata": metadata}, f)

        return cache_file

    def load_cached_data(self, name: str) -> Tuple[Any, Dict]:
        """
        Load cached data.

        Args:
            name: Cache name

        Returns:
            Tuple of (data, metadata)
        """
        cache_file = os.path.join(self.cached_dir, f"{name}.cache")

        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cached data not found: {cache_file}")

        with open(cache_file, "rb") as f:
            cache = pickle.load(f)

        return cache["data"], cache["metadata"]

    def is_cached(self, name: str) -> bool:
        """Check if data is cached."""
        cache_file = os.path.join(self.cached_dir, f"{name}.cache")
        return os.path.exists(cache_file)

    def clear_cache(self, name: Optional[str] = None) -> None:
        """
        Clear cache.

        Args:
            name: Specific cache to clear, or None to clear all
        """
        if name is None:
            # Clear all caches
            for filename in os.listdir(self.cached_dir):
                if filename.endswith(".cache"):
                    os.remove(os.path.join(self.cached_dir, filename))
        else:
            # Clear specific cache
            cache_file = os.path.join(self.cached_dir, f"{name}.cache")
            if os.path.exists(cache_file):
                os.remove(cache_file)
