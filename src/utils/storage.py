"""Storage utilities for model checkpoints and data management."""

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages data caching with memory and disk storage."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_memory_size: float = 2.0,  # GB
        max_disk_size: float = 20.0,  # GB
    ):
        """Initialize cache manager."""
        self.cache_dir = Path(cache_dir)
        self.max_memory_size = max_memory_size * (1024**3)  # Convert to bytes
        self.max_disk_size = max_disk_size * (1024**3)

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize caches
        self.memory_cache = {}  # key -> (data, size)
        self.disk_cache = {}  # key -> (path, size)

        logger.info(
            f"Initialized CacheManager with {max_memory_size}GB memory limit "
            f"and {max_disk_size}GB disk limit"
        )

    def _get_size(self, obj: Any) -> int:
        """Get approximate size of object in bytes."""
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def put(self, key: str, data: Any, to_disk: bool = False) -> None:
        """Store data in cache."""
        if to_disk:
            self._put_disk(key, data)
        else:
            self._put_memory(key, data)

    def _put_memory(self, key: str, data: Any) -> None:
        """Store in memory cache."""
        size = self._get_size(data)

        # Check size limit
        while (
            sum(s for _, s in self.memory_cache.values()) + size > self.max_memory_size
            and self.memory_cache
        ):
            # Remove oldest item
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]

        self.memory_cache[key] = (data, size)

    def _put_disk(self, key: str, data: Any) -> None:
        """Store in disk cache."""
        cache_path = self.cache_dir / f"{key}.cache"
        temp_path = cache_path.with_suffix(".tmp")

        try:
            # Save to temp file first
            with open(temp_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Get size
            size = temp_path.stat().st_size

            # Check size limit
            while (
                sum(s for _, s in self.disk_cache.values()) + size > self.max_disk_size
                and self.disk_cache
            ):
                # Remove oldest item
                oldest_key = next(iter(self.disk_cache))
                oldest_path = self.disk_cache[oldest_key][0]
                try:
                    Path(oldest_path).unlink()
                except Exception:
                    pass
                del self.disk_cache[oldest_key]

            # Atomic rename
            temp_path.replace(cache_path)
            self.disk_cache[key] = (str(cache_path), size)

        except Exception as e:
            logger.error(f"Error caching to disk: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve data from cache."""
        # Check memory first
        if key in self.memory_cache:
            return self.memory_cache[key][0]

        # Then check disk
        if key in self.disk_cache:
            try:
                with open(self.disk_cache[key][0], "rb") as f:
                    data = pickle.load(f)
                # Move to memory if possible
                self._put_memory(key, data)
                return data
            except Exception as e:
                logger.warning(f"Failed to load from disk cache: {e}")
                # Clean up failed cache
                self.remove(key)

        return default

    def remove(self, key: str) -> None:
        """Remove item from cache."""
        # Remove from memory
        self.memory_cache.pop(key, None)

        # Remove from disk
        if key in self.disk_cache:
            try:
                Path(self.disk_cache[key][0]).unlink()
            except Exception:
                pass
            del self.disk_cache[key]

    def clear(self) -> None:
        """Clear all caches."""
        # Clear memory
        self.memory_cache.clear()

        # Clear disk
        for path, _ in self.disk_cache.values():
            try:
                Path(path).unlink()
            except Exception:
                pass
        self.disk_cache.clear()

    def close(self) -> None:
        """Clean up resources."""
        self.clear()


class DatasetStorage:
    """Manages dataset storage and versioning."""

    def __init__(self, base_dir: Union[str, Path], compress: bool = True):
        """Initialize dataset storage."""
        self.base_dir = Path(base_dir)
        self.compress = compress

        # Create directories
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.cached_dir = self.base_dir / "cached"

        for directory in [self.raw_dir, self.processed_dir, self.cached_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def save_processed(
        self, data: Any, name: str, version: str = "v1", metadata: Optional[Dict] = None
    ) -> Path:
        """Save processed data safely using atomic operations."""
        # Prepare version directory
        version_dir = self.processed_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Prepare metadata
        meta = {
            "name": name,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        }

        # Save metadata
        meta_path = version_dir / f"{name}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        # Save data
        data_path = version_dir / f"{name}.pkl"
        temp_path = data_path.with_suffix(".tmp")

        try:
            # Save to temporary file first
            with open(temp_path, "wb") as f:
                if self.compress:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Atomic rename
            temp_path.replace(data_path)

            return data_path

        except Exception as e:
            logger.error(f"Error saving data: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise

    def load_processed(self, name: str, version: str = "v1") -> Tuple[Any, Dict]:
        """Load processed data."""
        version_dir = self.processed_dir / version
        data_path = version_dir / f"{name}.pkl"
        meta_path = version_dir / f"{name}.meta.json"

        if not data_path.exists():
            raise FileNotFoundError(f"Data not found: {data_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        # Load metadata
        metadata = json.loads(meta_path.read_text())

        # Load data
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        return data, metadata

    def cache_data(self, data: Any, name: str, metadata: Optional[Dict] = None) -> Path:
        """Cache data for faster access."""
        cache_path = self.cached_dir / f"{name}.cache"
        temp_path = cache_path.with_suffix(".tmp")

        try:
            # Save to temporary file first
            with open(temp_path, "wb") as f:
                pickle.dump(
                    {
                        "data": data,
                        "metadata": metadata or {},
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            # Atomic rename
            temp_path.replace(cache_path)

            return cache_path

        except Exception as e:
            logger.error(f"Error caching data: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise

    def load_cached(self, name: str) -> Tuple[Any, Dict]:
        """Load cached data."""
        cache_path = self.cached_dir / f"{name}.cache"

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")

        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

        return cache["data"], cache["metadata"]

    def clear_cache(self, name: Optional[str] = None) -> None:
        """Clear cache files."""
        if name:
            cache_path = self.cached_dir / f"{name}.cache"
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache {name}: {e}")
        else:
            # Clear all caches
            for cache_file in self.cached_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache {cache_file}: {e}")

    def is_cached(self, name: str) -> bool:
        """Check if data is cached."""
        return (self.cached_dir / f"{name}.cache").exists()


class CheckpointManager:
    """Manages model checkpointing with support for saving best models."""

    def __init__(
        self,
        dirpath: Union[str, Path],
        filename: str = "model_{epoch:02d}_{val_loss:.4f}",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        save_last: bool = True,
    ):
        """Initialize checkpoint manager."""
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last

        # Create checkpoint directory
        self.dirpath.mkdir(parents=True, exist_ok=True)

        # Initialize state
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_path = None
        self.checkpoints = []

        logger.info(f"Initialized CheckpointManager in {self.dirpath}")

    def _is_better(self, current: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == "min":
            return current < self.best_score
        return current > self.best_score

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if we have too many."""
        while len(self.checkpoints) > self.save_top_k:
            checkpoint_path = self.checkpoints.pop(0)  # Remove oldest
            if checkpoint_path != self.best_path:  # Don't remove best checkpoint
                try:
                    Path(checkpoint_path).unlink()
                except Exception as e:
                    logger.warning(
                        f"Failed to remove checkpoint {checkpoint_path}: {e}"
                    )

    def save(
        self,
        model: Union[torch.nn.Module, Dict],
        epoch: int,
        metrics: Dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> str:
        """Save model checkpoint."""
        # Format filename
        filename = self.filename.format(epoch=epoch, **metrics)
        checkpoint_path = self.dirpath / filename
        temp_path = checkpoint_path.with_suffix(".tmp")

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

        try:
            # Save to temp file first
            torch.save(checkpoint, temp_path)

            # Move to final location
            temp_path.replace(checkpoint_path)

            # Update checkpoints list
            self.checkpoints.append(str(checkpoint_path))

            # Check if this is best checkpoint
            current_score = metrics.get(self.monitor)
            if current_score is not None and self._is_better(current_score):
                self.best_score = current_score
                self.best_path = str(checkpoint_path)

                # Save as best checkpoint
                best_path = self.dirpath / "best.pt"
                torch.save(checkpoint, best_path)

            # Save as last checkpoint if requested
            if self.save_last:
                last_path = self.dirpath / "last.pt"
                torch.save(checkpoint, last_path)

            # Cleanup old checkpoints
            if self.save_top_k > 0:
                self._cleanup_old_checkpoints()

            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise

    @staticmethod
    def load(
        checkpoint_path: Union[str, Path],
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Restore model state
        if model is not None and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler state
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint
