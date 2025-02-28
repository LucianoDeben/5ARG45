# config/storage.py
import datetime
import json
import logging
import os
import pickle
import shutil
import tempfile
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.serialization import DEFAULT_PROTOCOL

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints with atomic saves and recovery."""

    def __init__(
        self,
        dirpath: Union[str, Path],
        filename: str = "model_{epoch:02d}_{val_loss:.4f}",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        save_last: bool = True,
        save_weights_only: bool = False,
        compression_level: int = 3,
        max_checkpoints_size: float = 10.0,  # GB
        verbose: bool = True,
    ):
        """
        Initialize checkpoint manager.

        Args:
            dirpath: Directory for checkpoints
            filename: Checkpoint filename pattern
            monitor: Metric to monitor
            mode: 'min' or 'max' for metric monitoring
            save_top_k: Number of best checkpoints to keep
            save_last: Whether to save last checkpoint
            save_weights_only: Only save model weights (reduces size)
            compression_level: Compression level (0-9)
            max_checkpoints_size: Maximum total size of checkpoints in GB
            verbose: Print detailed information
        """
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.save_weights_only = save_weights_only
        self.compression_level = compression_level
        self.max_checkpoints_size = max_checkpoints_size * (1024**3)  # Convert to bytes
        self.verbose = verbose

        # Create directory
        self.dirpath.mkdir(parents=True, exist_ok=True)

        # Initialize state
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_path = None
        self.checkpoints = OrderedDict()  # Path -> size
        self.lock = Lock()

        # Load existing checkpoints info
        self._load_checkpoints_info()

    def _load_checkpoints_info(self) -> None:
        """Load information about existing checkpoints."""
        info_file = self.dirpath / "checkpoints_info.json"
        if info_file.exists():
            try:
                info = json.loads(info_file.read_text())
                self.best_value = info["best_value"]
                self.best_path = info.get("best_path")
                self.checkpoints = OrderedDict(info["checkpoints"])
            except Exception as e:
                logger.warning(f"Failed to load checkpoints info: {e}")

    def _save_checkpoints_info(self) -> None:
        """Save checkpoints information to disk."""
        info = {
            "best_value": self.best_value,
            "best_path": self.best_path,
            "checkpoints": list(self.checkpoints.items()),
        }
        info_file = self.dirpath / "checkpoints_info.json"
        temp_file = info_file.with_suffix(".tmp")
        try:
            temp_file.write_text(json.dumps(info, indent=2))
            temp_file.replace(info_file)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def _get_checkpoint_size(self, filepath: Path) -> int:
        """Get size of checkpoint file in bytes."""
        try:
            return filepath.stat().st_size
        except FileNotFoundError:
            return 0

    def _enforce_size_limit(self) -> None:
        """Remove oldest checkpoints if total size exceeds limit."""
        total_size = sum(self.checkpoints.values())
        while total_size > self.max_checkpoints_size and self.checkpoints:
            # Remove oldest checkpoint that's not the best
            for path, size in self.checkpoints.items():
                if path != self.best_path:
                    try:
                        Path(path).unlink()
                        total_size -= size
                        del self.checkpoints[path]
                        if self.verbose:
                            logger.info(f"Removed checkpoint due to size limit: {path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to remove checkpoint {path}: {e}")

    def save(
        self,
        model: Union[torch.nn.Module, Dict],
        epoch: int,
        metrics: Dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save model checkpoint with atomic operation and compression."""
        with self.lock:
            # Format filename
            filename = self.filename.format(
                epoch=epoch,
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            )
            filepath = self.dirpath / f"{filename}.pt"

            # Prepare checkpoint data
            checkpoint = {
                "epoch": epoch,
                "metrics": metrics,
                "model_state_dict": (
                    model.state_dict() if isinstance(model, torch.nn.Module) else model
                ),
            }

            if not self.save_weights_only:
                if optimizer is not None:
                    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
                if scheduler is not None:
                    checkpoint["scheduler_state_dict"] = scheduler.state_dict()
                if additional_data is not None:
                    checkpoint["additional_data"] = additional_data

            # Save checkpoint atomically
            with tempfile.NamedTemporaryFile(dir=str(self.dirpath), delete=False) as tf:
                try:
                    torch.save(
                        checkpoint,
                        tf.name,
                        _use_new_zipfile_serialization=True,
                        pickle_protocol=DEFAULT_PROTOCOL,
                        _compression=self.compression_level,
                    )
                    temp_path = Path(tf.name)
                    temp_path.replace(filepath)
                except Exception as e:
                    if temp_path.exists():
                        temp_path.unlink()
                    raise RuntimeError(f"Failed to save checkpoint: {e}")

            # Update checkpoints tracking
            size = self._get_checkpoint_size(filepath)
            self.checkpoints[str(filepath)] = size

            # Update best checkpoint if needed
            current_value = metrics.get(self.monitor)
            if current_value is not None:
                is_better = (
                    current_value < self.best_value
                    if self.mode == "min"
                    else current_value > self.best_value
                )
                if is_better:
                    self.best_value = current_value
                    self.best_path = str(filepath)
                    best_path = self.dirpath / "best.pt"
                    shutil.copy2(filepath, best_path)
                    if self.verbose:
                        logger.info(
                            f"New best checkpoint: {filepath} "
                            f"with {self.monitor}={current_value:.6f}"
                        )

            # Save last checkpoint if requested
            if self.save_last:
                last_path = self.dirpath / "last.pt"
                shutil.copy2(filepath, last_path)

            # Enforce size limits
            self._enforce_size_limit()

            # Save checkpoint info
            self._save_checkpoints_info()

            return str(filepath)

    @staticmethod
    def load(
        filepath: Union[str, Path],
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None,
        weights_only: bool = False,
    ) -> Dict[str, Any]:
        """Load checkpoint with memory efficiency options."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        try:
            if weights_only:
                # Load only model weights
                checkpoint = torch.load(
                    filepath,
                    map_location="cpu" if map_location is None else map_location,
                    weights_only=True,
                )
            else:
                # Load full checkpoint
                checkpoint = torch.load(
                    filepath,
                    map_location="cpu" if map_location is None else map_location,
                )

            # Load model weights
            if model is not None and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])

            if not weights_only:
                # Load optimizer state
                if optimizer is not None and "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                # Load scheduler state
                if scheduler is not None and "scheduler_state_dict" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            return checkpoint

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")


class CacheManager:
    """Manages data caching with LRU policy and compression."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_memory_size: float = 2.0,  # GB
        max_disk_size: float = 20.0,  # GB
        compression_level: int = 1,
        num_workers: int = 4,
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for disk cache
            max_memory_size: Maximum memory cache size in GB
            max_disk_size: Maximum disk cache size in GB
            compression_level: Compression level (0-9)
            num_workers: Number of background workers
        """
        self.cache_dir = Path(cache_dir)
        self.max_memory_size = max_memory_size * (1024**3)  # Convert to bytes
        self.max_disk_size = max_disk_size * (1024**3)
        self.compression_level = compression_level

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize caches
        self.memory_cache = OrderedDict()  # key -> (data, size)
        self.disk_cache = OrderedDict()  # key -> (path, size)
        self.prefetch_queue = Queue()
        self.lock = Lock()

        # Initialize thread pool for background operations
        self.pool = ThreadPoolExecutor(max_workers=num_workers)

        # Load existing cache info
        self._load_cache_info()

    def _load_cache_info(self) -> None:
        """Load information about existing cache entries."""
        info_file = self.cache_dir / "cache_info.json"
        if info_file.exists():
            try:
                info = json.loads(info_file.read_text())
                self.disk_cache = OrderedDict(info["disk_cache"])
            except Exception as e:
                logger.warning(f"Failed to load cache info: {e}")

    def _save_cache_info(self) -> None:
        """Save cache information to disk."""
        info = {"disk_cache": list(self.disk_cache.items())}
        info_file = self.cache_dir / "cache_info.json"
        temp_file = info_file.with_suffix(".tmp")
        try:
            temp_file.write_text(json.dumps(info, indent=2))
            temp_file.replace(info_file)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def _compress_data(self, data: Any) -> bytes:
        """Compress data using pickle and zlib."""
        pickled = pickle.dumps(data, protocol=DEFAULT_PROTOCOL)
        return zlib.compress(pickled, level=self.compression_level)

    def _decompress_data(self, compressed: bytes) -> Any:
        """Decompress data."""
        pickled = zlib.decompress(compressed)
        return pickle.loads(pickled)

    def _enforce_memory_limit(self) -> None:
        """Remove oldest items from memory cache if size exceeds limit."""
        total_size = sum(size for _, size in self.memory_cache.values())
        while total_size > self.max_memory_size and self.memory_cache:
            _, (_, size) = self.memory_cache.popitem(last=False)
            total_size -= size

    def _enforce_disk_limit(self) -> None:
        """Remove oldest items from disk cache if size exceeds limit."""
        total_size = sum(size for _, size in self.disk_cache.values())
        while total_size > self.max_disk_size and self.disk_cache:
            key, (path, size) = self.disk_cache.popitem(last=False)
            try:
                Path(path).unlink()
                total_size -= size
            except Exception as e:
                logger.warning(f"Failed to remove cache file {path}: {e}")

    def put(self, key: str, data: Any, to_disk: bool = False) -> None:
        """
        Store data in cache.

        Args:
            key: Cache key
            data: Data to cache
            to_disk: Whether to store on disk
        """
        with self.lock:
            if to_disk:
                # Compress and save to disk
                compressed = self._compress_data(data)
                filepath = self.cache_dir / f"{key}.cache"
                temp_path = filepath.with_suffix(".tmp")
                try:
                    temp_path.write_bytes(compressed)
                    temp_path.replace(filepath)
                    size = filepath.stat().st_size
                    self.disk_cache[key] = (str(filepath), size)
                    self._enforce_disk_limit()
                finally:
                    if temp_path.exists():
                        temp_path.unlink()
            else:
                # Store in memory
                size = len(pickle.dumps(data, protocol=DEFAULT_PROTOCOL))
                self.memory_cache[key] = (data, size)
                self._enforce_memory_limit()

            self._save_cache_info()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve data from cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached data or default value
        """
        with self.lock:
            # Check memory cache first
            if key in self.memory_cache:
                data, _ = self.memory_cache.pop(key)
                self.memory_cache[key] = data  # Move to end (LRU)
                return data

            # Check disk cache
            if key in self.disk_cache:
                filepath, _ = self.disk_cache[key]
                try:
                    compressed = Path(filepath).read_bytes()
                    data = self._decompress_data(compressed)
                    return data
                except Exception as e:
                    logger.warning(f"Failed to read cached data for {key}: {e}")
                    self.disk_cache.pop(key)
                    self._save_cache_info()

            return default

    def prefetch(self, keys: List[str]) -> None:
        """
        Prefetch items into memory cache.

        Args:
            keys: List of keys to prefetch
        """

        def _prefetch_item(key: str) -> None:
            if key not in self.memory_cache and key in self.disk_cache:
                try:
                    data = self.get(key)
                    if data is not None:
                        self.put(key, data)
                except Exception as e:
                    logger.warning(f"Failed to prefetch {key}: {e}")

        for key in keys:
            self.pool.submit(_prefetch_item, key)

    def remove(self, key: str) -> None:
        """Remove item from cache."""
        with self.lock:
            # Remove from memory cache
            self.memory_cache.pop(key, None)

            # Remove from disk cache
            if key in self.disk_cache:
                filepath, _ = self.disk_cache.pop(key)
                try:
                    Path(filepath).unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove cache file for {key}: {e}")

            self._save_cache_info()

    def clear(self) -> None:
        """Clear all caches."""
        with self.lock:
            # Clear memory cache
            self.memory_cache.clear()

            # Clear disk cache
            for filepath, _ in self.disk_cache.values():
                try:
                    Path(filepath).unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {filepath}: {e}")

            self.disk_cache.clear()
            self._save_cache_info()

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self.memory_cache or key in self.disk_cache

    def __len__(self) -> int:
        """Get total number of cached items."""
        return len(self.memory_cache) + len(self.disk_cache)

    def close(self) -> None:
        """Clean up resources."""
        self.pool.shutdown()
        self._save_cache_info()


class DatasetStorage:
    """Manages dataset storage and versioning."""

    def __init__(
        self, base_dir: Union[str, Path], cache_manager: Optional[CacheManager] = None
    ):
        """
        Initialize dataset storage.

        Args:
            base_dir: Base directory for datasets
            cache_manager: Optional cache manager to use
        """
        self.base_dir = Path(base_dir)
        self.cache_manager = cache_manager or CacheManager(self.base_dir / "cache")

        # Create directories
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _get_version_dir(self, version: str) -> Path:
        """Get directory for specific version."""
        return self.processed_dir / version

    def save_processed(
        self,
        data: Any,
        name: str,
        version: str = "v1",
        metadata: Optional[Dict] = None,
        compress: bool = True,
    ) -> Path:
        """
        Save processed dataset.

        Args:
            data: Dataset to save
            name: Dataset name
            version: Version string
            metadata: Optional metadata
            compress: Whether to compress data

        Returns:
            Path to saved dataset
        """
        version_dir = self._get_version_dir(version)
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
        data_path = version_dir / f"{name}.data"
        if compress:
            data_path = data_path.with_suffix(".data.gz")
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                try:
                    compressed = self._compress_data(data)
                    Path(tf.name).write_bytes(compressed)
                    Path(tf.name).replace(data_path)
                finally:
                    if Path(tf.name).exists():
                        Path(tf.name).unlink()
        else:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                try:
                    pickle.dump(data, open(tf.name, "wb"), protocol=DEFAULT_PROTOCOL)
                    Path(tf.name).replace(data_path)
                finally:
                    if Path(tf.name).exists():
                        Path(tf.name).unlink()

        return data_path

    def load_processed(
        self, name: str, version: str = "v1", use_cache: bool = True
    ) -> Tuple[Any, Dict]:
        """
        Load processed dataset.

        Args:
            name: Dataset name
            version: Version string
            use_cache: Whether to use cache

        Returns:
            Tuple of (data, metadata)
        """
        version_dir = self._get_version_dir(version)

        # Check cache first
        cache_key = f"{name}_v{version}"
        if use_cache and cache_key in self.cache_manager:
            data = self.cache_manager.get(cache_key)
            meta_path = version_dir / f"{name}.meta.json"
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text())
                return data, metadata

        # Load metadata
        meta_path = version_dir / f"{name}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")
        metadata = json.loads(meta_path.read_text())

        # Load data
        data_path = version_dir / f"{name}.data"
        if not data_path.exists():
            data_path = data_path.with_suffix(".data.gz")
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset not found: {data_path}")

        try:
            if data_path.suffix == ".gz":
                compressed = data_path.read_bytes()
                data = self._decompress_data(compressed)
            else:
                with open(data_path, "rb") as f:
                    data = pickle.load(f)

            # Cache the loaded data
            if use_cache:
                self.cache_manager.put(cache_key, data)

            return data, metadata

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {name}: {e}")

    def list_versions(self, name: str) -> List[str]:
        """List available versions for a dataset."""
        versions = []
        for version_dir in self.processed_dir.iterdir():
            if version_dir.is_dir():
                meta_path = version_dir / f"{name}.meta.json"
                if meta_path.exists():
                    versions.append(version_dir.name)
        return sorted(versions)

    def remove_version(self, name: str, version: str) -> None:
        """Remove a specific version of a dataset."""
        version_dir = self._get_version_dir(version)
        meta_path = version_dir / f"{name}.meta.json"
        data_path = version_dir / f"{name}.data"
        gz_path = data_path.with_suffix(".data.gz")

        # Remove files
        for path in [meta_path, data_path, gz_path]:
            if path.exists():
                path.unlink()

        # Remove from cache
        cache_key = f"{name}_v{version}"
        self.cache_manager.remove(cache_key)
