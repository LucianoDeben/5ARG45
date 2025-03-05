# utils/storage.py
import json
import logging
import os
import pickle
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages data caching with memory and disk storage using LRU eviction policy.

    This class provides a two-tier caching system:
    1. In-memory cache for fast access to frequently used data
    2. Disk cache for larger datasets that don't fit in memory

    Example usage:
    ```python
    cache = CacheManager("./cache", max_memory_size=1.0, max_disk_size=10.0)

    # Store data
    cache.put("dataset_features", features_array)

    # Retrieve data
    features = cache.get("dataset_features")

    # Store data to disk (for larger items)
    cache.put("large_embeddings", embeddings, to_disk=True)

    # Clear all caches when done
    cache.clear()
    cache.close()
    ```
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_memory_size: float = 2.0,  # GB
        max_disk_size: float = 20.0,  # GB
        compression_level: int = 1,
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store disk cache
            max_memory_size: Maximum memory cache size in GB
            max_disk_size: Maximum disk cache size in GB
            compression_level: Compression level for disk cache (0-9, 0=none)
        """
        self.cache_dir = Path(cache_dir)
        self.max_memory_size = max_memory_size * (1024**3)  # Convert to bytes
        self.max_disk_size = max_disk_size * (1024**3)
        self.compression_level = compression_level

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize caches with access timestamps
        self.memory_cache = {}  # key -> (data, size, timestamp)
        self.disk_cache = {}  # key -> (path, size, timestamp)

        # Create index file for disk cache
        self.index_path = self.cache_dir / "index.json"
        self._load_disk_index()

        logger.info(
            f"Initialized CacheManager with {max_memory_size}GB memory limit, "
            f"{max_disk_size}GB disk limit, compression level {compression_level}"
        )

    def _get_size(self, obj: Any) -> int:
        """Get approximate size of object in bytes."""
        try:
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as e:
            logger.warning(f"Failed to calculate object size: {e}")
            return 0

    def put(self, key: str, data: Any, to_disk: bool = False) -> None:
        """
        Store data in cache.

        Args:
            key: Cache key to store data under
            data: Data to cache
            to_disk: Whether to store in disk cache instead of memory
        """
        if to_disk:
            self._put_disk(key, data)
        else:
            self._put_memory(key, data)

    def _put_memory(self, key: str, data: Any) -> None:
        """Store in memory cache with LRU eviction policy."""
        size = self._get_size(data)

        # Skip if data is too large for memory cache
        if size > self.max_memory_size:
            logger.warning(
                f"Data for key '{key}' ({size/1024**2:.2f}MB) exceeds memory cache "
                f"limit ({self.max_memory_size/1024**2:.2f}MB). Storing to disk instead."
            )
            self._put_disk(key, data)
            return

        # Free up space if needed
        current_size = sum(item[1] for item in self.memory_cache.values())
        while current_size + size > self.max_memory_size and self.memory_cache:
            # Find oldest accessed item
            oldest_key = min(self.memory_cache.items(), key=lambda x: x[1][2])[0]

            # Remove it
            _, removed_size, _ = self.memory_cache.pop(oldest_key)
            current_size -= removed_size
            logger.debug(f"Evicted '{oldest_key}' from memory cache")

        # Store with current timestamp
        self.memory_cache[key] = (data, size, datetime.now().timestamp())
        logger.debug(f"Stored '{key}' in memory cache ({size/1024**2:.2f}MB)")

    def _put_disk(self, key: str, data: Any) -> None:
        """Store in disk cache with LRU eviction policy."""
        # Create a safe filename from the key
        safe_key = "".join(c if c.isalnum() else "_" for c in key)
        cache_path = self.cache_dir / f"{safe_key}.cache"

        # Create a temp file to ensure atomic write
        fd, temp_path = tempfile.mkstemp(dir=self.cache_dir)
        os.close(fd)
        temp_path = Path(temp_path)

        try:
            # Save to temp file
            if self.compression_level > 0:
                with zipfile.ZipFile(
                    temp_path,
                    "w",
                    compression=zipfile.ZIP_DEFLATED,
                    compresslevel=self.compression_level,
                ) as zf:
                    # We need to pickle the data first
                    pickled_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                    zf.writestr("data.pkl", pickled_data)
            else:
                with open(temp_path, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Get size
            size = temp_path.stat().st_size

            # Free up space if needed
            current_size = sum(item[1] for item in self.disk_cache.values())
            while current_size + size > self.max_disk_size and self.disk_cache:
                # Find oldest accessed item
                oldest_key = min(self.disk_cache.items(), key=lambda x: x[1][2])[0]

                # Remove it
                oldest_path, removed_size, _ = self.disk_cache.pop(oldest_key)
                current_size -= removed_size
                try:
                    Path(oldest_path).unlink(missing_ok=True)
                    logger.debug(f"Evicted '{oldest_key}' from disk cache")
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {oldest_path}: {e}")

            # Atomic rename
            temp_path.replace(cache_path)
            self.disk_cache[key] = (str(cache_path), size, datetime.now().timestamp())

            # Update index file
            self._save_disk_index()

            logger.debug(f"Stored '{key}' in disk cache ({size/1024**2:.2f}MB)")

        except Exception as e:
            logger.error(f"Error caching to disk: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve data from cache.

        Args:
            key: Cache key to retrieve
            default: Default value to return if key not found

        Returns:
            Cached data or default value if not found
        """
        # Check memory first
        if key in self.memory_cache:
            data, size, _ = self.memory_cache[key]
            # Update access timestamp
            self.memory_cache[key] = (data, size, datetime.now().timestamp())
            logger.debug(f"Retrieved '{key}' from memory cache")
            return data

        # Then check disk
        if key in self.disk_cache:
            path, size, _ = self.disk_cache[key]
            try:
                # Update access timestamp
                self.disk_cache[key] = (path, size, datetime.now().timestamp())
                self._save_disk_index()

                # Load the data
                cache_path = Path(path)
                if cache_path.suffix == ".cache":
                    # Might be compressed or not
                    try:
                        # Try opening as zip file first
                        with zipfile.ZipFile(cache_path, "r") as zf:
                            with zf.open("data.pkl") as f:
                                data = pickle.load(f)
                                logger.debug(
                                    f"Retrieved '{key}' from compressed disk cache"
                                )
                    except zipfile.BadZipFile:
                        # Not compressed, read directly
                        with open(cache_path, "rb") as f:
                            data = pickle.load(f)
                            logger.debug(f"Retrieved '{key}' from disk cache")
                else:
                    # Old-style cache file
                    with open(cache_path, "rb") as f:
                        data = pickle.load(f)
                        logger.debug(f"Retrieved '{key}' from disk cache")

                # Try to move to memory if space allows
                memory_size = self._get_size(data)
                if (
                    memory_size <= self.max_memory_size / 2
                ):  # Only cache if reasonable size
                    self._put_memory(key, data)

                return data

            except Exception as e:
                logger.warning(f"Failed to load from disk cache: {e}")
                # Clean up failed cache
                self.remove(key)

        logger.debug(f"Cache miss for '{key}'")
        return default

    def remove(self, key: str) -> bool:
        """
        Remove item from cache.

        Args:
            key: Cache key to remove

        Returns:
            True if item was found and removed, False otherwise
        """
        found = False

        # Remove from memory
        if key in self.memory_cache:
            self.memory_cache.pop(key, None)
            found = True
            logger.debug(f"Removed '{key}' from memory cache")

        # Remove from disk
        if key in self.disk_cache:
            path, _, _ = self.disk_cache.pop(key)
            try:
                Path(path).unlink(missing_ok=True)
                logger.debug(f"Removed '{key}' from disk cache")
                self._save_disk_index()
                found = True
            except Exception as e:
                logger.warning(f"Failed to remove cache file {path}: {e}")

        return found

    def clear(self) -> None:
        """Clear all caches."""
        # Clear memory
        self.memory_cache.clear()
        logger.info("Cleared memory cache")

        # Clear disk
        for path, _, _ in self.disk_cache.values():
            try:
                Path(path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to remove cache file {path}: {e}")
        self.disk_cache.clear()
        self._save_disk_index()
        logger.info("Cleared disk cache")

    def _load_disk_index(self) -> None:
        """Load disk cache index from file."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    index_data = json.load(f)

                self.disk_cache = {}
                for key, (path, size, timestamp) in index_data.items():
                    # Verify file exists
                    if Path(path).exists():
                        self.disk_cache[key] = (path, size, timestamp)

                logger.debug(
                    f"Loaded disk cache index with {len(self.disk_cache)} entries"
                )
            except Exception as e:
                logger.warning(f"Failed to load disk cache index: {e}")
                self.disk_cache = {}

    def _save_disk_index(self) -> None:
        """Save disk cache index to file."""
        try:
            # Create temporary file first for atomic write
            fd, temp_path = tempfile.mkstemp(dir=self.cache_dir)
            os.close(fd)

            with open(temp_path, "w") as f:
                json.dump(self.disk_cache, f)

            # Atomic rename
            os.replace(temp_path, self.index_path)
            logger.debug(f"Saved disk cache index with {len(self.disk_cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to save disk cache index: {e}")
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass

    def close(self) -> None:
        """Clean up resources."""
        self._save_disk_index()


class DatasetStorage:
    """
    Manages dataset storage and versioning with compression support.

    This class provides:
    1. Organized storage for raw, processed, and cached data
    2. Versioning support for processed datasets
    3. Metadata tracking for each dataset
    4. Atomic file operations for crash safety

    Example usage:
    ```python
    storage = DatasetStorage("./data", compress=True)

    # Save processed dataset
    storage.save_processed(
        processed_data,
        name="lincs_drug_dataset",
        version="v2",
        metadata={"description": "LINCS L1000 drug response dataset"}
    )

    # Load dataset
    data, metadata = storage.load_processed("lincs_drug_dataset", version="v2")

    # Cache intermediate results
    storage.cache_data(embeddings, name="drug_embeddings")

    # Check if cached data exists
    if storage.is_cached("drug_embeddings"):
        embeddings, meta = storage.load_cached("drug_embeddings")
    ```
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        compress: bool = True,
        compression_level: int = 1,
        use_json_compression: bool = False,
    ):
        """
        Initialize dataset storage.

        Args:
            base_dir: Base directory for data storage
            compress: Whether to compress stored data
            compression_level: Compression level (0-9, 0=none)
            use_json_compression: Whether to use JSON compression for compatible data
        """
        self.base_dir = Path(base_dir)
        self.compress = compress
        self.compression_level = compression_level
        self.use_json_compression = use_json_compression

        # Create directories
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.cached_dir = self.base_dir / "cached"
        self.temp_dir = self.base_dir / "temp"

        for directory in [
            self.raw_dir,
            self.processed_dir,
            self.cached_dir,
            self.temp_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized DatasetStorage in {self.base_dir} "
            f"(compression: {'enabled' if compress else 'disabled'})"
        )

    def save_processed(
        self, data: Any, name: str, version: str = "v1", metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save processed data safely using atomic operations.

        Args:
            data: Data to save
            name: Dataset name
            version: Version string
            metadata: Additional metadata to store

        Returns:
            Path to saved data file
        """
        # Prepare version directory
        version_dir = self.processed_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Prepare metadata
        meta = {
            "name": name,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "compression": "zip" if self.compress else "none",
            **(metadata or {}),
        }

        # Create temporary directory for atomic operations
        temp_dir = self.temp_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_dir.mkdir(exist_ok=True)

        try:
            # Save metadata
            meta_path = temp_dir / f"{name}.meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            # Save data
            data_path = temp_dir / f"{name}.data"

            # Check if data is JSON-serializable and user wants JSON compression
            if self.use_json_compression and self._is_json_serializable(data):
                logger.debug(f"Using JSON serialization for {name}")
                with open(data_path, "w") as f:
                    json.dump(data, f)
                meta["serialization"] = "json"
            elif self.compress:
                logger.debug(f"Using compressed pickle serialization for {name}")
                with zipfile.ZipFile(
                    data_path,
                    "w",
                    compression=zipfile.ZIP_DEFLATED,
                    compresslevel=self.compression_level,
                ) as zf:
                    pickled_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                    zf.writestr("data.pkl", pickled_data)
                meta["serialization"] = "zip_pickle"
            else:
                logger.debug(f"Using pickle serialization for {name}")
                with open(data_path, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                meta["serialization"] = "pickle"

            # Update metadata with file size
            meta["size_bytes"] = data_path.stat().st_size
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            # Move to final locations atomically
            final_meta_path = version_dir / f"{name}.meta.json"
            final_data_path = version_dir / f"{name}.data"

            # Use shutil.move which works across file systems
            shutil.move(str(meta_path), str(final_meta_path))
            shutil.move(str(data_path), str(final_data_path))

            logger.info(
                f"Saved processed dataset '{name}' version '{version}' "
                f"({final_data_path.stat().st_size / (1024*1024):.2f} MB)"
            )

            return final_data_path

        except Exception as e:
            logger.error(f"Error saving processed data '{name}': {e}")
            raise
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")

    def _is_json_serializable(self, data: Any) -> bool:
        """Check if data is JSON-serializable."""
        try:
            json.dumps(data)
            return True
        except (TypeError, OverflowError):
            return False

    def load_processed(self, name: str, version: str = "v1") -> Tuple[Any, Dict]:
        """
        Load processed data.

        Args:
            name: Dataset name
            version: Version string

        Returns:
            Tuple of (data, metadata)

        Raises:
            FileNotFoundError: If data or metadata not found
        """
        version_dir = self.processed_dir / version
        data_path = version_dir / f"{name}.data"
        meta_path = version_dir / f"{name}.meta.json"

        if not data_path.exists():
            raise FileNotFoundError(f"Data not found: {data_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        # Load metadata
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        # Load data based on serialization format
        serialization = metadata.get("serialization", "pickle")

        if serialization == "json":
            with open(data_path, "r") as f:
                data = json.load(f)
        elif serialization == "zip_pickle":
            with zipfile.ZipFile(data_path, "r") as zf:
                with zf.open("data.pkl") as f:
                    data = pickle.load(f)
        else:  # Default to pickle
            with open(data_path, "rb") as f:
                data = pickle.load(f)

        logger.info(f"Loaded processed dataset '{name}' version '{version}'")
        return data, metadata

    def list_versions(self, name: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List available dataset versions.

        Args:
            name: Optional dataset name to filter by

        Returns:
            Dictionary mapping versions to dataset names
        """
        versions = {}

        for version_dir in self.processed_dir.glob("*"):
            if not version_dir.is_dir():
                continue

            version = version_dir.name
            versions[version] = []

            for meta_file in version_dir.glob("*.meta.json"):
                dataset_name = meta_file.stem.rsplit(".", 1)[0]
                if name is None or dataset_name == name:
                    versions[version].append(dataset_name)

        # Remove empty versions
        versions = {k: v for k, v in versions.items() if v}

        return versions

    def cache_data(self, data: Any, name: str, metadata: Optional[Dict] = None) -> Path:
        """
        Cache data for faster access.

        Args:
            data: Data to cache
            name: Cache name
            metadata: Additional metadata

        Returns:
            Path to cached data file
        """
        # Create safe filename
        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        cache_path = self.cached_dir / f"{safe_name}.cache"
        temp_path = (
            self.temp_dir
            / f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tmp"
        )

        try:
            cache_data = {
                "data": data,
                "metadata": {
                    "name": name,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {}),
                },
            }

            # Save based on compression settings
            if self.compress:
                with zipfile.ZipFile(
                    temp_path,
                    "w",
                    compression=zipfile.ZIP_DEFLATED,
                    compresslevel=self.compression_level,
                ) as zf:
                    pickled_data = pickle.dumps(
                        cache_data, protocol=pickle.HIGHEST_PROTOCOL
                    )
                    zf.writestr("data.pkl", pickled_data)
            else:
                with open(temp_path, "wb") as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Move to final location
            shutil.move(str(temp_path), str(cache_path))

            logger.info(
                f"Cached data '{name}' ({cache_path.stat().st_size / (1024*1024):.2f} MB)"
            )

            return cache_path

        except Exception as e:
            logger.error(f"Error caching data '{name}': {e}")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise

    def load_cached(self, name: str) -> Tuple[Any, Dict]:
        """
        Load cached data.

        Args:
            name: Cache name

        Returns:
            Tuple of (data, metadata)

        Raises:
            FileNotFoundError: If cache not found
        """
        # Create safe filename
        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        cache_path = self.cached_dir / f"{safe_name}.cache"

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")

        try:
            # Check if file is compressed (ZIP) or not
            try:
                with zipfile.ZipFile(cache_path, "r") as zf:
                    with zf.open("data.pkl") as f:
                        cache = pickle.load(f)
            except zipfile.BadZipFile:
                # Not compressed
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)

            logger.info(f"Loaded cached data '{name}'")
            return cache["data"], cache["metadata"]

        except Exception as e:
            logger.error(f"Error loading cached data '{name}': {e}")
            raise

    def clear_cache(self, name: Optional[str] = None) -> None:
        """
        Clear cache files.

        Args:
            name: Optional cache name to clear specific cache
        """
        if name:
            # Create safe filename
            safe_name = "".join(c if c.isalnum() else "_" for c in name)
            cache_path = self.cached_dir / f"{safe_name}.cache"
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    logger.info(f"Cleared cache '{name}'")
                except Exception as e:
                    logger.warning(f"Failed to delete cache '{name}': {e}")
        else:
            # Clear all caches
            cleared = 0
            for cache_file in self.cached_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                    cleared += 1
                except Exception as e:
                    logger.warning(f"Failed to delete cache {cache_file}: {e}")
            logger.info(f"Cleared {cleared} cache files")

    def is_cached(self, name: str) -> bool:
        """
        Check if data is cached.

        Args:
            name: Cache name

        Returns:
            True if data is cached, False otherwise
        """
        # Create safe filename
        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        return (self.cached_dir / f"{safe_name}.cache").exists()


class CheckpointManager:
    """
    Manages model checkpointing with support for distributed training and metrics tracking.

    Features:
    1. Automatic maintenance of top-k checkpoints
    2. Best and last checkpoint tracking
    3. Distributed training support
    4. Checkpoint metadata and metrics tracking
    5. Performance profile tracking for models

    Example usage:
    ```python
    # Initialize
    checkpoint_mgr = CheckpointManager(
        "models/resnet50",
        monitor="val_loss",
        mode="min",
        save_top_k=3
    )

    # Save checkpoint during training
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader)
        val_loss = validate_epoch(model, val_loader)

        metrics = {"train_loss": train_loss, "val_loss": val_loss}
        checkpoint_mgr.save(
            model, epoch, metrics, optimizer, scheduler
        )

    # Load checkpoint
    checkpoint = CheckpointManager.load(
        "models/resnet50/best.pt", model, optimizer, scheduler
    )
    ```
    """

    def __init__(
        self,
        dirpath: Union[str, Path],
        filename: str = "model_{epoch:02d}_{val_loss:.4f}",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        save_last: bool = True,
        save_best: bool = True,
        metric_history: bool = True,
    ):
        """
        Initialize checkpoint manager.

        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename pattern with placeholders
            monitor: Metric to monitor for determining best model
            mode: 'min' or 'max' for determining best model
            save_top_k: Number of best checkpoints to keep
            save_last: Whether to save last checkpoint
            save_best: Whether to save best checkpoint
            metric_history: Whether to track metric history
        """
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.save_best = save_best
        self.metric_history = metric_history

        # Create checkpoint directory and subdirectories
        self.dirpath.mkdir(parents=True, exist_ok=True)

        # Directory for metrics history
        if self.metric_history:
            (self.dirpath / "metrics").mkdir(exist_ok=True)

        # Initialize state
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_path = None
        self.checkpoints = []  # (path, score) tuples
        self.metrics_history = {}

        # Load existing state if present
        self._load_state()

        logger.info(
            f"Initialized CheckpointManager in {self.dirpath} "
            f"(monitor={monitor}, mode={mode}, save_top_k={save_top_k})"
        )

    def _load_state(self) -> None:
        """Load checkpoint manager state from disk."""
        state_path = self.dirpath / "checkpoint_state.json"
        if state_path.exists():
            try:
                with open(state_path, "r") as f:
                    state = json.load(f)

                self.best_score = state.get("best_score", self.best_score)
                self.best_path = state.get("best_path", self.best_path)
                self.checkpoints = [
                    (path, score)
                    for path, score in state.get("checkpoints", [])
                    if Path(path).exists()  # Only keep existing checkpoints
                ]

                logger.info(
                    f"Loaded checkpoint state: {len(self.checkpoints)} checkpoints, "
                    f"best score: {self.best_score}"
                )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint state: {e}")

    def _save_state(self) -> None:
        """Save checkpoint manager state to disk."""
        state_path = self.dirpath / "checkpoint_state.json"
        temp_path = self.dirpath / "checkpoint_state.tmp"

        try:
            state = {
                "best_score": self.best_score,
                "best_path": self.best_path,
                "checkpoints": self.checkpoints,
                "last_updated": datetime.now().isoformat(),
            }

            with open(temp_path, "w") as f:
                json.dump(state, f, indent=2)

            # Atomic rename
            os.replace(str(temp_path), str(state_path))

        except Exception as e:
            logger.warning(f"Failed to save checkpoint state: {e}")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _is_better(self, current: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == "min":
            return current < self.best_score
        return current > self.best_score

    def _update_metrics_history(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Update metrics history."""
        if not self.metric_history:
            return

        metrics_path = self.dirpath / "metrics" / f"metrics_{epoch:04d}.json"

        try:
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "metrics": metrics,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

            # Update in-memory history
            for key, value in metrics.items():
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append((epoch, value))

        except Exception as e:
            logger.warning(f"Failed to save metrics history: {e}")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if we have too many."""
        while len(self.checkpoints) > self.save_top_k:
            # Find checkpoint with worst score
            if self.mode == "min":
                # Higher is worse for "min" mode
                idx = max(
                    range(len(self.checkpoints)), key=lambda i: self.checkpoints[i][1]
                )
            else:
                # Lower is worse for "max" mode
                idx = min(
                    range(len(self.checkpoints)), key=lambda i: self.checkpoints[i][1]
                )

            # Get checkpoint to remove
            checkpoint_path, _ = self.checkpoints.pop(idx)

            # Skip if it's best or last
            best_path = self.dirpath / "best.pt"
            last_path = self.dirpath / "last.pt"

            if Path(checkpoint_path) == best_path or Path(checkpoint_path) == last_path:
                continue

            # Remove the checkpoint
            try:
                Path(checkpoint_path).unlink(missing_ok=True)
                logger.debug(f"Removed checkpoint {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")

        # Save updated state
        self._save_state()

    def save(
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
            model: PyTorch model or state dict
            epoch: Current epoch
            metrics: Dictionary of metrics
            optimizer: Optional optimizer to save state
            scheduler: Optional scheduler to save state
            additional_data: Additional data to save in checkpoint

        Returns:
            Path to saved checkpoint
        """
        # Format filename with metrics
        try:
            filename = self.filename.format(epoch=epoch, **metrics)
        except KeyError:
            # Fall back to simpler filename if formatting fails
            filename = f"model_epoch{epoch:04d}.pt"

        checkpoint_path = self.dirpath / filename
        temp_path = self.dirpath / f"{filename}.tmp"

        # Check if monitored metric exists
        current_score = metrics.get(self.monitor)
        if current_score is None:
            logger.warning(
                f"Monitored metric '{self.monitor}' not found in metrics. "
                f"Available metrics: {list(metrics.keys())}"
            )
            current_score = float("inf") if self.mode == "min" else float("-inf")

        # Update metrics history
        self._update_metrics_history(epoch, metrics)

        # Prepare checkpoint
        model_state_dict = None
        if isinstance(model, torch.nn.Module):
            # Handle distributed models
            if isinstance(model, DistributedDataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
        else:
            # Assume it's already a state dict
            model_state_dict = model

        checkpoint = {
            "epoch": epoch,
            "metrics": metrics,
            "model_state_dict": model_state_dict,
            "timestamp": datetime.now().isoformat(),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if additional_data is not None:
            checkpoint["additional_data"] = additional_data

        try:
            # Save to temp file first
            torch.save(checkpoint, temp_path)

            # Move to final location
            temp_path.replace(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Update checkpoints list
            self.checkpoints.append((str(checkpoint_path), current_score))

            # Check if this is best checkpoint
            is_best = False
            if current_score is not None and self._is_better(current_score):
                self.best_score = current_score
                self.best_path = str(checkpoint_path)
                is_best = True

                # Save as best checkpoint if requested
                if self.save_best:
                    best_path = self.dirpath / "best.pt"
                    shutil.copy2(str(checkpoint_path), str(best_path))
                    logger.info(
                        f"Updated best model checkpoint (score: {current_score:.6f})"
                    )

            # Save as last checkpoint if requested
            if self.save_last:
                last_path = self.dirpath / "last.pt"
                shutil.copy2(str(checkpoint_path), str(last_path))
                logger.debug("Updated last checkpoint")

            # Cleanup old checkpoints
            if self.save_top_k > 0:
                self._cleanup_old_checkpoints()

            # Save state
            self._save_state()

            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            raise

    def get_best_model_path(self) -> Optional[str]:
        """Get path to the best model checkpoint."""
        if self.best_path and Path(self.best_path).exists():
            return self.best_path

        # Check if best.pt exists
        best_path = self.dirpath / "best.pt"
        if best_path.exists():
            return str(best_path)

        return None

    def get_last_model_path(self) -> Optional[str]:
        """Get path to the last model checkpoint."""
        last_path = self.dirpath / "last.pt"
        if last_path.exists():
            return str(last_path)

        # Return most recent checkpoint if no last.pt
        if self.checkpoints:
            return self.checkpoints[-1][0]

        return None

    def get_metric_history(self, metric_name: str) -> List[Tuple[int, float]]:
        """
        Get history of a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            List of (epoch, value) tuples
        """
        return self.metrics_history.get(metric_name, [])

    @staticmethod
    def load(
        checkpoint_path: Union[str, Path],
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Optional model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            map_location: Optional device mapping for model loading
            strict: Whether to strictly enforce model state dict keys match

        Returns:
            Checkpoint dictionary

        Raises:
            FileNotFoundError: If checkpoint file not found
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

            # Restore model state
            if model is not None and "model_state_dict" in checkpoint:
                if isinstance(model, DistributedDataParallel):
                    model.module.load_state_dict(
                        checkpoint["model_state_dict"], strict=strict
                    )
                else:
                    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

            # Restore optimizer state
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Restore scheduler state
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            logger.info(
                f"Loaded checkpoint from {checkpoint_path} "
                f"(epoch: {checkpoint.get('epoch', 'unknown')})"
            )

            return checkpoint

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise
