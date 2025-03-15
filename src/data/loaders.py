import logging
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from functools import lru_cache
import weakref
import time

logger = logging.getLogger(__name__)


class GCTXLoader:
    """Efficient loader for GCTX files with simplified interface and optimized data access."""

    def __init__(self, 
        gctx_file: str, 
        cache_metadata: bool = True, 
        validate: bool = True,
        chunk_size: int = 10000,
        enable_lru_cache: bool = True,
        max_matrix_cache_size: int = 5):
        """
        Initialize the GCTX loader with advanced caching.
        
        Args:
            gctx_file: Path to the GCTX file
            cache_metadata: Whether to cache metadata in memory
            validate: Whether to validate the file on initialization
            chunk_size: Default chunk size for loading large datasets
            enable_lru_cache: Whether to enable LRU caching for frequent access
            max_matrix_cache_size: Maximum number of data matrices to cache
        """
        self.gctx_file = gctx_file
        self.chunk_size = chunk_size
        self.cache_metadata = cache_metadata
        self.enable_lru_cache = enable_lru_cache
        
        # Cached properties
        self._file_handle = None
        self._dimensions = None
        self._row_metadata = None
        self._col_metadata = None
        self._file_attrs = None
        
        # Advanced caching for data matrices
        self._matrix_cache = {}
        self._matrix_cache_access_times = {}
        self._max_matrix_cache_size = max_matrix_cache_size
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Optional validation
        if validate:
            self._validate_file()
            
        # Cache metadata if requested
        if cache_metadata:
            self._cache_metadata()
        
        # Initialize LRU caching for methods if enabled
        if enable_lru_cache:
            # Apply LRU caching to methods that return small data
            self._lru_get_row_ids = lru_cache(maxsize=128)(self.get_row_ids)
            self._lru_get_gene_symbols = lru_cache(maxsize=128)(self.get_gene_symbols)
        else:
            self._lru_get_row_ids = self.get_row_ids
            self._lru_get_gene_symbols = self.get_gene_symbols

    def _cache_matrix(self, key, data):
        """
        Cache a data matrix with LRU eviction policy.
        
        Args:
            key: Cache key (hashable)
            data: NumPy array to cache
        """
        if not self.enable_lru_cache:
            return
            
        # Check if we need to evict an item
        if len(self._matrix_cache) >= self._max_matrix_cache_size:
            # Find least recently used item
            lru_key = min(self._matrix_cache_access_times, 
                        key=self._matrix_cache_access_times.get)
            del self._matrix_cache[lru_key]
            del self._matrix_cache_access_times[lru_key]
            logger.debug(f"Evicted matrix from cache: {lru_key}")
        
        # Cache the new data
        self._matrix_cache[key] = data
        self._matrix_cache_access_times[key] = time.time()
        logger.debug(f"Cached matrix with key: {key}, shape: {data.shape}")
        
    def _handle_unordered_indices(self, indices):
        """
        Handle unordered indices for H5py which requires ascending order.
        
        Args:
            indices: Original indices, potentially unordered
            
        Returns:
            Tuple of (sorted_indices, restore_order) where restore_order is a list
            that can be used to restore the original order after fetching data
        """
        if indices is None:
            return slice(None), None
        if isinstance(indices, slice):
            return indices, None
        if isinstance(indices, slice):
            return indices, None
            
        # Convert to numpy array if it's a list
        if isinstance(indices, list):
            indices = np.array(indices)
            
        # Check if indices are already sorted
        if np.all(np.diff(indices) >= 0):
            return indices, None
            
        # Create sorting index and its inverse for restoring original order
        sorting_idx = np.argsort(indices)
        sorted_indices = indices[sorting_idx]
        
        # This mapping will restore original order: fetched_data[restore_order] = result
        restore_order = np.argsort(sorting_idx)
        
        return sorted_indices, restore_order
    
    def _validate_file(self) -> None:
        """Validate GCTX file structure."""
        try:
            with h5py.File(self.gctx_file, "r") as f:
                # Check required paths
                required_paths = [
                    "0/DATA/0/matrix", 
                    "0/META/ROW", 
                    "0/META/COL", 
                    "0/META/ROW/id", 
                    "0/META/COL/id"
                ]
                
                missing = [path for path in required_paths if path not in f]
                if missing:
                    raise ValueError(f"Invalid GCTX file: missing {', '.join(missing)}")
                
                # Verify dimensions can be read
                _ = f["0/DATA/0/matrix"].shape
                
                logger.debug(f"GCTX file validated: {self.gctx_file}")
        except Exception as e:
            logger.error(f"GCTX file validation failed: {str(e)}")
            raise
    
    def _cache_metadata(self) -> None:
        """Cache metadata to improve performance for repeated access."""
        try:
            with h5py.File(self.gctx_file, "r") as f:
                # Cache dimensions
                self._dimensions = f["0/DATA/0/matrix"].shape
                
                # Cache row and column metadata
                self._row_metadata = self._load_metadata(f, "ROW")
                self._col_metadata = self._load_metadata(f, "COL")
                
                # Cache file attributes
                self._file_attrs = {
                    k: v.decode("utf-8") if isinstance(v, bytes) else v 
                    for k, v in f.attrs.items()
                }
                
                logger.debug(
                    f"Cached metadata: {self._dimensions[0]} rows, "
                    f"{self._dimensions[1]} columns"
                )
        except Exception as e:
            logger.warning(f"Failed to cache metadata: {str(e)}")
            # Reset cache values if caching fails
            self._dimensions = None
            self._row_metadata = None
            self._col_metadata = None
            self._file_attrs = None
    
    def _load_metadata(self, file_handle, meta_type: str) -> pd.DataFrame:
        """
        Load row or column metadata.
        
        Args:
            file_handle: Open h5py file handle
            meta_type: 'ROW' or 'COL'
            
        Returns:
            DataFrame containing metadata
        """
        meta_group = file_handle[f"0/META/{meta_type}"]
        meta_dict = {}
        
        for key in meta_group.keys():
            data = meta_group[key][:]
            
            # Convert byte strings to unicode
            if data.dtype.kind == "S":
                data = [s.decode("utf-8", errors="ignore") for s in data]
                
            meta_dict[key] = data
        
        return pd.DataFrame(meta_dict)
    
    def _get_indices(self, 
                      selection: Union[slice, list, np.ndarray, None], 
                      max_size: int) -> Union[slice, list]:
        """
        Convert various index types to a consistent format.
        
        Args:
            selection: Input selection (slice, list, array, or None)
            max_size: Maximum valid index
            
        Returns:
            Standardized selection as slice or list
        """
        if selection is None:
            return slice(None)
            
        # Convert numpy array to list
        if isinstance(selection, np.ndarray):
            selection = selection.tolist()
            
        # Handle slice
        if isinstance(selection, slice):
            start = 0 if selection.start is None else selection.start
            stop = max_size if selection.stop is None else selection.stop
            
            if start < 0 or (stop is not None and stop > max_size):
                raise IndexError(f"Slice indices out of bounds (0 to {max_size-1})")
                
            return selection
            
        # Handle list of indices
        if isinstance(selection, list):
            if not selection:
                return []
                
            selection = [int(i) for i in selection]
            
            if any(i < 0 or i >= max_size for i in selection):
                raise IndexError(f"Indices out of bounds (0 to {max_size-1})")
                
            return selection
            
        raise TypeError(
            f"Selection must be None, slice, list, or array, got {type(selection)}"
        )
    
    def _get_gene_indices(self, feature_space: Union[str, List[str]]) -> List[int]:
        """
        Get gene indices for specified feature space.
        
        Args:
            feature_space: Feature space identifier(s)
            
        Returns:
            List of column indices
        """
        col_meta = self.get_column_metadata()
        
        if "feature_space" not in col_meta.columns:
            raise ValueError("Column 'feature_space' not found in column metadata")
            
        allowed = {"landmark", "best inferred", "inferred"}
        fs = [feature_space] if isinstance(feature_space, str) else feature_space
        
        if any(x == "all" for x in fs):
            fs = list(allowed)
        elif any(x not in allowed for x in fs):
            raise ValueError(f"Invalid feature_space: {fs}. Allowed: {allowed} or 'all'")
            
        selected = col_meta[col_meta["feature_space"].isin(fs)]
        
        if selected.empty:
            raise ValueError(f"No genes found for feature_space: {fs}")
            
        return selected.index.tolist()
    
    def _indices_to_hashable(self, indices):
        """Convert various index types to a hashable form for cache keys."""
        if indices is None:
            return None
        elif isinstance(indices, slice):
            return (indices.start, indices.stop, indices.step)
        elif isinstance(indices, list):
            # For long lists, hash the content to keep key size manageable
            if len(indices) > 100:
                # Use first, last, and length + hash to represent the list
                return (indices[0], indices[-1], len(indices), hash(tuple(indices)))
            return tuple(indices)
        elif isinstance(indices, np.ndarray):
            # Hash numpy array for caching
            return hash(indices.tobytes())
        return indices
    
    @property
    def dimensions(self) -> Tuple[int, int]:
        """Get data matrix dimensions (rows, columns)."""
        if self._dimensions is None:
            with h5py.File(self.gctx_file, "r") as f:
                self._dimensions = f["0/DATA/0/matrix"].shape
        return self._dimensions
    
    @property
    def n_rows(self) -> int:
        """Get number of rows in the data matrix."""
        return self.dimensions[0]
    
    @property
    def n_cols(self) -> int:
        """Get number of columns in the data matrix."""
        return self.dimensions[1]
    
    def clear_cache(self) -> None:
        """Clear all cached metadata."""
        self._dimensions = None
        self._row_metadata = None
        self._col_metadata = None
        self._file_attrs = None
        logger.info("Cache cleared")
    
    def get_file_attributes(self) -> Dict:
        """Get file-level metadata attributes."""
        if self._file_attrs is not None:
            return self._file_attrs
            
        with h5py.File(self.gctx_file, "r") as f:
            attrs = {
                k: v.decode("utf-8") if isinstance(v, bytes) else v 
                for k, v in f.attrs.items()
            }
            if self.cache_metadata:
                self._file_attrs = attrs
            return attrs
    

    def get_data_matrix(self, 
        row_indices: Union[slice, list, np.ndarray, None] = None,
        col_indices: Union[slice, list, np.ndarray, None] = None,
        feature_space: Optional[Union[str, List[str]]] = None,
        chunk_size: Optional[int] = None,
        use_cache: bool = True) -> np.ndarray:
        """
        Get expression data matrix with caching.
        
        Args:
            row_indices: Rows to select
            col_indices: Columns to select
            feature_space: Feature space filter
            chunk_size: Size of chunks for loading large data
            use_cache: Whether to use cache for this request
            
        Returns:
            NumPy array of expression data
        """
        if col_indices is not None and feature_space is not None:
            raise ValueError("Cannot specify both col_indices and feature_space")
            
        if feature_space is not None:
            col_indices = self._get_gene_indices(feature_space)
        
        cache_key = None    
        # Generate a cache key
        if use_cache and self.enable_lru_cache:
            # Convert indices to hashable form for the cache key
            row_key = self._indices_to_hashable(row_indices)
            col_key = self._indices_to_hashable(col_indices)
            cache_key = (row_key, col_key)
            
            # Check if data is in cache
            if cache_key in self._matrix_cache:
                # Update access time
                self._matrix_cache_access_times[cache_key] = time.time()
                self._cache_hits += 1
                logger.debug(f"Cache hit for matrix: {cache_key}")
                return self._matrix_cache[cache_key]
            
            self._cache_misses += 1
            
        chunk_size = chunk_size or self.chunk_size
        
        # Sort indices if needed for h5py
        rows, row_restore_order = self._handle_unordered_indices(row_indices)
        cols, col_restore_order = self._handle_unordered_indices(col_indices)
        
        with h5py.File(self.gctx_file, "r") as f:
            matrix = f["0/DATA/0/matrix"]
            h5py_rows = self._get_indices(rows, self.n_rows)
            h5py_cols = self._get_indices(cols, self.n_cols)
            
            # For simple slice operations, direct access is fastest
            if isinstance(h5py_rows, slice) and isinstance(h5py_cols, slice):
                data = matrix[h5py_rows, h5py_cols]
                
            # Chunked loading for large list selections
            elif isinstance(h5py_rows, list) and len(h5py_rows) > chunk_size:
                chunks = [h5py_rows[i:i+chunk_size] for i in range(0, len(h5py_rows), chunk_size)]
                data_chunks = []
                
                for chunk in tqdm(chunks, desc="Loading data chunks"):
                    if isinstance(h5py_cols, list):
                        # Need to load rows then select columns
                        chunk_data = matrix[chunk][:][:, h5py_cols]
                    else:
                        # Can select columns directly
                        chunk_data = matrix[chunk, h5py_cols]
                    data_chunks.append(chunk_data)
                    
                data = np.vstack(data_chunks)
                
            # Handle large slices by converting to chunks
            elif isinstance(h5py_rows, slice):
                start = 0 if h5py_rows.start is None else h5py_rows.start
                stop = self.n_rows if h5py_rows.stop is None else h5py_rows.stop
                step = 1 if h5py_rows.step is None else h5py_rows.step
                
                if (stop - start) > chunk_size:
                    chunks = list(range(start, stop, chunk_size))
                    data_chunks = []
                    
                    for chunk_start in tqdm(chunks, desc="Loading data chunks"):
                        chunk_stop = min(chunk_start + chunk_size, stop)
                        chunk_slice = slice(chunk_start, chunk_stop, step)
                        
                        if isinstance(cols, list):
                            chunk_data = matrix[chunk_slice][:][:, h5py_cols]
                        else:
                            chunk_data = matrix[chunk_slice, h5py_cols]
                            
                        data_chunks.append(chunk_data)
                        
                    data = np.vstack(data_chunks)
                else:
                    # Small slice, handle directly
                    if isinstance(h5py_cols, list):
                        data = matrix[h5py_rows][:][:, h5py_cols]
                    else:
                        data = matrix[h5py_rows, h5py_cols]
            
            # Handle remaining cases (small lists or single indices)
            else:
                if isinstance(h5py_cols, list):
                    data = matrix[rows][:][:, cols]
                else:
                    data = matrix[rows, cols]
                    
            # Restore original order if needed
            if row_restore_order is not None:
                data = data[row_restore_order]
            if col_restore_order is not None:
                data = data[:, col_restore_order]
        
        # Cache the result if caching is enabled (after we have the data)
        if use_cache and self.enable_lru_cache and cache_key is not None:
            self._cache_matrix(cache_key, data)
            
        return data
            
    def get_cache_stats(self):
        """Get statistics about cache performance."""
        if not self.enable_lru_cache:
            return {"enabled": False}
        
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "enabled": True,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "matrices_cached": len(self._matrix_cache),
            "max_matrices": self._max_matrix_cache_size
        }
    
    def get_row_metadata(self,
                         row_indices: Union[slice, list, np.ndarray, None] = None,
                         columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Get row metadata.
        
        Args:
            row_indices: Rows to select
            columns: Metadata columns to include
            
        Returns:
            DataFrame of row metadata
        """
        # Return cached complete metadata if applicable
        if self._row_metadata is not None and row_indices is None:
            df = self._row_metadata
            if columns is not None:
                columns = [columns] if isinstance(columns, str) else columns
                return df[columns]
            return df
            
        with h5py.File(self.gctx_file, "r") as f:
            meta_group = f["0/META/ROW"]
            available_keys = list(meta_group.keys())
            
            # Handle column selection
            if columns is None:
                columns = available_keys
            elif isinstance(columns, str):
                columns = [columns]
                
            missing = [col for col in columns if col not in available_keys]
            if missing:
                raise KeyError(f"Columns not found: {', '.join(missing)}")
                
            # Sort indices if needed for h5py
            indices, restore_order = self._handle_unordered_indices(row_indices)
            
            # Convert row selection to valid indices
            h5py_indices = self._get_indices(indices, self.n_rows)
            
            # Extract requested metadata
            meta_dict = {}
            for key in columns:
                data = meta_group[key][h5py_indices]
                
                # Restore original order if needed
                if restore_order is not None:
                    data = data[restore_order]
                    
                if data.dtype.kind == "S":
                    data = [s.decode("utf-8", errors="ignore") for s in data]
                meta_dict[key] = data
                
            df = pd.DataFrame(meta_dict)
            if len(columns) == 1:
                return df[columns[0]]
            return df
    
    def get_column_metadata(self,
                            col_indices: Union[slice, list, np.ndarray, None] = None,
                            columns: Optional[Union[str, List[str]]] = None,
                            feature_space: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Get column metadata.
        
        Args:
            col_indices: Columns to select
            columns: Metadata columns to include
            feature_space: Filter by feature space
            
        Returns:
            DataFrame of column metadata
        """
        if col_indices is not None and feature_space is not None:
            raise ValueError("Cannot specify both col_indices and feature_space")
            
        if feature_space is not None:
            col_indices = self._get_gene_indices(feature_space)
            
        # Return cached complete metadata if applicable  
        if self._col_metadata is not None and col_indices is None:
            df = self._col_metadata
            if columns is not None:
                columns = [columns] if isinstance(columns, str) else columns
                return df[columns]
            return df
            
        with h5py.File(self.gctx_file, "r") as f:
            meta_group = f["0/META/COL"]
            available_keys = list(meta_group.keys())
            
            # Handle column selection
            if columns is None:
                columns = available_keys
            elif isinstance(columns, str):
                columns = [columns]
                
            missing = [col for col in columns if col not in available_keys]
            if missing:
                raise KeyError(f"Columns not found: {', '.join(missing)}")
                
            # Convert column selection to valid indices
            indices = self._get_indices(col_indices, self.n_cols)
            
            # Extract requested metadata
            meta_dict = {}
            for key in columns:
                data = meta_group[key][indices]
                if data.dtype.kind == "S":
                    data = [s.decode("utf-8", errors="ignore") for s in data]
                meta_dict[key] = data
                
            df = pd.DataFrame(meta_dict)
            if len(columns) == 1:
                return df[columns[0]]
            return df
    
    def get_row_ids(self, 
        row_indices: Union[slice, list, np.ndarray, None] = None) -> List[str]:
        """
        Get row IDs (experiment identifiers) with optional caching.
        
        Args:
            row_indices: Rows to select
            
        Returns:
            List of row IDs
        """
        # if self.enable_lru_cache and row_indices is None:
        #     # Use cached version for complete dataset
        #     return self._lru_get_row_ids(None)
        
        # Fall back to standard implementation
        return self.get_row_metadata(row_indices, "id").values.flatten().tolist()
    
    def get_gene_symbols(self, 
        col_indices: Union[slice, list, np.ndarray, None] = None,
        feature_space: Optional[Union[str, List[str]]] = None) -> List[str]:
        """
        Get gene symbols with optional caching.
        
        Args:
            col_indices: Columns to select
            feature_space: Filter by feature space
            
        Returns:
            List of gene symbols
        """        
        return self.get_column_metadata(col_indices, "id", feature_space).values.flatten().tolist()
    
    def get_complete_dataset(self,
                            row_indices: Union[slice, list, np.ndarray, None] = None,
                            col_indices: Union[slice, list, np.ndarray, None] = None,
                            feature_space: Optional[Union[str, List[str]]] = None,
                            row_meta_columns: Optional[Union[str, List[str]]] = None,
                            col_meta_columns: Optional[Union[str, List[str]]] = None,
                            chunk_size: Optional[int] = None) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Get data matrix with corresponding row and column metadata.
        
        Args:
            row_indices: Rows to select
            col_indices: Columns to select
            feature_space: Filter by feature space
            row_meta_columns: Row metadata columns to include
            col_meta_columns: Column metadata columns to include
            chunk_size: Size of chunks for loading large data
            
        Returns:
            Tuple of (data matrix, row metadata, column metadata)
        """
        # Handle feature space
        if col_indices is not None and feature_space is not None:
            raise ValueError("Cannot specify both col_indices and feature_space")
            
        if feature_space is not None:
            col_indices = self._get_gene_indices(feature_space)
        
        # Get data and metadata with one file open operation
        with h5py.File(self.gctx_file, "r") as f:
            # Get data matrix
            matrix = f["0/DATA/0/matrix"]
            rows = self._get_indices(row_indices, self.n_rows)
            cols = self._get_indices(col_indices, self.n_cols)
            
            # Get data with chunking if needed
            chunk_size = chunk_size or self.chunk_size
            
            # Simple case: both are slices
            if isinstance(rows, slice) and isinstance(cols, slice):
                data = matrix[rows, cols]
            # Handle more complex cases with chunking
            else:
                if isinstance(rows, list) and len(rows) > chunk_size:
                    # Chunked loading for large list selections
                    chunks = [rows[i:i+chunk_size] for i in range(0, len(rows), chunk_size)]
                    data_chunks = []
                    
                    for chunk in tqdm(chunks, desc="Loading data chunks"):
                        if isinstance(cols, list):
                            # Need to load rows then select columns
                            chunk_data = matrix[chunk][:][:, cols]
                        else:
                            # Can select columns directly
                            chunk_data = matrix[chunk, cols]
                        data_chunks.append(chunk_data)
                        
                    data = np.vstack(data_chunks)
                else:
                    # Handle remaining cases
                    if isinstance(cols, list):
                        data = matrix[rows][:][:, cols]
                    else:
                        data = matrix[rows, cols]
            
            # Get row metadata
            row_meta = self.get_row_metadata(row_indices, row_meta_columns)
            
            # Get column metadata
            col_meta = self.get_column_metadata(col_indices, col_meta_columns)
            
            return data, row_meta, col_meta
    
    # Context manager support for with statements
    def __enter__(self):
        """Context manager entry."""
        self._file_handle = h5py.File(self.gctx_file, "r")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def get_data_matrix_parallel(self, 
        row_indices: Union[slice, list, np.ndarray, None] = None,
        col_indices: Union[slice, list, np.ndarray, None] = None,
        feature_space: Optional[Union[str, List[str]]] = None,
        chunk_size: Optional[int] = None,
        n_workers: int = 4) -> np.ndarray:
        """
        Get expression data matrix using parallel loading.
        
        Args:
            row_indices: Rows to select
            col_indices: Columns to select
            feature_space: Feature space filter
            chunk_size: Size of chunks for loading large data
            n_workers: Number of parallel workers
            
        Returns:
            NumPy array of expression data
        """
        if col_indices is not None and feature_space is not None:
            raise ValueError("Cannot specify both col_indices and feature_space")
            
        if feature_space is not None:
            col_indices = self._get_gene_indices(feature_space)
            
        chunk_size = chunk_size or self.chunk_size
        
        # Get dimensions and prepare indices
        rows = self._get_indices(row_indices, self.n_rows)
        cols = self._get_indices(col_indices, self.n_cols)
        
        # For simple slice operations, use the non-parallel version
        if isinstance(rows, slice) and isinstance(cols, slice):
            with h5py.File(self.gctx_file, "r") as f:
                return f["0/DATA/0/matrix"][rows, cols]
        
        # For list-based row selections, prepare chunks for parallel loading
        if isinstance(rows, list):
            chunks = [rows[i:i+chunk_size] for i in range(0, len(rows), chunk_size)]
        elif isinstance(rows, slice):
            # Convert slice to chunks
            start = 0 if rows.start is None else rows.start
            stop = self.n_rows if rows.stop is None else rows.stop
            step = 1 if rows.step is None else rows.step
            
            if (stop - start) > chunk_size:
                # Create chunks from slice
                chunks = []
                for chunk_start in range(start, stop, chunk_size):
                    chunk_stop = min(chunk_start + chunk_size, stop)
                    chunks.append(slice(chunk_start, chunk_stop, step))
            else:
                # Small slice, no need for chunking
                chunks = [rows]
        else:
            # Single row or empty selection
            chunks = [rows]
        
        # Define function to process a single chunk
        def process_chunk(chunk):
            # Need to open a new file handle for each thread
            with h5py.File(self.gctx_file, "r") as f:
                matrix = f["0/DATA/0/matrix"]
                
                if isinstance(cols, list):
                    # Need two-step indexing for list-based column selection
                    return matrix[chunk][:][:, cols]
                else:
                    # Can use direct indexing for slice-based column selection
                    return matrix[chunk, cols]
        
        # Use thread pool to process chunks in parallel
        results = []
        with tqdm(total=len(chunks), desc="Loading data chunks in parallel") as pbar:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                # Submit all tasks
                future_to_chunk = {executor.submit(process_chunk, chunk): i 
                                for i, chunk in enumerate(chunks)}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        chunk_data = future.result()
                        # Store result with original index to maintain order
                        results.append((chunk_idx, chunk_data))
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                        raise
        
        # Sort results by original index to maintain correct order
        results.sort(key=lambda x: x[0])
        data_chunks = [r[1] for r in results]
        
        # Combine chunks into final result
        return np.vstack(data_chunks) if len(data_chunks) > 1 else data_chunks[0]

    def clear_cache(self, clear_metadata=True, clear_matrices=True):
        """
        Clear cache selectively.
        
        Args:
            clear_metadata: Whether to clear metadata cache
            clear_matrices: Whether to clear matrix cache
        """
        if clear_metadata:
            self._dimensions = None
            self._row_metadata = None
            self._col_metadata = None
            self._file_attrs = None
        
        if clear_matrices:
            self._matrix_cache.clear()
            self._matrix_cache_access_times.clear()
        
        # Reset cache statistics
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"Cache cleared - metadata:{clear_metadata}, matrices:{clear_matrices}")