# src/data/loaders.py
import logging
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GCTXDataLoader:
    """Data loader for GCTX files with chunked loading and progress feedback."""

    def __init__(self, gctx_file: str, preload_metadata: bool = True, validate_on_init: bool = False):
        self.gctx_file = gctx_file
        self.f = None
        self._n_rows, self._n_cols = None, None
        self._col_metadata = None
        self._row_metadata = None 
        self._metadata_attrs = None
        
        # Optional validation
        if validate_on_init:
            validation_results = self.validate()
            if not all(validation_results.values()):
                issues = [k for k, v in validation_results.items() if not v]
                logger.warning(f"Validation issues detected: {issues}")
                if not validation_results["file_accessible"]:
                    raise IOError(f"Cannot access GCTX file: {self.gctx_file}")
                if not validation_results["correct_format"]:
                    raise ValueError(f"Invalid GCTX file format: {self.gctx_file}")

        # Preload metadata during initialization
        if preload_metadata:
            try:
                with h5py.File(self.gctx_file, "r") as f:
                    self._n_rows, self._n_cols = f["0/DATA/0/matrix"].shape
                    self._col_metadata = self._load_column_metadata(f)
                    self._row_metadata = self._load_row_metadata(f)
                    self._metadata_attrs = self._load_metadata_attributes(f)
                    logger.debug(
                        f"Preloaded metadata with {len(self._col_metadata)} genes and {len(self._row_metadata)} samples."
                    )
            except Exception as e:
                logger.warning(f"Failed to preload metadata: {str(e)}")

    def _load_column_metadata(self, f=None) -> pd.DataFrame:
        """
        Load column metadata, with option to pass an open file handle.

        Args:
            f: Optional open h5py File object

        Returns:
            DataFrame with column metadata
        """
        close_file = f is None
        try:
            if f is None:
                f = h5py.File(self.gctx_file, "r")

            meta_col_grp = f["0/META/COL"]
            col_dict = {key: meta_col_grp[key][:] for key in meta_col_grp.keys()}
            for key, data in col_dict.items():
                if data.dtype.kind == "S":
                    col_dict[key] = [s.decode("utf-8", errors="ignore") for s in data]

            return pd.DataFrame(col_dict)

        finally:
            if close_file and "f" in locals():
                f.close()
                
    def _load_row_metadata(self, f=None) -> pd.DataFrame:
        """
        Load row metadata, with option to pass an open file handle.

        Args:
            f: Optional open h5py File object

        Returns:
            DataFrame with row metadata
        """
        close_file = f is None
        try:
            if f is None:
                f = h5py.File(self.gctx_file, "r")

            meta_row_grp = f["0/META/ROW"]
            row_dict = {key: meta_row_grp[key][:] for key in meta_row_grp.keys()}
            for key, data in row_dict.items():
                if data.dtype.kind == "S":
                    row_dict[key] = [s.decode("utf-8", errors="ignore") for s in data]

            return pd.DataFrame(row_dict)

        finally:
            if close_file and "f" in locals():
                f.close()
                
    def _load_metadata_attributes(self, f=None) -> Dict:
        """
        Load file metadata attributes.
        
        Args:
            f: Optional open h5py File object
            
        Returns:
            Dictionary of metadata attributes
        """
        close_file = f is None
        try:
            if f is None:
                f = h5py.File(self.gctx_file, "r")
                
            attrs = {
                key: (
                    value.decode("utf-8", errors="ignore")
                    if isinstance(value, bytes)
                    else value
                )
                for key, value in f.attrs.items()
            }
            return attrs
            
        finally:
            if close_file and "f" in locals():
                f.close()

    def __enter__(self):
        """Context manager entry."""
        if self.f is not None:
            self.f.close()

        self.f = h5py.File(self.gctx_file, "r")

        # Validate file structure
        required = ["0/DATA/0/matrix", "0/META/ROW", "0/META/COL", "0/META/ROW/id", "0/META/COL/id"]
        for path in required:
            if path not in self.f:
                raise ValueError(f"Invalid .gctx file: missing {path}")

        self._n_rows, self._n_cols = self.f["0/DATA/0/matrix"].shape

        # Ensure metadata is loaded
        if self._col_metadata is None:
            self._col_metadata = self._load_column_metadata(self.f)

        if self._row_metadata is None:
            self._row_metadata = self._load_row_metadata(self.f)
            
        if self._metadata_attrs is None:
            self._metadata_attrs = self._load_metadata_attributes(self.f)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.f is not None:
            self.f.close()
            self.f = None
            
    @property
    def n_rows(self) -> int:
        """Get the number of rows in the data matrix."""
        if self._n_rows is None:
            with h5py.File(self.gctx_file, "r") as f:
                self._n_rows = f["0/DATA/0/matrix"].shape[0]
        return self._n_rows
    
    @property
    def n_cols(self) -> int:
        """Get the number of columns in the data matrix."""
        if self._n_cols is None:
            with h5py.File(self.gctx_file, "r") as f:
                self._n_cols = f["0/DATA/0/matrix"].shape[1]
        return self._n_cols
        
    @property
    def row_metadata(self) -> pd.DataFrame:
        """Get the full row metadata as a DataFrame."""
        if self._row_metadata is None:
            self._row_metadata = self._load_row_metadata()
        return self._row_metadata
        
    @property
    def col_metadata(self) -> pd.DataFrame:
        """Get the full column metadata as a DataFrame."""
        if self._col_metadata is None:
            self._col_metadata = self._load_column_metadata()
        return self._col_metadata
        
    @property
    def metadata_attrs(self) -> Dict:
        """Get metadata attributes."""
        if self._metadata_attrs is None:
            self._metadata_attrs = self._load_metadata_attributes()
        return self._metadata_attrs
    
    def get_row_metadata_keys(self) -> List[str]:
        """
        Get the keys/column names of the row metadata.
        
        Returns:
            List of row metadata column names
        """
        with h5py.File(self.gctx_file, "r") as f:
            return list(f["0/META/ROW"].keys())
    
    def get_column_metadata_keys(self) -> List[str]:
        """
        Get the keys/column names of the column metadata.
        
        Returns:
            List of column metadata column names
        """
        with h5py.File(self.gctx_file, "r") as f:
            return list(f["0/META/COL"].keys())
        
    def _convert_to_indices(
        self, selection: Union[slice, list, np.ndarray, None], max_size: int
    ) -> Union[slice, list]:
        """
        Convert various index types to a consistent format.
        
        Args:
            selection: Input selection (slice, list, numpy array, or None)
            max_size: Maximum number of rows/columns
            
        Returns:
            Slice or list of indices
        """
        if selection is None:
            return slice(None)
            
        # Convert numpy array to list
        if isinstance(selection, np.ndarray):
            selection = selection.tolist()
            
        # Handle slice
        if isinstance(selection, slice):
            # Validate slice bounds
            start = 0 if selection.start is None else selection.start
            stop = max_size if selection.stop is None else selection.stop
            
            if start < 0 or (stop is not None and stop > max_size):
                raise IndexError(f"Slice indices out of bounds (0 to {max_size-1})")
                
            return selection
            
        # Handle list of indices
        if isinstance(selection, list):
            # Handle empty list
            if not selection:
                return []
                
            # Convert to integer indices if not already
            selection = [int(i) for i in selection]
            
            # Check bounds
            if any(i < 0 or i >= max_size for i in selection):
                raise IndexError(f"Indices out of bounds (0 to {max_size-1})")
                
            return selection
            
        # If we've reached here, it's an unsupported type
        raise TypeError(
            f"Selection must be None, a slice, list, or numpy array of indices, got {type(selection)}"
        )

    def get_gene_indices_for_feature_space(
        self, feature_space: Union[str, List[str]]
    ) -> List[int]:
        if "feature_space" not in self._col_metadata.columns:
            raise ValueError("Column 'feature_space' not found in column metadata.")
        allowed_values = {"landmark", "best inferred", "inferred"}
        fs = [feature_space] if isinstance(feature_space, str) else feature_space
        if any(x == "all" for x in fs):
            fs = list(allowed_values)
        elif any(x not in allowed_values for x in fs):
            raise ValueError(
                f"Invalid feature_space: {fs}. Allowed: {allowed_values} or 'all'."
            )
        selected_genes = self._col_metadata[
            self._col_metadata["feature_space"].isin(fs)
        ]
        if selected_genes.empty:
            raise ValueError(f"No genes found for feature_space: {fs}")
        return selected_genes.index.tolist()

    def get_expression_data(
        self,
        row_slice: Union[slice, list, np.ndarray, None] = None,
        col_slice: Union[slice, list, np.ndarray, None] = None,
        feature_space: Optional[Union[str, List[str]]] = None,
        chunk_size: int = 10000,
    ) -> np.ndarray:
        """
        Robust method to get expression data with proper file handling.

        Args:
            row_slice: Rows to select
            col_slice: Columns to select
            feature_space: Feature space to use
            chunk_size: Size of chunks for large datasets

        Returns:
            Numpy array of expression data
        """
        try:
            # Open file
            with h5py.File(self.gctx_file, "r") as f:
                # Validate file structure
                if "0/DATA/0/matrix" not in f:
                    raise ValueError("Invalid GCTX file structure")

                matrix = f["0/DATA/0/matrix"]
                self._n_rows, self._n_cols = matrix.shape

                # Process column selection
                if col_slice is not None and feature_space is not None:
                    raise ValueError("Cannot specify both col_slice and feature_space.")

                col_selection = self._convert_to_indices(
                    (
                        col_slice
                        or self.get_gene_indices_for_feature_space(feature_space)
                        if feature_space
                        else None
                    ),
                    self._n_cols,
                )

                # Process row selection
                row_selection = self._convert_to_indices(row_slice, self._n_rows)

                # Strategy for handling different selection types
                def select_data(rows, cols):
                    if isinstance(rows, slice) and isinstance(cols, slice):
                        return matrix[rows, cols]
                    elif isinstance(rows, slice):
                        return matrix[rows, :][:, cols]
                    elif isinstance(cols, slice):
                        return matrix[rows, :][:, cols]
                    else:
                        # This is the most memory-intensive approach
                        # Retrieve the entire matrix first, then select
                        full_data = matrix[:]
                        return full_data[rows, :][:, cols]

                # Handle chunked loading for list-based row selection
                if isinstance(row_selection, list) and len(row_selection) > chunk_size:
                    data = []
                    for chunk in tqdm(
                        [
                            row_selection[i : i + chunk_size]
                            for i in range(0, len(row_selection), chunk_size)
                        ],
                        desc="Loading expression chunks",
                    ):
                        chunk_data = select_data(chunk, col_selection)
                        data.append(chunk_data)
                    
                    return np.concatenate(data, axis=0)
                # For slice-based selection that might be too large
                elif isinstance(row_selection, slice):
                    start = 0 if row_selection.start is None else row_selection.start
                    stop = self._n_rows if row_selection.stop is None else row_selection.stop
                    step = 1 if row_selection.step is None else row_selection.step
                    
                    # If slice covers a large range, process in chunks
                    if (stop - start) > chunk_size:
                        data = []
                        for chunk_start in tqdm(
                            range(start, stop, chunk_size), 
                            desc="Loading expression chunks"
                        ):
                            chunk_stop = min(chunk_start + chunk_size, stop)
                            chunk_slice = slice(chunk_start, chunk_stop, step)
                            chunk_data = select_data(chunk_slice, col_selection)
                            data.append(chunk_data)
                        
                        return np.concatenate(data, axis=0)

                # Direct data selection
                return select_data(row_selection, col_selection)

        except Exception as e:
            logger.error(f"Error retrieving expression data: {e}")
            raise
    
    def invalidate_cache(self):
        """Clear all cached metadata to force reload on next access."""
        self._row_metadata = None 
        self._col_metadata = None
        self._metadata_attrs = None
        logger.info("Cache invalidated")
        
    def validate(self) -> Dict[str, bool]:
        """
        Validate the current state of the loader.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "file_accessible": False,
            "correct_format": False,
            "metadata_loaded": False
        }
        
        try:
            with h5py.File(self.gctx_file, "r") as f:
                results["file_accessible"] = True
                
                # Check for required paths
                required = ["0/DATA/0/matrix", "0/META/ROW", "0/META/COL", "0/META/ROW/id", "0/META/COL/id"]
                results["correct_format"] = all(path in f for path in required)
                
            # Check metadata status
            results["metadata_loaded"] = (
                self._col_metadata is not None and 
                self._row_metadata is not None
            )
            
            return results
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return results

    def get_row_metadata(
        self,
        row_slice: Union[slice, list, None] = None,
        columns: Optional[Union[str, List[str]]] = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        # Return cached metadata if available
        if (
            self._row_metadata is not None
            and row_slice is None
            and columns is None
        ):
            return self._row_metadata
        
        # Determine if we need to manage the file handle ourselves
        need_to_open_file = self.f is None
        
        try:
            # Open file if not already open
            if need_to_open_file:
                f = h5py.File(self.gctx_file, "r")
            else:
                f = self.f
            
            # Load metadata from file
            row_selection = self._convert_to_indices(row_slice, self._n_rows)
            available_keys = list(f["0/META/ROW"].keys())
            columns = columns if columns else available_keys
            columns = [columns] if isinstance(columns, str) else columns
            
            # Validate column selection
            if not isinstance(columns, list):
                raise TypeError("columns must be None, a string, or a list of strings")
            missing_cols = [col for col in columns if col not in available_keys]
            if missing_cols:
                raise KeyError(f"Columns {missing_cols} not found in row metadata")
            
            # Extract metadata
            row_meta = {
                key: (
                    [
                        s.decode("utf-8", errors="ignore")
                        for s in f["0/META/ROW"][key][row_selection]
                    ]
                    if f["0/META/ROW"][key].dtype.kind == "S"
                    else f["0/META/ROW"][key][row_selection]
                )
                for key in columns
            }
            
            # Convert to DataFrame
            df = pd.DataFrame(row_meta)
            return df[columns[0]] if len(columns) == 1 else df
            
        finally:
            # Close file if we opened it
            if need_to_open_file and 'f' in locals():
                f.close()

    def get_column_metadata(
        self,
        col_slice: Union[slice, list, None] = None,
        columns: Optional[Union[str, List[str]]] = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        # Return cached metadata if available
        if self._col_metadata is not None and col_slice is None and columns is None:
            return self._col_metadata
        
        # Determine if we need to manage the file handle ourselves
        need_to_open_file = self.f is None
        
        try:
            # Open file if not already open
            if need_to_open_file:
                f = h5py.File(self.gctx_file, "r")
            else:
                f = self.f
            
            # Load metadata from file
            col_indices = self._convert_to_indices(col_slice, self._n_cols)
            available_keys = list(f["0/META/COL"].keys())
            columns = columns or available_keys
            columns = [columns] if isinstance(columns, str) else columns
            
            # Validate column selection
            if not isinstance(columns, list):
                raise TypeError("columns must be None, a string, or a list of strings")
            missing_cols = [col for col in columns if col not in available_keys]
            if missing_cols:
                raise KeyError(f"Columns {missing_cols} not found in column metadata")
            
            # Extract metadata
            col_meta = {
                key: (
                    [
                        s.decode("utf-8", errors="ignore")
                        for s in f["0/META/COL"][key][col_indices]
                    ]
                    if f["0/META/COL"][key].dtype.kind == "S"
                    else f["0/META/COL"][key][col_indices]
                )
                for key in columns
            }
            
            # Convert to DataFrame
            df = pd.DataFrame(col_meta)
            return df[columns[0]] if len(columns) == 1 else df
            
        finally:
            # Close file if we opened it
            if need_to_open_file and 'f' in locals():
                f.close()

    def get_column_metadata_for_feature_space(
        self,
        feature_space: Union[str, List[str]],
        columns: Optional[Union[str, List[str]]] = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        indices = self.get_gene_indices_for_feature_space(feature_space)
        return self.get_column_metadata(col_slice=indices, columns=columns)

    def get_metadata_attributes(self) -> Dict:
        """
        Get metadata attributes from the file.
        
        Returns:
            Dictionary of metadata attributes
        """
        if self._metadata_attrs is not None:
            return self._metadata_attrs
            
        # Load attributes if not already cached
        self._metadata_attrs = self._load_metadata_attributes()
        return self._metadata_attrs

    def get_data_with_metadata(
        self,
        row_slice: Union[slice, list, None] = None,
        col_slice: Union[slice, list, None] = None,
        feature_space: Optional[Union[str, List[str]]] = None,
        row_columns: Optional[Union[str, List[str]]] = None,
        col_columns: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        actual_col_slice = (
            col_slice or self.get_gene_indices_for_feature_space(feature_space)
            if feature_space
            else None
        )
        if col_slice is not None and feature_space is not None:
            raise ValueError("Cannot specify both col_slice and feature_space.")
        expression_data = self.get_expression_data(row_slice, actual_col_slice)
        row_metadata = self.get_row_metadata(row_slice, row_columns)
        col_metadata = self.get_column_metadata(actual_col_slice, col_columns)
        return expression_data, row_metadata, col_metadata