# src/data/loaders.py
import logging
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GCTXDataLoader:
    """
    Data loader for GCTX files containing gene expression data and metadata.
    Supports NumPy/Pandas-style slicing for rows and columns, metadata
    retrieval by column names, and gene selection by L1000 feature_space.

    Example usage:
    ```
    # Load expression data for landmark genes
    with GCTXDataLoader("data.gctx") as loader:
        # Get expression data for first 100 samples, landmark genes only
        expr_data = loader.get_expression_data(
            row_slice=slice(0, 100),
            feature_space="landmark"
        )

        # Get matching metadata
        meta = loader.get_row_metadata(row_slice=slice(0, 100))

        # Get both data and metadata at once
        expr_data, row_meta, col_meta = loader.get_data_with_metadata(
            row_slice=slice(0, 100),
            feature_space="landmark"
        )
    ```

    Attributes:
        gctx_file (str): Path to the .gctx file.
        f (h5py.File): HDF5 file handle (active within context manager).
        _n_rows (int): Number of rows in the expression matrix.
        _n_cols (int): Number of columns in the expression matrix.
        _col_metadata (pd.DataFrame): Preloaded column metadata for genes.
        _row_metadata_cache (pd.DataFrame): Cached row metadata (loaded on first access).
    """

    def __init__(self, gctx_file: str, preload_metadata: bool = True):
        """
        Initialize the GCTXDataLoader.

        Args:
            gctx_file: Path to the .gctx file.
            preload_metadata: Whether to preload column metadata at initialization.
        """
        self.gctx_file = gctx_file
        self.f = None
        self._n_rows, self._n_cols = None, None
        self._col_metadata = None
        self._row_metadata_cache = None
        self._metadata_attrs = None

        # Preload column metadata if requested (default)
        if preload_metadata:
            try:
                self._col_metadata = self._load_column_metadata()
                logger.debug(
                    f"Loaded column metadata with {len(self._col_metadata)} genes."
                )
            except Exception as e:
                logger.warning(f"Failed to preload column metadata: {str(e)}")

    def _load_column_metadata(self) -> pd.DataFrame:
        """Load column metadata from the GCTX file."""
        with h5py.File(self.gctx_file, "r") as f:
            try:
                meta_col_grp = f["0/META/COL"]
                col_dict = {}
                for key in meta_col_grp.keys():
                    data = meta_col_grp[key][:]
                    if data.dtype.kind == "S":
                        data = [s.decode("utf-8", errors="ignore") for s in data]
                    col_dict[key] = data
                return pd.DataFrame(col_dict)
            except Exception as e:
                logger.error(f"Error loading column metadata: {str(e)}")
                raise

    def __enter__(self):
        try:
            self.f = h5py.File(self.gctx_file, "r")
            self._n_rows, self._n_cols = self.f["0/DATA/0/matrix"].shape

            # Load column metadata if not already loaded
            if self._col_metadata is None:
                self._col_metadata = self._load_column_metadata()

            return self
        except Exception as e:
            logger.error(f"Error opening GCTX file: {str(e)}")
            if self.f is not None:
                self.f.close()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.f is not None:
            self.f.close()

    def _convert_to_indices(
        self, selection: Union[slice, list, None], max_size: int
    ) -> Union[slice, list]:
        """Convert selection to slice or list of indices."""
        if selection is None:
            return slice(None)
        elif isinstance(selection, slice):
            return selection
        elif isinstance(selection, list):
            if any(i < 0 or i >= max_size for i in selection):
                raise IndexError(f"Indices out of bounds (0 to {max_size-1})")
            return selection
        else:
            raise TypeError("Selection must be None, a slice, or a list of indices")

    def get_gene_indices_for_feature_space(
        self, feature_space: Union[str, List[str]]
    ) -> List[int]:
        """
        Get column indices for genes matching the specified L1000 feature_space.

        Args:
            feature_space: Single value (e.g., 'landmark') or list (e.g., ['landmark', 'best inferred']).

        Returns:
            List of integer indices for selected genes.
        """
        if self._col_metadata is None:
            self._col_metadata = self._load_column_metadata()

        if "feature_space" not in self._col_metadata.columns:
            raise ValueError("Column 'feature_space' not found in column metadata.")

        allowed_values = {"landmark", "best inferred", "inferred"}
        if isinstance(feature_space, str):
            if feature_space.lower() == "all":
                fs = list(allowed_values)
            else:
                fs = [feature_space]
                if fs[0] not in allowed_values:
                    raise ValueError(
                        f"Invalid feature_space: {feature_space}. Allowed: {allowed_values} or 'all'."
                    )
        elif isinstance(feature_space, list):
            if any(x.lower() == "all" for x in feature_space):
                fs = list(allowed_values)
            else:
                fs = feature_space
                for x in fs:
                    if x not in allowed_values:
                        raise ValueError(
                            f"Invalid feature_space value: {x}. Allowed: {allowed_values} or 'all'."
                        )
        else:
            raise ValueError("feature_space must be a string or a list of strings.")

        selected_genes = self._col_metadata[
            self._col_metadata["feature_space"].isin(fs)
        ]
        if selected_genes.empty:
            raise ValueError(f"No genes found for feature_space values: {fs}")
        indices = selected_genes.index.tolist()
        logger.debug(f"Selected {len(indices)} genes for feature_space {fs}")
        return indices

    def get_expression_data(
        self,
        row_slice: Union[slice, list, None] = None,
        col_slice: Union[slice, list, None] = None,
        feature_space: Optional[Union[str, List[str]]] = None,
    ) -> np.ndarray:
        """
        Load gene expression data, optionally filtered by feature_space.

        Args:
            row_slice: Rows to load (slice or list of indices).
            col_slice: Columns to load (slice or list of indices).
            feature_space: L1000 feature_space to filter genes (mutually exclusive with col_slice).

        Returns:
            Numpy array of shape (n_rows, n_cols) with expression data.
        """
        if col_slice is not None and feature_space is not None:
            raise ValueError("Cannot specify both col_slice and feature_space.")
        if feature_space is not None:
            col_slice = self.get_gene_indices_for_feature_space(feature_space)

        row_selection = self._convert_to_indices(row_slice, self._n_rows)
        col_selection = self._convert_to_indices(col_slice, self._n_cols)

        try:
            if isinstance(row_selection, list):
                unique_sorted_indices = np.unique(row_selection)
                data = self.f["0/DATA/0/matrix"][unique_sorted_indices, :]
                data = data[:, col_selection]  # Apply column selection
                index_map = {idx: i for i, idx in enumerate(unique_sorted_indices)}
                reordered_indices = [index_map[idx] for idx in row_selection]
                data = data[reordered_indices, :]
                logger.debug(
                    f"Loaded expression data with shape {data.shape} using list-based row selection"
                )
            else:
                data = self.f["0/DATA/0/matrix"][row_selection, col_selection]
                logger.debug(
                    f"Loaded expression data with shape {data.shape} using slice-based selection"
                )
            return data
        except Exception as e:
            logger.error(f"Error loading expression data: {str(e)}")
            raise

    def get_row_metadata(
        self,
        row_slice: Union[slice, list, None] = None,
        columns: Optional[Union[str, List[str]]] = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Load row metadata (experiment metadata).

        Args:
            row_slice: Rows to load (slice or list of indices).
            columns: Specific metadata columns to retrieve (default: all).

        Returns:
            DataFrame or Series with row metadata.
        """
        row_selection = self._convert_to_indices(row_slice, self._n_rows)

        try:
            available_keys = list(self.f["0/META/ROW"].keys())

            if columns is None:
                columns = available_keys
            elif isinstance(columns, str):
                columns = [columns]
            elif not isinstance(columns, list):
                raise TypeError("columns must be None, a string, or a list of strings")

            missing_cols = [col for col in columns if col not in available_keys]
            if missing_cols:
                raise KeyError(f"Columns {missing_cols} not found in row metadata")

            if isinstance(row_selection, list):
                unique_sorted_indices = np.unique(row_selection)
                row_meta = {}
                for key in columns:
                    dataset = self.f["0/META/ROW"][key]
                    data = dataset[unique_sorted_indices]
                    if dataset.dtype.kind == "S":
                        data = [s.decode("utf-8", errors="ignore") for s in data]
                    index_map = {idx: i for i, idx in enumerate(unique_sorted_indices)}
                    reordered_indices = [index_map[idx] for idx in row_selection]
                    row_meta[key] = np.array(data)[reordered_indices]
                df = pd.DataFrame(row_meta)
            else:
                row_meta = {}
                for key in columns:
                    dataset = self.f["0/META/ROW"][key]
                    data = dataset[row_selection]
                    if dataset.dtype.kind == "S":
                        data = [s.decode("utf-8", errors="ignore") for s in data]
                    row_meta[key] = data
                df = pd.DataFrame(row_meta)

            return df[columns[0]] if len(columns) == 1 else df
        except Exception as e:
            logger.error(f"Error loading row metadata: {str(e)}")
            raise

    def get_column_metadata(
        self,
        col_slice: Union[slice, list, None] = None,
        columns: Optional[Union[str, List[str]]] = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Load column metadata (gene metadata).

        Args:
            col_slice: Columns to load (slice or list of indices).
            columns: Specific metadata columns to retrieve (default: all).

        Returns:
            DataFrame or Series with column metadata.
        """
        # Use cached column metadata when possible
        if self._col_metadata is not None and col_slice is None:
            if columns is None:
                return self._col_metadata
            elif isinstance(columns, str):
                return self._col_metadata[columns]
            else:
                return self._col_metadata[columns]

        col_indices = self._convert_to_indices(col_slice, self._n_cols)

        try:
            available_keys = list(self.f["0/META/COL"].keys())

            if columns is None:
                columns = available_keys
            elif isinstance(columns, str):
                columns = [columns]
            elif not isinstance(columns, list):
                raise TypeError("columns must be None, a string, or a list of strings")

            missing_cols = [col for col in columns if col not in available_keys]
            if missing_cols:
                raise KeyError(f"Columns {missing_cols} not found in column metadata")

            col_meta = {}
            for key in columns:
                dataset = self.f["0/META/COL"][key]
                data = dataset[col_indices]
                if dataset.dtype.kind == "S":
                    data = [s.decode("utf-8", errors="ignore") for s in data]
                col_meta[key] = data

            df = pd.DataFrame(col_meta)
            return df[columns[0]] if len(columns) == 1 else df
        except Exception as e:
            logger.error(f"Error loading column metadata: {str(e)}")
            raise

    def get_column_metadata_for_feature_space(
        self,
        feature_space: Union[str, List[str]],
        columns: Optional[Union[str, List[str]]] = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Get column metadata for genes in the specified L1000 feature_space.

        Args:
            feature_space: Single value or list of feature_space categories.
            columns: Specific metadata columns to retrieve (default: all).

        Returns:
            DataFrame or Series with metadata for selected genes.
        """
        indices = self.get_gene_indices_for_feature_space(feature_space)
        return self.get_column_metadata(col_slice=indices, columns=columns)

    def get_metadata_attributes(self) -> Dict:
        """
        Load dataset-level metadata attributes.

        Returns:
            Dictionary of global metadata attributes.
        """
        # Use cached metadata attributes if available
        if self._metadata_attrs is not None:
            return self._metadata_attrs

        try:
            attrs = {}
            for key, value in self.f.attrs.items():
                if isinstance(value, bytes):
                    attrs[key] = value.decode("utf-8", errors="ignore")
                else:
                    attrs[key] = value
            self._metadata_attrs = attrs
            return attrs
        except Exception as e:
            logger.error(f"Error loading metadata attributes: {str(e)}")
            raise

    def get_data_with_metadata(
        self,
        row_slice: Union[slice, list, None] = None,
        col_slice: Union[slice, list, None] = None,
        feature_space: Optional[Union[str, List[str]]] = None,
        row_columns: Optional[Union[str, List[str]]] = None,
        col_columns: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Convenience method to get expression data and matching metadata in one call.

        Args:
            row_slice: Rows to load (slice or list of indices).
            col_slice: Columns to load (slice or list of indices).
            feature_space: L1000 feature_space to filter genes (mutually exclusive with col_slice).
            row_columns: Specific row metadata columns to retrieve (default: all).
            col_columns: Specific column metadata columns to retrieve (default: all).

        Returns:
            Tuple of (expression_data, row_metadata, column_metadata)
        """
        # Get the actual column indices if using feature_space
        actual_col_slice = col_slice
        if feature_space is not None:
            if col_slice is not None:
                raise ValueError("Cannot specify both col_slice and feature_space.")
            actual_col_slice = self.get_gene_indices_for_feature_space(feature_space)

        # Get expression data and matching metadata
        expression_data = self.get_expression_data(
            row_slice=row_slice, col_slice=actual_col_slice
        )
        row_metadata = self.get_row_metadata(row_slice=row_slice, columns=row_columns)
        col_metadata = self.get_column_metadata(
            col_slice=actual_col_slice, columns=col_columns
        )

        return expression_data, row_metadata, col_metadata
