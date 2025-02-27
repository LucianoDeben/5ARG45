import logging
from typing import Dict, List, Optional, Union

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GCTXDataLoader:
    """
    Data loader for GCTX files containing gene expression data and metadata.
    Supports NumPy/Pandas-style slicing for rows and columns (e.g., [1000:], [:500]),
    metadata retrieval by column names, and gene selection by L1000 feature_space.

    Attributes:
        gctx_file (str): Path to the .gctx file.
        f (h5py.File): HDF5 file handle (active within context manager).
        _n_rows (int): Number of rows in the expression matrix.
        _n_cols (int): Number of columns in the expression matrix.
        _col_metadata (pd.DataFrame): Preloaded column metadata for genes.
    """

    def __init__(self, gctx_file: str):
        """
        Initialize the GCTXDataLoader.

        Args:
            gctx_file: Path to the .gctx file.
        """
        self.gctx_file = gctx_file
        self.f = None
        self._n_rows, self._n_cols = None, None
        self._col_metadata = self._load_column_metadata()
        logger.debug(f"Loaded column metadata with {len(self._col_metadata)} genes.")

    def _load_column_metadata(self) -> pd.DataFrame:
        """Load column metadata from the GCTX file."""
        with h5py.File(self.gctx_file, "r") as f:
            meta_col_grp = f["0/META/COL"]
            col_dict = {}
            for key in meta_col_grp.keys():
                data = meta_col_grp[key][:]
                if data.dtype.kind == "S":
                    data = [s.decode("utf-8", errors="ignore") for s in data]
                col_dict[key] = data
            return pd.DataFrame(col_dict)

    def __enter__(self):
        self.f = h5py.File(self.gctx_file, "r")
        self._n_rows, self._n_cols = self.f["0/DATA/0/matrix"].shape
        return self

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

        if isinstance(row_selection, list):
            unique_sorted_indices = np.unique(row_selection)
            data = self.f["0/DATA/0/matrix"][unique_sorted_indices, col_selection]
            index_map = {idx: i for i, idx in enumerate(unique_sorted_indices)}
            reordered_indices = [index_map[idx] for idx in row_selection]
            data = data[reordered_indices]
            logger.debug(
                f"Loaded expression data with shape {data.shape} using list-based row selection"
            )
        else:
            data = self.f["0/DATA/0/matrix"][row_selection, col_selection]
            logger.debug(
                f"Loaded expression data with shape {data.shape} using slice-based selection"
            )
        return data

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
        col_indices = self._convert_to_indices(col_slice, self._n_cols)
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
        """Load dataset-level metadata attributes."""
        attrs = {}
        for key, value in self.f.attrs.items():
            if isinstance(value, bytes):
                attrs[key] = value.decode("utf-8", errors="ignore")
            else:
                attrs[key] = value
        return attrs
