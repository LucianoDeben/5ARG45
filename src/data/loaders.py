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

    def __init__(self, gctx_file: str, preload_metadata: bool = True):
        self.gctx_file = gctx_file
        self.f = None
        self._n_rows, self._n_cols = None, None
        self._col_metadata = None
        self._row_metadata_cache = None
        self._metadata_attrs = None
        if preload_metadata:
            try:
                self._col_metadata = self._load_column_metadata()
                logger.debug(
                    f"Loaded column metadata with {len(self._col_metadata)} genes."
                )
            except Exception as e:
                logger.warning(f"Failed to preload column metadata: {str(e)}")

    def _load_column_metadata(self) -> pd.DataFrame:
        with h5py.File(self.gctx_file, "r") as f:
            meta_col_grp = f["0/META/COL"]
            col_dict = {key: meta_col_grp[key][:] for key in meta_col_grp.keys()}
            for key, data in col_dict.items():
                if data.dtype.kind == "S":
                    col_dict[key] = [s.decode("utf-8", errors="ignore") for s in data]
            return pd.DataFrame(col_dict)

    def __enter__(self):
        self.f = h5py.File(self.gctx_file, "r")
        required = ["0/DATA/0/matrix", "0/META/ROW", "0/META/COL"]
        for path in required:
            if path not in self.f:
                raise ValueError(f"Invalid .gctx file: missing {path}")
        self._n_rows, self._n_cols = self.f["0/DATA/0/matrix"].shape
        if self._col_metadata is None:
            self._col_metadata = self._load_column_metadata()
        if self._row_metadata_cache is None:
            self._row_metadata_cache = self.get_row_metadata()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.f is not None:
            self.f.close()

    def _convert_to_indices(
        self, selection: Union[slice, list, None], max_size: int
    ) -> Union[slice, list]:
        if selection is None:
            return slice(None)
        elif isinstance(selection, slice):
            return selection
        elif isinstance(selection, list):
            if any(i < 0 or i >= max_size for i in selection):
                raise IndexError(f"Indices out of bounds (0 to {max_size-1})")
            return selection
        raise TypeError("Selection must be None, a slice, or a list of indices")

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
        row_slice: Union[slice, list, None] = None,
        col_slice: Union[slice, list, None] = None,
        feature_space: Optional[Union[str, List[str]]] = None,
        chunk_size: int = 10000,
    ) -> np.ndarray:
        if col_slice is not None and feature_space is not None:
            raise ValueError("Cannot specify both col_slice and feature_space.")
        row_selection = self._convert_to_indices(row_slice, self._n_rows)
        col_selection = self._convert_to_indices(
            (
                col_slice or self.get_gene_indices_for_feature_space(feature_space)
                if feature_space
                else None
            ),
            self._n_cols,
        )
        if isinstance(row_selection, list):
            chunks = [
                row_selection[i : i + chunk_size]
                for i in range(0, len(row_selection), chunk_size)
            ]
            data = [
                self.f["0/DATA/0/matrix"][chunk, col_selection]
                for chunk in tqdm(chunks, desc="Loading expression chunks")
            ]
            return np.concatenate(data, axis=0)
        return self.f["0/DATA/0/matrix"][row_selection, col_selection]

    def get_row_metadata(
        self,
        row_slice: Union[slice, list, None] = None,
        columns: Optional[Union[str, List[str]]] = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        if (
            self._row_metadata_cache is not None
            and row_slice is None
            and columns is None
        ):
            return self._row_metadata_cache
        row_selection = self._convert_to_indices(row_slice, self._n_rows)
        available_keys = list(self.f["0/META/ROW"].keys())
        columns = columns if columns else available_keys
        columns = [columns] if isinstance(columns, str) else columns
        if not isinstance(columns, list):
            raise TypeError("columns must be None, a string, or a list of strings")
        missing_cols = [col for col in columns if col not in available_keys]
        if missing_cols:
            raise KeyError(f"Columns {missing_cols} not found in row metadata")
        row_meta = {
            key: (
                [
                    s.decode("utf-8", errors="ignore")
                    for s in self.f["0/META/ROW"][key][row_selection]
                ]
                if self.f["0/META/ROW"][key].dtype.kind == "S"
                else self.f["0/META/ROW"][key][row_selection]
            )
            for key in columns
        }
        df = pd.DataFrame(row_meta)
        return df[columns[0]] if len(columns) == 1 else df

    def get_column_metadata(
        self,
        col_slice: Union[slice, list, None] = None,
        columns: Optional[Union[str, List[str]]] = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        if self._col_metadata is not None and col_slice is None and columns is None:
            return self._col_metadata
        col_indices = self._convert_to_indices(col_slice, self._n_cols)
        available_keys = list(self.f["0/META/COL"].keys())
        columns = columns or available_keys
        columns = [columns] if isinstance(columns, str) else columns
        if not isinstance(columns, list):
            raise TypeError("columns must be None, a string, or a list of strings")
        missing_cols = [col for col in columns if col not in available_keys]
        if missing_cols:
            raise KeyError(f"Columns {missing_cols} not found in column metadata")
        col_meta = {
            key: (
                [
                    s.decode("utf-8", errors="ignore")
                    for s in self.f["0/META/COL"][key][col_indices]
                ]
                if self.f["0/META/COL"][key].dtype.kind == "S"
                else self.f["0/META/COL"][key][col_indices]
            )
            for key in columns
        }
        df = pd.DataFrame(col_meta)
        return df[columns[0]] if len(columns) == 1 else df

    def get_column_metadata_for_feature_space(
        self,
        feature_space: Union[str, List[str]],
        columns: Optional[Union[str, List[str]]] = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        indices = self.get_gene_indices_for_feature_space(feature_space)
        return self.get_column_metadata(col_slice=indices, columns=columns)

    def get_metadata_attributes(self) -> Dict:
        if self._metadata_attrs is not None:
            return self._metadata_attrs
        attrs = {
            key: (
                value.decode("utf-8", errors="ignore")
                if isinstance(value, bytes)
                else value
            )
            for key, value in self.f.attrs.items()
        }
        self._metadata_attrs = attrs
        return attrs

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
