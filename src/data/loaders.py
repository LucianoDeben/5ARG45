import h5py
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List
import logging

logger = logging.getLogger(__name__)

class GCTXDataLoader:
    """
    Data loader for GCTX files containing gene expression data and metadata.
    Supports NumPy/Pandas-style slicing for rows and columns (e.g., [1000:], [:500]),
    and metadata retrieval by column names.
    """
    def __init__(self, gctx_file: str):
        self.gctx_file = gctx_file
        self.f = None
        self._n_rows, self._n_cols = None, None

    def __enter__(self):
        self.f = h5py.File(self.gctx_file, 'r')
        self._n_rows, self._n_cols = self.f["0/DATA/0/matrix"].shape
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.f is not None:
            self.f.close()

    def _convert_to_indices(self, selection: Union[slice, list, None], max_size: int) -> Union[slice, list]:
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

    def get_expression_data(self, row_slice: Union[slice, list, None] = None, col_slice: Union[slice, list, None] = None) -> np.ndarray:
        row_selection = self._convert_to_indices(row_slice, self._n_rows)
        col_selection = self._convert_to_indices(col_slice, self._n_cols)

        if isinstance(row_selection, list):
            unique_sorted_indices = np.unique(row_selection)  # Unique and sorted for h5py
            data = self.f["0/DATA/0/matrix"][unique_sorted_indices, col_selection]
            index_map = {idx: i for i, idx in enumerate(unique_sorted_indices)}
            reordered_indices = [index_map[idx] for idx in row_selection]
            data = data[reordered_indices]
        else:
            data = self.f["0/DATA/0/matrix"][row_selection, col_selection]

        return data

    def get_row_metadata(self, row_slice: Union[slice, list, None] = None, columns: Optional[Union[str, List[str]]] = None) -> Union[pd.DataFrame, pd.Series]:
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
                if dataset.dtype.kind == 'S':
                    data = [s.decode('utf-8', errors='ignore') for s in data]
                index_map = {idx: i for i, idx in enumerate(unique_sorted_indices)}
                reordered_indices = [index_map[idx] for idx in row_selection]
                row_meta[key] = np.array(data)[reordered_indices]
            df = pd.DataFrame(row_meta)
        else:
            row_meta = {}
            for key in columns:
                dataset = self.f["0/META/ROW"][key]
                data = dataset[row_selection]
                if dataset.dtype.kind == 'S':
                    data = [s.decode('utf-8', errors='ignore') for s in data]
                row_meta[key] = data
            df = pd.DataFrame(row_meta)

        return df[columns[0]] if len(columns) == 1 else df

    def get_column_metadata(self, col_slice: Union[slice, list, None] = None, columns: Optional[Union[str, List[str]]] = None) -> Union[pd.DataFrame, pd.Series]:
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
            if dataset.dtype.kind == 'S':
                data = [s.decode('utf-8', errors='ignore') for s in data]
            col_meta[key] = data

        df = pd.DataFrame(col_meta)
        return df[columns[0]] if len(columns) == 1 else df

    def get_metadata_attributes(self) -> Dict:
        attrs = {}
        for key, value in self.f.attrs.items():
            if isinstance(value, bytes):
                attrs[key] = value.decode('utf-8', errors='ignore')
            else:
                attrs[key] = value
        return attrs