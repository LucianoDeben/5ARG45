# src/data/data_validation.py
import logging

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Update the path to your generated .gctx file.
h5_file = "../data/processed/LINCS2.gctx"

with h5py.File(h5_file, "r") as f:
    # 1. Check that expected groups exist.
    expected_groups = ["/0/DATA/0", "/0/META/ROW", "/0/META/COL"]
    for grp in expected_groups:
        assert grp in f, f"Expected group '{grp}' not found in file."
        logger.info(f"Group '{grp}' found.")
    # 2. Check gene expression data dimensions and type.
    data_dset = f["/0/DATA/0/matrix"]
    data_shape = data_dset.shape
    data_dtype = data_dset.dtype
    logger.info(f"Data shape: {data_shape}, dtype: {data_dtype}")
    assert data_shape[1] == 12328, f"Expected 12328 columns, got {data_shape[1]}"
    # Optionally, check a subset of rows (if you have expected values):
    sample_data = data_dset[0:5, :]
    logger.info(f"Sample data (first 5 rows):\n{sample_data}")
    # 3. Check row metadata.
    # Row metadata (experiments) is now stored in /0/META/ROW
    meta_row_grp = f["/0/META/ROW"]
    row_ids = meta_row_grp["id"][:]
    logger.info(f"Number of row IDs: {len(row_ids)}")
    # (Assume you expect 31567 row IDs.)
    assert (
        len(row_ids) == data_shape[0]
    ), f"Number of row IDs ({len(row_ids)}) does not match number of rows in data ({data_shape[0]})."

    # 4. Check column metadata.
    # Column metadata (genes) is now stored in /0/META/COL
    meta_col_grp = f["/0/META/COL"]
    col_ids = meta_col_grp["id"][:]
    logger.info(f"Number of column IDs: {len(col_ids)}")
    # (Assume you expect 12328 column IDs.)
    assert (
        len(col_ids) == data_shape[1]
    ), f"Number of column IDs ({len(col_ids)}) does not match number of columns in data ({data_shape[1]})."

    # 5. (Optional) Load a small portion of the metadata as a DataFrame to visually inspect.
    # For example, load row metadata from one field (e.g., 'cell_mfc_name') from /0/META/ROW.
    row_meta_field = meta_row_grp["cell_mfc_name"][
        :
    ]  # Adjust the field name as needed.
    logger.info(f"First 5 values in 'cell_mfc_name': {row_meta_field[:5]}")

    # 6. Verify that the data is stored as float32.
    assert (
        data_dtype == np.float32
    ), f"Data dtype expected to be float32, got {data_dtype}"

logger.info("All checks passed successfully!")
