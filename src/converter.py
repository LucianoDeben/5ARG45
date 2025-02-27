import h5py
import numpy as np
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

"""
Convert gene expression and metadata files into a .gctx file for LINCS data.
Inputs: Binary expression data, TSV row metadata, CSV compound info, TSV gene info.
Output: HDF5 file in .gctx format.
"""

# File paths
bin_file = "../data/raw/X_RNA.bin"          # Gene expression data (31567 x 12328)
y_tsv_file = "../data/raw/Y.tsv"              # Row metadata (experiments)
compoundinfo_file = "../data/raw/compoundinfo.csv"  # Compound metadata (drug info)
geneinfo_file = "../data/raw/geneinfo_beta.txt"      # Gene metadata (columns)
output_h5_file = "../data/processed/LINCS.gctx"       # Output .gctx file

# Load gene expression data from .bin
logging.info("Loading gene expression data from binary file...")
try:
    X_rna = np.memmap(bin_file, dtype=np.float64, mode="r").reshape(31567, 12328)
except FileNotFoundError:
    logging.error(f"File not found: {bin_file}")
    exit(1)
assert X_rna.shape == (31567, 12328), f"Expected shape (31567,12328), got {X_rna.shape}"
logging.info(f"Gene expression data shape: {X_rna.shape}")

# Load and prepare row metadata
logging.info("Loading row metadata from Y.tsv...")
row_metadata = pd.read_csv(y_tsv_file, sep="\t")
assert row_metadata.shape[0] == 31567, f"Expected 31567 rows before join, got {row_metadata.shape[0]}"

logging.info("Loading compound info...")
compound_info = pd.read_csv(compoundinfo_file)
compound_info = compound_info[["pert_id", "cmap_name", "canonical_smiles"]]
compound_info.set_index("pert_id", inplace=True)
compound_info = compound_info[~compound_info.index.duplicated(keep='first')]
logging.info(f"Compound info shape after dropping duplicates: {compound_info.shape}")

row_metadata = pd.merge(
    row_metadata,
    compound_info,
    left_on="pert_mfc_id",
    right_index=True,
    how="left"
)
assert X_rna.shape[0] == row_metadata.shape[0], (
    f"Mismatch in rows: {X_rna.shape[0]} (X_rna) vs. {row_metadata.shape[0]} (row_metadata)"
)

# Drop rows with NaN in either metadata or expression data
mask_metadata = row_metadata.notnull().all(axis=1).values
mask_expr = ~np.any(np.isnan(X_rna), axis=1)
final_mask = mask_metadata & mask_expr
num_dropped = np.sum(~final_mask)
logging.info(f"Dropping {num_dropped} rows due to NaN values.")
row_metadata = row_metadata.loc[final_mask].copy()
X_rna = X_rna[final_mask, :]

# Process row and column metadata
row_metadata.set_index("sig_id", inplace=True)
row_ids = row_metadata.index.astype(str).tolist()
logging.info(f"Final row metadata shape: {row_metadata.shape}")

logging.info("Loading column metadata from geneinfo_beta.txt...")
col_metadata = pd.read_csv(geneinfo_file, sep="\t")
assert X_rna.shape[1] == col_metadata.shape[0], (
    f"Mismatch in columns: {X_rna.shape[1]} (X_rna) vs. {col_metadata.shape[0]} (col_metadata)"
)
col_ids = col_metadata["gene_id"].astype(str).tolist()
col_symbols = col_metadata["gene_symbol"].tolist()
logging.info(f"Column metadata loaded with {len(col_ids)} entries.")

# Write to .gctx file
n_rows_out = len(row_ids)
n_cols_out = X_rna.shape[1]
logging.info(f"Writing {n_rows_out} rows and {n_cols_out} columns to HDF5 file.")
chunk_shape_out = (256, n_cols_out)

with h5py.File(output_h5_file, "w") as f_out:
    grp_data = f_out.create_group("0/DATA/0")
    dset = grp_data.create_dataset(
        "matrix",
        shape=(n_rows_out, n_cols_out),
        dtype="float32",
        compression="gzip",
        chunks=chunk_shape_out,
    )
    chunk_rows_read = 16384
    out_row = 0
    for start in range(0, n_rows_out, chunk_rows_read):
        end = min(start + chunk_rows_read, n_rows_out)
        chunk_data = X_rna[start:end, :].astype("float32")
        dset[out_row:out_row + (end - start), :] = chunk_data
        out_row += (end - start)
    logging.info("Gene expression data written.")

    # Writing row metadata
    meta_row_grp = f_out.create_group("0/META/ROW")
    for col in row_metadata.columns:
        data = row_metadata[col].values
        if pd.api.types.is_numeric_dtype(row_metadata[col]):
            # Store numeric data as-is (int, float)
            meta_row_grp.create_dataset(col, data=data)
        else:
            # Store non-numeric data (strings, objects) as bytes
            meta_row_grp.create_dataset(col, data=data.astype(str).astype("S"))
    meta_row_grp.create_dataset("id", data=np.array(row_ids, dtype="S"))
    logging.info("Row metadata written.")

    # Writing column metadata
    meta_col_grp = f_out.create_group("0/META/COL")
    for col in col_metadata.columns:
        data = col_metadata[col].values
        if pd.api.types.is_numeric_dtype(col_metadata[col]):
            # Store numeric data as-is (int, float)
            meta_col_grp.create_dataset(col, data=data)
        else:
            # Store non-numeric data (strings, objects) as bytes
            meta_col_grp.create_dataset(col, data=data.astype(str).astype("S"))
    meta_col_grp.create_dataset("id", data=np.array(col_symbols, dtype="S"))
    logging.info("Column metadata written.")
    
    f_out.attrs["version"] = "1.0"
    f_out.attrs["description"] = "LINCS gene expression data in .gctx format combined with CTRPv2 cell viability data."
    f_out.attrs["data_source"] = "CLUE.io, CTRPv2"
    f_out.attrs["creation_date"] = pd.Timestamp.now().isoformat()

logging.info(f"Data and metadata successfully written to {output_h5_file} in .gctx format!")
print("Conversion complete.")
