import h5py
import numpy as np
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------
# File paths (update these as needed)
# ---------------------------
bin_file = "../data/raw/X_RNA.bin"          # float64 gene expression matrix (31567 x 12328)
y_tsv_file = "../data/raw/Y.tsv"              # row metadata (experiments)
compoundinfo_file = "../data/raw/compoundinfo.csv"  # compound metadata (drug info)
geneinfo_file = "../data/raw/geneinfo_beta.txt"      # gene metadata (columns)
output_h5_file = "../data/processed/LINCS.gctx"       # output file using .gctx convention

# ---------------------------
# Step 1: Load Gene Expression Data from .bin
# ---------------------------
logging.info("Loading gene expression data from binary file...")
X_rna = np.memmap(bin_file, dtype=np.float64, mode="r")
X_rna = X_rna.reshape(31567, 12328)
assert X_rna.shape == (31567, 12328), f"Expected shape (31567,12328), got {X_rna.shape}"
logging.info(f"Gene expression data shape: {X_rna.shape}")

# ---------------------------
# Step 2: Load and Prepare Row Metadata
# ---------------------------
logging.info("Loading row metadata from Y.tsv...")
row_metadata = pd.read_csv(y_tsv_file, sep="\t")
assert row_metadata.shape[0] == 31567, f"Expected 31567 rows before join, got {row_metadata.shape[0]}"

logging.info("Loading compound info...")
compound_info = pd.read_csv(compoundinfo_file)
# Select only desired columns (e.g., SMILES string info)
compound_info = compound_info[["pert_id", "cmap_name", "canonical_smiles"]]
compound_info.set_index("pert_id", inplace=True)
# Remove duplicate entries, keeping only the first instance for each compound.
compound_info = compound_info[~compound_info.index.duplicated(keep='first')]
logging.info(f"Compound info shape after dropping duplicates: {compound_info.shape}")

# Merge row metadata with compound info.
# Use a left join on "pert_mfc_id" (from row_metadata) matching compound_info's index.
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

# ---------------------------
# Step 2.1: Drop Rows with NaN in Expression Data or Metadata
# ---------------------------
# Create a mask for rows without any NaN in row_metadata.
mask_metadata = row_metadata.notnull().all(axis=1).values  # boolean array

# Create a mask for rows in X_rna without any NaN.
# (Note: This loads a boolean mask from the memmap; be cautious with very large datasets.)
mask_expr = ~np.any(np.isnan(X_rna), axis=1)

# Combine both masks.
final_mask = mask_metadata & mask_expr
num_dropped = np.sum(~final_mask)
logging.info(f"Dropping {num_dropped} rows due to NaN values.")

# Apply the mask to row_metadata and X_rna.
row_metadata = row_metadata.loc[final_mask].copy()
X_rna = X_rna[final_mask, :]

# ---------------------------
# Step 3: Process Row and Column IDs, and Column Metadata
# ---------------------------
# Set the index to "sig_id" so that the order aligns with the gene data.
row_metadata.set_index("sig_id", inplace=True)
# Do not re-sort; we want the original order preserved.
row_ids = row_metadata.index.astype(str).tolist()
logging.info(f"Final row metadata shape: {row_metadata.shape}")

logging.info("Loading column metadata from geneinfo_beta.txt...")
col_metadata = pd.read_csv(geneinfo_file, sep="\t")
assert X_rna.shape[1] == col_metadata.shape[0], (
    f"Mismatch in columns: {X_rna.shape[1]} (X_rna) vs. {col_metadata.shape[0]} (col_metadata)"
)

# We assume the gene metadata rows match the order of columns in X_rna.
col_ids = col_metadata["gene_id"].astype(str).tolist()
col_symbols = col_metadata["gene_symbol"].tolist()
logging.info(f"Column metadata loaded with {len(col_ids)} entries.")

# ---------------------------
# Step 4: Write to .gctx File
# ---------------------------
n_rows_out = len(row_ids)  # updated number of experiments after dropping NaNs
n_cols_out = X_rna.shape[1]  # number of genes (should remain 12328)
logging.info(f"Writing {n_rows_out} rows and {n_cols_out} columns to HDF5 file.")

# Define the output chunk shape for the gene expression data.
chunk_shape_out = (256, n_cols_out)

with h5py.File(output_h5_file, "w") as f_out:
    # Create the .gctx structure.
    # Data matrix at "/0/DATA/0/matrix"
    grp_data = f_out.create_group("0/DATA/0")
    dset = grp_data.create_dataset(
        "matrix",
        shape=(n_rows_out, n_cols_out),
        dtype="float32",  # Converting to float32 for efficiency.
        compression="gzip",
        chunks=chunk_shape_out,
    )

    # Write the gene expression data in chunks.
    chunk_rows_read = 16384   # number of rows to read at a time from X_rna
    out_row = 0
    for start in range(0, n_rows_out, chunk_rows_read):
        end = min(start + chunk_rows_read, n_rows_out)
        # IMPORTANT: Here we assume that the order in the .bin file corresponds exactly to the order in row_metadata.
        chunk_data = X_rna[start:end, :].astype("float32")
        dset[out_row:out_row + (end - start), :] = chunk_data
        out_row += (end - start)
    logging.info("Gene expression data written.")

    # Write Row Metadata in .gctx Format (/0/META/ROW)
    meta_row_grp = f_out.create_group("0/META/ROW")
    for col in row_metadata.columns:
        data_array = row_metadata[col].astype(str).values
        meta_row_grp.create_dataset(col, data=data_array.astype("S"))
    # Also store row IDs.
    meta_row_grp.create_dataset("id", data=np.array(row_ids, dtype="S"))
    logging.info("Row metadata written.")

    # Write Column Metadata in .gctx Format (/0/META/COL)
    meta_col_grp = f_out.create_group("0/META/COL")
    for col in col_metadata.columns:
        data_array = col_metadata[col].astype(str).values
        meta_col_grp.create_dataset(col, data=data_array.astype("S"))
    # Also store column IDs (using gene symbols).
    meta_col_grp.create_dataset("id", data=np.array(col_symbols, dtype="S"))
    logging.info("Column metadata written.")
    
    # Add attributes to the root group.
    f_out.attrs["version"] = "1.0"
    f_out.attrs["description"] = "LINCS gene expression data in .gctx format combined with CTRPv2 cell viability data."
    f_out.attrs["data_source"] = "CLUE.io, CTRPv2"
    f_out.attrs["creation_date"] = pd.Timestamp.now().isoformat()

print(f"Data and metadata successfully written to {output_h5_file} in .gctx format!")
print("Conversion complete.")
