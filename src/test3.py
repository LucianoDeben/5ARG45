import h5py
import numpy as np
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# File paths
bin_file = "../data/raw/X_RNA.bin"          # Gene expression data (31567 x 12328)
y_tsv_file = "../data/raw/Y.tsv"              # Row metadata (experiments)
compoundinfo_file = "../data/raw/compoundinfo.csv"  # Compound metadata (drug info)
geneinfo_file = "../data/raw/geneinfo_beta.txt"      # Gene metadata (columns)
output_h5_file = "../data/processed/LINCS.gctx"       # Output .gctx file

# Load gene expression data from .bin
logging.info("Loading gene expression data from binary file...")
X_rna = np.memmap(bin_file, dtype=np.float64, mode="r").reshape(31567, 12328)
assert X_rna.shape == (31567, 12328), f"Expected shape (31567,12328), got {X_rna.shape}"
logging.info(f"Gene expression data shape: {X_rna.shape}")
# Print a sample of the data
logging.info(f"Sample data:\n{X_rna[:20, :20]}")

# # Load and prepare row metadata
# logging.info("Loading row metadata from Y.tsv...")
# row_metadata = pd.read_csv(y_tsv_file, sep="\t")
# assert row_metadata.shape[0] == 31567, f"Expected 31567 rows before join, got {row_metadata.shape[0]}"

# logging.info("Loading compound info...")
# compound_info = pd.read_csv(compoundinfo_file)
# compound_info = compound_info[["pert_id", "cmap_name", "canonical_smiles"]]
# compound_info.set_index("pert_id", inplace=True)
# compound_info = compound_info[~compound_info.index.duplicated(keep='first')]
# logging.info(f"Compound info shape after dropping duplicates: {compound_info.shape}")

# row_metadata = pd.merge(
#     row_metadata,
#     compound_info,
#     left_on="pert_mfc_id",
#     right_index=True,
#     how="left"
# )
# assert X_rna.shape[0] == row_metadata.shape[0], (
#     f"Mismatch in rows: {X_rna.shape[0]} (X_rna) vs. {row_metadata.shape[0]} (row_metadata)"
# )

# # Drop rows with NaN in either metadata or expression data
# mask_metadata = row_metadata.notnull().all(axis=1).values
# mask_expr = ~np.any(np.isnan(X_rna), axis=1)
# final_mask = mask_metadata & mask_expr
# num_dropped = np.sum(~final_mask)
# logging.info(f"Dropping {num_dropped} rows due to NaN values.")
# row_metadata = row_metadata.loc[final_mask].copy()
# X_rna = X_rna[final_mask, :]

# # Process row and column metadata
# row_metadata.set_index("sig_id", inplace=True)
# row_ids = row_metadata.index.astype(str).tolist()
# logging.info(f"Final row metadata shape: {row_metadata.shape}")

# logging.info("Loading column metadata from geneinfo_beta.txt...")
# col_metadata = pd.read_csv(geneinfo_file, sep="\t")
# assert X_rna.shape[1] == col_metadata.shape[0], (
#     f"Mismatch in columns: {X_rna.shape[1]} (X_rna) vs. {col_metadata.shape[0]} (col_metadata)"
# )
# col_ids = col_metadata["gene_id"].astype(str).tolist()
# col_symbols = col_metadata["gene_symbol"].tolist()
# logging.info(f"Column metadata loaded with {len(col_ids)} entries.")


