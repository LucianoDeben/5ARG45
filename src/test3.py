import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import os
import logging

from data_sets import LINCSDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# File paths
bin_file = "../data/raw/X_RNA.bin"          # Gene expression data (31567 x 12328)
y_tsv_file = "../data/raw/Y.tsv"              # Row metadata (experiments)
compoundinfo_file = "../data/raw/compoundinfo.csv"  # Compound metadata (drug info)
geneinfo_file = "../data/raw/geneinfo_beta.txt"      # Gene metadata (columns)
output_h5_file = "../data/processed/LINCS.gctx"       # Output .gctx file

# Load gene expression data from .bin
logging.info("Loading gene expression data from binary file...")

dataset = LINCSDataset(
    gctx_path="../data/processed/LINCS.gctx",
    normalize="z-score",
    feature_space="all"
)

X_rna = dataset.get_expression_matrix()
# gctx_path = "../data/processed/LINCS.gctx"  
# with h5py.File(gctx_path, "r") as f:
#     X_rna = f["0/DATA/0/matrix"][:].astype(np.float32)

X_rna = scale(X_rna, copy=False)

# Load gene expression data from .bin
logging.info("Loading gene expression data from binary file...")
X_rna2 = np.memmap(bin_file, dtype=np.float64, mode="r").reshape(31567, 12328)
assert X_rna2.shape == (31567, 12328), f"Expected shape (31567,12328), got {X_rna2.shape}"
logging.info(f"Gene expression data shape: {X_rna2.shape}")

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

# Drop rows with NaN in either metadata or expression data
mask_metadata = row_metadata.notnull().all(axis=1).values
mask_expr = ~np.any(np.isnan(X_rna2), axis=1)
final_mask = mask_metadata & mask_expr
num_dropped = np.sum(~final_mask)
logging.info(f"Dropping {num_dropped} rows due to NaN values.")
row_metadata = row_metadata.loc[final_mask].copy()
X_rna2 = X_rna2[final_mask, :]



# Check wherer loading from memmap and Lincsdataset are equal
logging.info(f"Gene expression data shape: {X_rna2.shape}")
# Print a sample of the data
X_rna2 = scale(X_rna2, copy=False)
logging.info(f"Sample data:\n{X_rna2[:20, :20]}")
assert np.allclose(X_rna[:20, :20], X_rna2[:20, :20]), "Data loaded from memmap and fromfile are not equal after scaling."




