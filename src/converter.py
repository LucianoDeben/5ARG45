import h5py
import numpy as np
import pandas as pd
import os

# Example file paths – update these as needed.
gctx_file = "../data/raw/level3_beta_trt_cp_n1805898x12328.gctx"
h5_file = "../data/processed/compound_perturbation_dataset.h5"
geneinfo_file = "../data/raw/geneinfo_beta.txt"
pert_info_file = "../data/raw/compound_perturbation_metadata.txt"

# ---------------------------
# Step 1: Read and filter row metadata
# ---------------------------

# Open the .gctx file and extract the row IDs.
with h5py.File(gctx_file, "r") as f_in:
    # Here, we assume the row metadata is stored at the path "/0/META/COL/id"
    row_ids = f_in["/0/META/COL/id"][:].astype(str)

# Load external metadata for genes and perturbations.
gene_metadata = pd.read_csv(geneinfo_file, sep="\t")
pert_info = pd.read_csv(pert_info_file, sep="\t")

# Filter perturbation metadata: for example, keep only samples that pass QC and of a given type.
filtered_pert_info = pert_info[(pert_info["qc_pass"] == 1.0) & (pert_info["pert_type"] == "trt_cp")].copy()
filtered_pert_info.set_index("sample_id", inplace=True)

# Determine which row IDs (samples) from the gctx file are present in the filtered perturbation metadata.
subset_row_ids = pd.Index(row_ids).intersection(filtered_pert_info.index)
# Get the subset of metadata corresponding to the kept row IDs.
subset_pert_info = filtered_pert_info.loc[subset_row_ids].copy()
subset_pert_info.sort_index(inplace=True)
subset_pert_info["sample_id"] = subset_pert_info.index

# Map each row ID from the gctx file to its index.
row_idx_map = {rid: idx for idx, rid in enumerate(row_ids)}
# Build a list of row indices (from the gctx file) that correspond to our filtered samples.
filtered_row_indices = [row_idx_map[sid] for sid in subset_pert_info.index]

# ---------------------------
# Step 2: Determine Column Subset (or Use All Columns)
# ---------------------------

# In your gctx, assume the column metadata (gene IDs) is stored at "/0/META/ROW/id".
with h5py.File(gctx_file, "r") as f_in:
    col_ids_all = f_in["/0/META/ROW/id"][:].astype(int)

# Create a mapping from gene ID to gene symbol using gene_metadata.
gene_mapping = gene_metadata.set_index("gene_id")["gene_symbol"].to_dict()
# Create a list of gene symbols corresponding to each column.
col_symbols_all = [gene_mapping.get(gid, f"Gene_{gid}") for gid in col_ids_all]

# Define which columns to include.
# Here we take all columns. To use a subset (e.g., landmark genes), define a subset of indices.
col_indices = np.arange(len(col_ids_all))

# ---------------------------
# Step 3: Create the Output HDF5 Dataset in Chunks
# ---------------------------

# Define how many rows to read at a time from the .gctx file.
chunk_rows_read = 50000  # Adjust this based on your memory constraints.

# Determine the final shape of the output:
n_rows_out = len(filtered_row_indices)  # Only keeping filtered rows.
n_cols_out = len(col_indices)           # Number of columns chosen.

# Define an appropriate chunk shape for the output HDF5 dataset.
chunk_shape_out = (128, n_cols_out)

# Open the input and output files.
with h5py.File(gctx_file, "r") as f_in, h5py.File(h5_file, "w") as f_out:
    # Access the large gene expression matrix in the gctx file.
    data_in = f_in["/0/DATA/0/matrix"]
    
    # Create an output dataset in HDF5 for the gene expression matrix with compression.
    dset_out = f_out.create_dataset(
        "data",
        shape=(n_rows_out, n_cols_out),
        dtype="float32",
        compression="gzip",
        chunks=chunk_shape_out,
    )

    # Keep track of the current row in the output.
    out_row_start = 0
    total_rows = n_rows_out

    # Process the data in chunks.
    for start_idx in range(0, total_rows, chunk_rows_read):
        end_idx = min(start_idx + chunk_rows_read, total_rows)
        chunk_size = end_idx - start_idx
        
        # Get the corresponding global row indices for this chunk.
        row_subset = filtered_row_indices[start_idx:end_idx]
        
        # Read the chunk from the input file: first select the rows, then the columns.
        chunk_data = data_in[row_subset, :][:, col_indices]
        
        # Write the chunk data to the output dataset.
        dset_out[out_row_start:out_row_start + chunk_size, :] = chunk_data
        out_row_start += chunk_size

    # ---------------------------
    # Step 4: Write Metadata into the HDF5 File
    # ---------------------------
    
    # Write row-level metadata. Each metadata field is saved as a separate dataset.
    for column in subset_pert_info.columns:
        f_out.create_dataset(
            f"row_metadata/{column}",
            data=subset_pert_info[column].values.astype("S"),
        )
    
    # Write column-level metadata.
    # You can store the full gene metadata table or only selected columns.
    for col_name in gene_metadata.columns:
        f_out.create_dataset(
            f"col_metadata/{col_name}",
            data=gene_metadata[col_name].values.astype("S"),
        )
    
    # Write the final row IDs and column symbols.
    f_out.create_dataset("row_ids", data=subset_pert_info.index.values.astype("S"))
    selected_col_symbols = [col_symbols_all[i] for i in col_indices]
    f_out.create_dataset("col_ids", data=[s.encode("utf-8") for s in selected_col_symbols])

print(f"Filtered data and metadata successfully written to {h5_file} in chunks!")
