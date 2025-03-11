import h5py
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def update_gctx_file(input_file, output_file):
    """
    Update a GCTX file with modified column names and clipped viability values.
    
    Args:
        input_file: Path to the input GCTX file
        output_file: Path to the output GCTX file
    """
    logger.info(f"Processing file: {input_file}")
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read the input file
    with h5py.File(input_file, "r") as f_in:
        # Get dimensions
        matrix = f_in["0/DATA/0/matrix"]
        n_rows, n_cols = matrix.shape
        
        logger.info(f"GCTX file dimensions: {n_rows} rows, {n_cols} columns")
        
        # Create output file
        with h5py.File(output_file, "w") as f_out:
            # Copy the expression data
            grp_data_out = f_out.create_group("0/DATA/0")
            dset_out = grp_data_out.create_dataset(
                "matrix",
                shape=(n_rows, n_cols),
                dtype="float32",
                compression="gzip",
                chunks=(256, n_cols)
            )
            
            # Copy data in chunks
            chunk_size = 1000
            for i in range(0, n_rows, chunk_size):
                end = min(i + chunk_size, n_rows)
                dset_out[i:end, :] = matrix[i:end, :]
            
            logger.info("Expression data copied")
            
            # Copy column metadata
            meta_col_grp_in = f_in["0/META/COL"]
            meta_col_grp_out = f_out.create_group("0/META/COL")
            
            for key in meta_col_grp_in.keys():
                data = meta_col_grp_in[key][:]
                if data.dtype.kind == 'S':
                    meta_col_grp_out.create_dataset(key, data=data)
                else:
                    meta_col_grp_out.create_dataset(key, data=data)
            
            logger.info("Column metadata copied")
            
            # Modify row metadata
            meta_row_grp_in = f_in["0/META/ROW"]
            meta_row_grp_out = f_out.create_group("0/META/ROW")
            
            # Process all row metadata
            for key in meta_row_grp_in.keys():
                data = meta_row_grp_in[key][:]
                
                # Skip the keys we're going to modify
                if key in ["viability", "pert_dose"]:
                    continue
                
                if data.dtype.kind == 'S':
                    meta_row_grp_out.create_dataset(key, data=data)
                else:
                    meta_row_grp_out.create_dataset(key, data=data)
            
            # Rename and clip viability if it exists
            if "viability" in meta_row_grp_in:
                viability_data = meta_row_grp_in["viability"][:]
                # Clip values between 0 and 1
                viability_clipped = np.clip(viability_data, 0, 1)
                meta_row_grp_out.create_dataset("viability_clipped", data=viability_clipped)
                logger.info(f"Renamed 'viability' to 'viability_clipped' and clipped values")
            
            # Rename pert_dose if it exists
            if "pert_dose" in meta_row_grp_in:
                pert_dose_data = meta_row_grp_in["pert_dose"][:]
                meta_row_grp_out.create_dataset("pert_dose_log2", data=pert_dose_data)
                logger.info(f"Renamed 'pert_dose' to 'pert_dose_log2'")
            
            # Copy file attributes
            for key, value in f_in.attrs.items():
                f_out.attrs[key] = value
            
            # Add an update note
            f_out.attrs["update_note"] = "Updated column names: 'viability' to 'viability_clipped' (clipped 0-1), 'pert_dose' to 'pert_dose_log2'"
    
    logger.info(f"File updated and saved to {output_file}")

if __name__ == "__main__":
    input_file = "data/processed/LINCS_small.gctx"
    output_file = "data/processed/LINCS_small_updated.gctx"
    
    update_gctx_file(input_file, output_file)
    
    # Verify the changes
    logger.info("Verifying changes...")
    with h5py.File(output_file, "r") as f:
        # Check the row metadata keys
        row_keys = list(f["0/META/ROW"].keys())
        logger.info(f"Row metadata keys: {row_keys}")
        
        # Verify viability_clipped exists and is clipped
        if "viability_clipped" in row_keys:
            viability = f["0/META/ROW/viability_clipped"][:]
            min_val = np.min(viability)
            max_val = np.max(viability)
            logger.info(f"viability_clipped range: [{min_val}, {max_val}]")
            
        # Verify pert_dose_log2 exists
        if "pert_dose_log2" in row_keys:
            logger.info("pert_dose_log2 found in row metadata")
    
    logger.info("Verification complete!")