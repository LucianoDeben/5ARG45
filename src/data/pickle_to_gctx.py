import h5py
import numpy as np
import pandas as pd
import logging
import os
import warnings
import json
from tqdm.auto import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def match_lincs_ctrp_to_gctx(output_gctx_file, env_file="environment.json"):
    """
    Create a GCTX file directly from LINCS and CTRP data sources.
    Includes ALL genes and necessary metadata.
    
    Args:
        output_gctx_file: Path where to save the GCTX file
        env_file: Path to environment.json file with file paths
    """
    logger.info("Starting LINCS-CTRP matching and GCTX creation process")
    
    # Load environment config
    with open(env_file) as handle:
        env = json.load(handle)
        logger.info(f"Environment '{env['name']}' loaded")
    
    # Create output directories
    os.makedirs(os.path.dirname(output_gctx_file), exist_ok=True)
    
    # Suppress warnings for data loading
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Load gene info
        logger.info("Loading gene info...")
        geneinfo = pd.read_csv(env['geneinfo'], sep="\t", low_memory=False)
        geneinfo_df = geneinfo.set_index('gene_id')
        
        # Load signature info
        logger.info("Loading signature info...")
        siginfo = pd.read_csv(env['siginfo'], sep="\t", low_memory=False)
        
        # Keep only 24 hour timepoint
        siginfo_df = siginfo.loc[siginfo.pert_time == 24.0]
        logger.info(f"After filtering for 24h timepoint: {len(siginfo_df)} signatures")
        
        # Keep only signatures with filled in perturbation dosage
        siginfo_df = siginfo_df.loc[~(siginfo_df.pert_dose.isna())]
        logger.info(f"After filtering NaN dosage: {len(siginfo_df)} signatures")
        
        # Reset the index
        siginfo_df.reset_index(drop=True, inplace=True)
        
        # Loading GCTX file
        logger.info("Loading GCTX file...")
        with h5py.File(env["gctx_file"], 'r') as f:
            # Get the dimensions
            matrix = f['0']['DATA']['0']['matrix']
            matrix_shape = matrix.shape
            logger.info(f"Original GCTX matrix dimensions: {matrix_shape}")
            
            # Signature ids (in gctx order)
            sig_ids = pd.Series(f['0']['META']['COL']['id']).astype(str)
            logger.info(f"Total signatures in GCTX: {len(sig_ids)}")
            
            # Gene ids (in gctx order)
            gene_ids = np.array(f['0']['META']['ROW']['id']).astype(int)
            
            # Gene symbols (in gctx order)
            gene_symbols = geneinfo_df.loc[gene_ids].gene_symbol.tolist()
            
            # Keep track of all feature spaces
            feature_spaces = geneinfo_df.loc[gene_ids].feature_space.tolist()
            
        # Loading CTRP data
        logger.info("Loading CTRP data...")
        per_cpd_post_qc = pd.read_csv(env["per_cpd_post_qc"], sep="\t")
        curves_df = pd.read_csv(env["curves_post_qc"], sep="\t")
        per_compound = pd.read_csv(env["per_compound"], sep="\t")
        per_cell_line = pd.read_csv(env["per_cell_line"], sep="\t")
        per_experiment = pd.read_csv(env["per_experiment"], sep="\t")
    
    # Define the sig3upper function with error handling
    def sig3upper(p, X):
        """
        This is the curve implement from the CTRP provided Matlab script.
        Added error handling for numeric overflow.
        """
        alpha = p[0]      # EC50
        beta = p[1]       # slope
        b = p[2]          # lower limit
        
        # Handle extreme values to prevent overflow
        exp_term = np.zeros_like(X)
        valid_mask = np.abs(X-alpha) < 700*np.abs(beta)  # Avoid extreme values
        
        if np.any(valid_mask):
            with np.errstate(over='ignore', invalid='ignore'):
                exp_term[valid_mask] = np.exp(-(X[valid_mask]-alpha)/beta)
            
        A = b + (1-b) / (1 + exp_term)
        # Clip to handle any remaining invalid values
        A = np.clip(A, 0, 1)
        return A
    
    # MATCHING LOGIC FROM SUPERVISOR'S SCRIPT
    logger.info("Starting LINCS-CTRP matching...")
    
    # Experiment id
    experiment_id_list = curves_df.experiment_id.tolist()

    # Experiment id -> Master ccl id
    master_ccl_id_mapping = per_experiment.set_index('experiment_id')['master_ccl_id'].to_dict()

    # Master ccl id -> ccl name
    ccl_name_mapping = per_cell_line.set_index('master_ccl_id')['ccl_name'].to_dict()

    # Master cpd id
    master_cpd_id_list = curves_df.master_cpd_id.tolist()

    # Master cpd id -> Compound name
    cpd_name_mapping = per_compound.set_index('master_cpd_id')['cpd_name'].to_dict()

    # Master cpd id -> Broad cpd id
    broad_cpd_id_mapping = per_compound.set_index('master_cpd_id')['broad_cpd_id'].to_dict()

    # (Series) Curves idx -> Smile
    per_compound_smiles_curves_ordered = per_compound.set_index('master_cpd_id').loc[curves_df.master_cpd_id].cpd_smiles
    per_compound_smiles_curves_ordered.index = curves_df.index

    # Master ccl id (in curves order)
    logger.info("Processing cell line mappings...")
    master_ccl_id_list = [master_ccl_id_mapping[experiment_id] for experiment_id in experiment_id_list]

    # Ccl name (in curves order)
    ccl_name_list = [ccl_name_mapping[master_ccl_id] for master_ccl_id in tqdm(master_ccl_id_list, desc="Mapping cell line names")]

    # Cpd name (in curves order)
    cpd_name_list = [cpd_name_mapping[master_cpd_id] for master_cpd_id in tqdm(master_cpd_id_list, desc="Mapping compound names")]

    # Broad cpd id (in curves order)
    broad_cpd_id_list = [broad_cpd_id_mapping[master_cpd_id] for master_cpd_id in tqdm(master_cpd_id_list, desc="Mapping Broad IDs")]

    # Cell mfc name
    cell_mfc_name_list = siginfo_df.cell_mfc_name.tolist()

    # Pert mfc id (compound name 1)
    pert_mfc_id_list = siginfo_df.pert_mfc_id.tolist()

    # Cmap name (compound name 2)
    cmap_name_list = siginfo_df.cmap_name.tolist()

    # (List) (cell line name, compound name) (siginfo order)
    logger.info("Creating matching pairs...")
    siginfo_1 = list(zip(cell_mfc_name_list, pert_mfc_id_list))
    siginfo_2 = list(zip(cell_mfc_name_list, cmap_name_list))

    # (Series) (cell line name, compound name) (siginfo order)
    siginfo_1_Series = pd.Series(siginfo_1)
    siginfo_2_Series = pd.Series(siginfo_2)

    # (List) (cell line name, compound name) (curve order)
    ctrp_1 = list(zip(ccl_name_list, broad_cpd_id_list))
    ctrp_2 = list(zip(ccl_name_list, cpd_name_list))

    # (Series) (cell line name, compound name) (curve order)
    ctrp_1_Series = pd.Series(ctrp_1)
    ctrp_2_Series = pd.Series(ctrp_2)

    # (set) (cell line name, compound name) (in siginfo and curves)
    isct_1 = set(siginfo_1).intersection(set(ctrp_1))
    isct_2 = set(siginfo_2).intersection(set(ctrp_2))
    
    logger.info(f"Found {len(isct_1)} matches in first mapping strategy")
    logger.info(f"Found {len(isct_2)} matches in second mapping strategy")
    logger.info(f"Total unique matched pairs: {len(isct_1) + len(isct_2)}")

    # (list) (cell line name, compound name) (in siginfo and curves)
    isct_1_list = list(isct_1)
    isct_2_list = list(isct_2)

    # (Series) (cell line name, compound name) (in siginfo and curves) (with curve index)
    ctrp_1_in_isct = ctrp_1_Series.loc[ctrp_1_Series.isin(isct_1)]
    ctrp_2_in_isct = ctrp_2_Series.loc[ctrp_2_Series.isin(isct_2)]

    # (Series) (cell line name, compound name) (in siginfo and curves) (with siginfo index)
    siginfo_1_in_isct = siginfo_1_Series.loc[siginfo_1_Series.isin(isct_1)]
    siginfo_2_in_isct = siginfo_2_Series.loc[siginfo_2_Series.isin(isct_2)]

    # # Get boolean array of indices if the siginfo series index is in the first matching
    # siginfo_2_idx_in_siginfo_1 = siginfo_2_in_isct.index.isin(siginfo_1_in_isct.index)

    # # Keep only the siginfo indices that are not in the series index of the first matching
    # siginfo_2_in_isct = siginfo_2_in_isct.loc[~(siginfo_2_idx_in_siginfo_1)]
    
    # Proposed: Keep all matches without deduplication
    # Comment out the deduplication step and concatenate directly
    siginfo_concat = pd.concat([siginfo_1_in_isct, siginfo_2_in_isct])
    siginfo_concat = siginfo_concat.drop_duplicates(subset='sig_id')  # Ensure unique sig_id only
    
    logger.info(f"After removing duplicates: {len(siginfo_2_in_isct)} unique matches in second mapping")

    # From the indices that are only in the second matching, acquire the unique (cell line name, compound name)
    isct_2_not_covered_list = list(siginfo_2_in_isct.unique())

    # Helper function to get CTRP index mapping
    def get_ctrp_idx_mapping(siginfo_in_isct, ctrp_in_isct, isct_list):
        """Returns a dictionary mapping siginfo index to ctrp index."""
        logger.info("Creating CTRP index mapping...")
        # map pair to ctrp index
        ctrp_in_isct_mapping = {}
        for i in tqdm(isct_list, desc="Mapping pairs to CTRP indices"):
            ctrp_in_isct_mapping[i] = ctrp_in_isct.index[ctrp_in_isct == i].tolist()

        # map siginfo index to ctrp index
        ctrp_idx_mapping = {}
        for i, pair in tqdm(siginfo_in_isct.items(), desc="Mapping siginfo indices to CTRP indices"):
            ctrp_idx_mapping[i] = ctrp_in_isct_mapping[pair]
            
        return ctrp_idx_mapping

    # Helper function to get viability mapping
    def get_viability_mapping(ctrp_idx_mapping):
        """Returns a dictionary mapping siginfo index to a list of curve-evaluated viabilities."""
        logger.info("Creating viability mapping...")
        viability_mapping = dict()
        for key, val in tqdm(ctrp_idx_mapping.items(), desc="Calculating viability values"):
            curves_i = curves_df.loc[val]
            x = siginfo_df.loc[key].pert_dose
            viability = []
            for _, curve in curves_i.iterrows():
                p = curve[['p1_center', 'p2_slope', 'p4_baseline']].tolist()
                eval = sig3upper(p, np.log2(x))
                viability.append(eval)
            viability_mapping[key] = viability
        return viability_mapping

    # Note that this is an IMPORTANT mapping that gets instantiated
    ctrp_1_idx_mapping = get_ctrp_idx_mapping(siginfo_1_in_isct, ctrp_1_in_isct, isct_1_list)
    ctrp_2_idx_mapping = get_ctrp_idx_mapping(siginfo_2_in_isct, ctrp_2_in_isct, isct_2_not_covered_list)
    
    # Add these diagnostic lines here after the mappings are created
    logger.info(f"First mapping keys count: {len(ctrp_1_idx_mapping.keys())}")
    logger.info(f"Second mapping keys count: {len(ctrp_2_idx_mapping.keys())}")
    logger.info(f"Total mapping keys count: {len(ctrp_1_idx_mapping.keys()) + len(ctrp_2_idx_mapping.keys())}")

    # Only the first curve index, as the drug will be identical, for easier handling
    ctrp_1_idx_mapping_first = [i[0] for i in ctrp_1_idx_mapping.values()]
    ctrp_2_idx_mapping_first = [i[0] for i in ctrp_2_idx_mapping.values()]

    # Get the smiles using the curve indices (first matching)
    per_compound_smiles_series_1 = pd.Series(per_compound_smiles_curves_ordered.loc[ctrp_1_idx_mapping_first])
    per_compound_smiles_series_1.index = list(ctrp_1_idx_mapping.keys())

    # Get the smiles using the curve indices (second matching)
    per_compound_smiles_series_2 = pd.Series(per_compound_smiles_curves_ordered.loc[ctrp_2_idx_mapping_first])
    per_compound_smiles_series_2.index = list(ctrp_2_idx_mapping.keys())

    # Concatenate the smiles for both matchings
    per_compound_smiles_concat = pd.concat([per_compound_smiles_series_1, per_compound_smiles_series_2])

    # Using the ctrp indices mapping, get the mapping from siginfo index to viability
    viability_mapping_1 = get_viability_mapping(ctrp_1_idx_mapping)
    viability_mapping_2 = get_viability_mapping(ctrp_2_idx_mapping)

    # Take the mean of the viabilities when there were multiple curves for the siginfo index
    logger.info("Calculating mean viabilities...")
    viability_mapping_1_mean = {siginfo_i: np.array(viability_i).mean() for siginfo_i, viability_i in tqdm(viability_mapping_1.items(), desc="Processing first mapping")}
    viability_mapping_2_mean = {siginfo_i: np.array(viability_i).mean() for siginfo_i, viability_i in tqdm(viability_mapping_2.items(), desc="Processing second mapping")}

    # Concatenate the mean viability values from both matchings
    viability_concat = list(viability_mapping_1_mean.values()) + list(viability_mapping_2_mean.values())

    # Concatenate the two siginfo subsets using the matched siginfo indices
    logger.info("Combining matched data...")
    siginfo_concat = pd.concat([
        siginfo_df.loc[ctrp_1_idx_mapping.keys()],
        siginfo_df.loc[ctrp_2_idx_mapping.keys()]])
    
    # Add this diagnostic line here
    logger.info(f"Number of unique sig_ids in siginfo_concat: {siginfo_concat['sig_id'].nunique()}")
    logger.info(f"Combined siginfo size: {len(siginfo_concat)}")

    # Add smiles, viability to the siginfo dataframe
    siginfo_concat['smiles'] = per_compound_smiles_concat
    siginfo_concat['viability'] = viability_concat
    siginfo_concat['viability_clipped'] = np.clip(siginfo_concat['viability'], 0, 1)
    siginfo_concat['pert_dose_log2'] = np.log2(siginfo_concat['pert_dose'])

    # Get the siginfo slices and their corresponding gctx indices
    siginfo_slice_1 = siginfo_concat.loc[ctrp_1_idx_mapping.keys()]
    siginfo_slice_2 = siginfo_concat.loc[ctrp_2_idx_mapping.keys()]
    
    # Define slice_1_idx and slice_2_idx before attempting to access them
    slice_1_idx = sig_ids.loc[sig_ids.isin(siginfo_slice_1.sig_id)]
    slice_2_idx = sig_ids.loc[sig_ids.isin(siginfo_slice_2.sig_id)]
    
    # Now that they're defined, we can add these diagnostics
    logger.info(f"Number of indices in slice_1_idx: {len(slice_1_idx)}")
    logger.info(f"Number of indices in slice_2_idx: {len(slice_2_idx)}")
    logger.info(f"Total indices: {len(slice_1_idx) + len(slice_2_idx)}")
    
    # Show detailed stats
    logger.info(f"First slice size: {len(siginfo_slice_1)}")
    logger.info(f"Second slice size: {len(siginfo_slice_2)}")
    
    # Add these detailed diagnostics
    logger.info(f"Number of unique cell lines used: {siginfo_concat['cell_mfc_name'].nunique()}")
    logger.info(f"Number of unique compounds used: {siginfo_concat['pert_mfc_id'].nunique()}")
    
    # Check if all signatures from siginfo are found in GCTX
    missing_sig_ids_1 = set(siginfo_slice_1.sig_id) - set(slice_1_idx)
    missing_sig_ids_2 = set(siginfo_slice_2.sig_id) - set(slice_2_idx)
    
    if missing_sig_ids_1:
        logger.warning(f"{len(missing_sig_ids_1)} signatures from first slice not found in GCTX")
    
    if missing_sig_ids_2:
        logger.warning(f"{len(missing_sig_ids_2)} signatures from second slice not found in GCTX")

    # Getting the FULL gene expression data from the matrix 
    logger.info("Extracting full gene expression data from matrix...")
    with h5py.File(env["gctx_file"], 'r') as f:
        matrix = f['0']['DATA']['0']['matrix']
        
        # First matching - ALL GENES
        logger.info("Processing first matching slice...")
        matrix_slice_1 = np.zeros((len(slice_1_idx), matrix.shape[1]), dtype=np.float32)
        
        # Extract data in chunks for better memory handling
        chunk_size = 1000
        for i in tqdm(range(0, len(slice_1_idx), chunk_size), desc="Reading matrix slice 1"):
            end = min(i + chunk_size, len(slice_1_idx))
            matrix_slice_1[i:end] = matrix[slice_1_idx.index[i:end], :]
        
        # Second matching - ALL GENES
        logger.info("Processing second matching slice...")
        matrix_slice_2 = np.zeros((len(slice_2_idx), matrix.shape[1]), dtype=np.float32)
        
        for i in tqdm(range(0, len(slice_2_idx), chunk_size), desc="Reading matrix slice 2"):
            end = min(i + chunk_size, len(slice_2_idx))
            matrix_slice_2[i:end] = matrix[slice_2_idx.index[i:end], :]
    
    # Concatenate both slices
    logger.info("Combining matrix slices...")
    matrix_all_concat_array = np.vstack([matrix_slice_1, matrix_slice_2])
    
    logger.info(f"Matrix dimensions after concatenation: {matrix_all_concat_array.shape}")
    
    # Create a combined set of signature IDs in the right order
    combined_sig_ids = np.concatenate([slice_1_idx.values, slice_2_idx.values])
    
    # Make sure siginfo is in the same order as the matrix
    siginfo_order = list(slice_1_idx) + list(slice_2_idx)
    siginfo_concat_ordered = siginfo_concat.loc[siginfo_concat['sig_id'].isin(siginfo_order)]
    siginfo_concat_ordered = siginfo_concat_ordered.set_index('sig_id').loc[siginfo_order].reset_index()
    
    # Validate data quality
    logger.info("Validating data quality...")
    nan_count = np.isnan(matrix_all_concat_array).sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in expression data")
        # Replace NaNs with zeros to prevent issues
        matrix_all_concat_array = np.nan_to_num(matrix_all_concat_array, nan=0.0)
    
    # Check for infinite values
    inf_count = np.isinf(matrix_all_concat_array).sum()
    if inf_count > 0:
        logger.warning(f"Found {inf_count} infinite values in expression data")
        # Replace infs with large values
        matrix_all_concat_array = np.nan_to_num(matrix_all_concat_array, posinf=1e30, neginf=-1e30)
    
    # Check dimensions match expectations
    n_rows, n_cols = matrix_all_concat_array.shape
    logger.info(f"Final dimensions: {n_rows} rows x {n_cols} columns")
    
    if n_rows != len(siginfo_concat_ordered):
        logger.warning(f"Row count mismatch: Matrix has {n_rows} rows, metadata has {len(siginfo_concat_ordered)} rows")
    
    if n_cols != len(gene_ids):
        logger.warning(f"Column count mismatch: Matrix has {n_cols} columns, gene info has {len(gene_ids)} columns")
    
    # Write to GCTX file
    logger.info(f"Writing GCTX file to {output_gctx_file}")
    
    chunk_shape = (min(256, n_rows), min(256, n_cols))
    
    with h5py.File(output_gctx_file, "w") as f:
        # Write expression data
        grp_data = f.create_group("0/DATA/0")
        dset = grp_data.create_dataset(
            "matrix",
            shape=(n_rows, n_cols),
            dtype="float32",
            compression="gzip",
            chunks=chunk_shape,
        )
        
        # Write data in chunks
        chunk_rows = 1000
        for i in tqdm(range(0, n_rows, chunk_rows), desc="Writing expression data"):
            end = min(i + chunk_rows, n_rows)
            dset[i:end, :] = matrix_all_concat_array[i:end, :].astype("float32")
        
        # Write row metadata (siginfo)
        meta_row_grp = f.create_group("0/META/ROW")
        
        for col in tqdm(siginfo_concat_ordered.columns, desc="Writing row metadata"):
            if col != 'sig_id':  # Skip the ID column as it's already the index
                data = siginfo_concat_ordered[col].values
                if pd.api.types.is_numeric_dtype(siginfo_concat_ordered[col]):
                    meta_row_grp.create_dataset(col, data=data)
                else:
                    meta_row_grp.create_dataset(col, data=data.astype(str).astype("S"))
        
        meta_row_grp.create_dataset("id", data=np.array(siginfo_concat_ordered['sig_id'].tolist(), dtype="S"))
        
        # Write column metadata (gene info) - ALL GENES
        meta_col_grp = f.create_group("0/META/COL")
        
        # Create column metadata dataframe
        col_metadata = pd.DataFrame({
            'gene_id': gene_ids,
            'gene_symbol': gene_symbols,
            'feature_space': feature_spaces
        })
        
        for col in tqdm(col_metadata.columns, desc="Writing column metadata"):
            data = col_metadata[col].values
            if pd.api.types.is_numeric_dtype(col_metadata[col]):
                meta_col_grp.create_dataset(col, data=data)
            else:
                meta_col_grp.create_dataset(col, data=data.astype(str).astype("S"))
        
        meta_col_grp.create_dataset("id", data=np.array(gene_symbols, dtype="S"))
        
        # Add file attributes
        f.attrs["version"] = "1.0"
        f.attrs["description"] = "LINCS-CTRP matched data with ALL genes (raw data without TF activity inference)"
        f.attrs["data_source"] = "LINCS, CTRP"
        f.attrs["creation_date"] = pd.Timestamp.now().isoformat()
        f.attrs["platform"] = "L1000"
    
    # Verify the file
    logger.info("Verifying created GCTX file")
    with h5py.File(output_gctx_file, "r") as f:
        # Check dimensions
        matrix_shape = f["0/DATA/0/matrix"].shape
        logger.info(f"Final matrix dimensions: {matrix_shape[0]} rows, {matrix_shape[1]} columns")
        
        # Check metadata
        row_keys = list(f["0/META/ROW"].keys())
        logger.info(f"Found {len(row_keys)} row metadata fields")
        
        # Verify key columns exist
        for key in ["viability_clipped", "pert_dose_log2", "smiles"]:
            if key in row_keys:
                logger.info(f"{key} column present ✓")
            else:
                logger.error(f"{key} column missing!")
        
        # Check column metadata
        col_keys = list(f["0/META/COL"].keys())
        logger.info(f"Found {len(col_keys)} column metadata fields")
        
        if "feature_space" in col_keys:
            feature_spaces = f["0/META/COL/feature_space"][:]
            feature_space_counts = {}
            for fs in feature_spaces:
                fs_str = fs.decode('utf-8') if isinstance(fs, bytes) else str(fs)
                if fs_str in feature_space_counts:
                    feature_space_counts[fs_str] += 1
                else:
                    feature_space_counts[fs_str] = 1
            
            logger.info(f"Feature space distribution: {feature_space_counts}")
        else:
            logger.error("feature_space column missing in column metadata!")
    
    logger.info(f"Successfully created GCTX file: {output_gctx_file}")
    logger.info(f"Total samples: {n_rows}, Total genes: {n_cols}")

if __name__ == "__main__":
    # Path to output file
    output_gctx_file = "data/processed/LINCS_CTRP.gctx"
    env_file = "environment.json"
    
    # Create the GCTX file
    match_lincs_ctrp_to_gctx(output_gctx_file, env_file)