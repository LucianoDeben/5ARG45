#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LINCS-CTRP Data Matcher

This script matches LINCS transcriptomics data with CTRP cell viability data
and outputs the results in a standardized .gctx file format.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import adapter code
from ..data.adapters import DataAdapter, DatasetMetadata, LINCSAdapter

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def sig3upper(p, X):
    """
    This is the curve implementation from the CTRP provided Matlab script.
    Modified to handle extreme values to prevent overflow.
    
    Args:
        p: Parameters [EC50, slope, lower limit]
        X: Concentration values (log2)
        
    Returns:
        Viability values
    """
    alpha = p[0]  # EC50
    beta = p[1]  # slope
    b = p[2]  # lower limit

    # Clip extreme values to prevent overflow
    exp_term = np.clip(-(X - alpha) / beta, -709, 709)  # max value that exp can handle
    A = b + (1 - b) / (1 + np.exp(exp_term))
    return A


def get_viability_mapping(ctrp_idx_mapping, curves_df, siginfo_df):
    """
    Returns a dictionary mapping siginfo index to a list of curve-evaluated viabilities.
    
    Args:
        ctrp_idx_mapping: Mapping from siginfo index to CTRP indices
        curves_df: DataFrame containing curve data
        siginfo_df: DataFrame containing signature info
        
    Returns:
        Dictionary mapping siginfo index to viability values
    """
    viability_mapping = dict()
    for key, val in tqdm(list(ctrp_idx_mapping.items()), desc="Calculating viabilities"):
        curves_i = curves_df.loc[val]
        x = siginfo_df.loc[key].pert_dose
        viability = []
        for i, curve in curves_i.iterrows():
            p = curve[["p1_center", "p2_slope", "p4_baseline"]].tolist()
            eval = sig3upper(p, np.log2(x))
            viability.append(eval)
        viability_mapping[key] = viability
    return viability_mapping


def get_ctrp_idx_mapping(siginfo_in_isct, ctrp_in_isct, isct_list):
    """
    Returns a dictionary mapping siginfo index to CTRP index.
    
    Args:
        siginfo_in_isct: Series with siginfo indices and (cell line, compound) pairs
        ctrp_in_isct: Series with CTRP indices and (cell line, compound) pairs
        isct_list: List of (cell line, compound) pairs in the intersection
        
    Returns:
        Dictionary mapping siginfo index to CTRP indices
    """
    # Map pair to CTRP index
    ctrp_in_isct_mapping = {
        i: ctrp_in_isct.index[ctrp_in_isct == i].tolist() for i in tqdm(isct_list, desc="Mapping CTRP indices")
    }

    # Map siginfo index to CTRP index
    ctrp_idx_mapping = {
        i: ctrp_in_isct_mapping[pair] for i, pair in tqdm(siginfo_in_isct.items(), desc="Creating siginfo-CTRP mapping")
    }
    return ctrp_idx_mapping


def lookup_pair_per_cpd_post_qc(pair_experiment_id, pair_master_cpd_id, per_cpd_post_qc):
    """
    For a given experiment id and compound, returns the per_cpd_post dataframe matching rows.
    
    Args:
        pair_experiment_id: Experiment ID
        pair_master_cpd_id: Master compound ID
        per_cpd_post_qc: DataFrame containing post-QC compound data
        
    Returns:
        DataFrame with matching rows
    """
    experiment_id_bool = per_cpd_post_qc.experiment_id == pair_experiment_id
    master_cpd_id_bool = per_cpd_post_qc.master_cpd_id == pair_master_cpd_id

    pair_bool = experiment_id_bool & master_cpd_id_bool
    pair_per_cpd_post_qc = per_cpd_post_qc.loc[pair_bool]
    return pair_per_cpd_post_qc


def get_min(ctrp_idx_first, curves_df, per_cpd_post_qc):
    """
    From a list of CTRP indices, fetches the lowest concentration value (log2)
    
    Args:
        ctrp_idx_first: List of CTRP indices
        curves_df: DataFrame containing curve data
        per_cpd_post_qc: DataFrame containing post-QC compound data
        
    Returns:
        List of minimum concentration values (log2)
    """
    min_list = list()

    for ctrp_i in tqdm(ctrp_idx_first, desc="Calculating minimum concentrations"):
        # Index the curve to acquire the experiment id and compound id
        curve = curves_df.loc[ctrp_i]

        # Get the cpd_conc_post_qc subset
        cpd_post_qc_subset = lookup_pair_per_cpd_post_qc(
            curve.experiment_id, curve.master_cpd_id, per_cpd_post_qc
        )
        curve_conc = cpd_post_qc_subset.cpd_conc_umol

        # Note that we take the log2 of the concentration as per convention in handling these concentrations
        curve_conc_min = np.log2(curve_conc.min())

        min_list.append(curve_conc_min)
    return min_list


class LincsCtprMatcher:
    """Class for matching LINCS and CTRP data"""
    
    def __init__(self, config):
        """
        Initialize the matcher with configuration.
        
        Args:
            config: Dictionary containing file paths and configuration
        """
        self.config = config
        self.expression_data = None
        self.row_metadata = None
        self.row_ids = None
        self.col_ids = None
        self.gene_symbols = None
        
    def load_data(self):
        """Load LINCS and CTRP data"""
        logger.info("Loading LINCS and CTRP data")
        
        # Load LINCS gene info
        self.geneinfo = pd.read_csv(self.config["geneinfo"], sep="\t")
        self.geneinfo_df = self.geneinfo.set_index("gene_id")
        
        # Load LINCS signature info, keeping only 24h timepoint and signatures with filled perturbation dosage
        self.siginfo = pd.read_csv(self.config["siginfo"], sep="\t", low_memory=False)
        self.siginfo_df = self.siginfo.loc[self.siginfo.pert_time == 24.0]
        self.siginfo_df = self.siginfo_df.loc[~(self.siginfo_df.pert_dose.isna())]
        self.siginfo_df.reset_index(drop=False, inplace=True)
        
        # Load LINCS .gctx file
        self.f = h5py.File(self.config["gctx_file"])
        
        # Signature IDs (in gctx order)
        self.sig_ids = pd.Series(self.f["0"]["META"]["COL"]["id"]).astype(str)
        
        # Gene IDs (in gctx order)
        self.gene_ids = np.array(self.f["0"]["META"]["ROW"]["id"]).astype(int)
        
        # Gene symbols (in gctx order)
        self.gene_symbols = self.geneinfo_df.loc[self.gene_ids].gene_symbol.tolist()
        
        # The matrix object containing gene expression values
        self.matrix = self.f["0"]["DATA"]["0"]["matrix"]
        
        # Landmark genes indices
        self.landmark_idxs = np.where(self.geneinfo_df.loc[self.gene_ids].feature_space.isin(["landmark"]))[0]
        
        # Gene symbols for landmark genes
        self.landmark_gene_symbol = self.geneinfo_df.loc[self.gene_ids].iloc[self.landmark_idxs].gene_symbol.tolist()
        
        # All gene symbols in order
        self.all_gene_symbol = self.geneinfo_df.loc[self.gene_ids].gene_symbol.tolist()
        
        # Gene info in gctx order
        self.geneinfo_ordered = self.geneinfo_df.loc[self.gene_ids]
        
        # Load CTRP data
        self.per_cpd_post_qc = pd.read_csv(self.config["per_cpd_post_qc"], sep="\t")
        self.curves_df = pd.read_csv(self.config["curves_post_qc"], sep="\t")
        self.per_compound = pd.read_csv(self.config["per_compound"], sep="\t")
        self.per_cell_line = pd.read_csv(self.config["per_cell_line"], sep="\t")
        self.per_experiment = pd.read_csv(self.config["per_experiment"], sep="\t")
        
        logger.info("Data loading complete")
        
    def create_mappings(self):
        """Create mappings between LINCS and CTRP data"""
        logger.info("Creating mappings between LINCS and CTRP data")
        
        # Experiment ID list
        self.experiment_id_list = self.curves_df.experiment_id.tolist()
        
        # Experiment ID -> Master CCL ID
        self.master_ccl_id_mapping = self.per_experiment.set_index("experiment_id")["master_ccl_id"].to_dict()
        
        # Master CCL ID -> CCL name
        self.ccl_name_mapping = self.per_cell_line.set_index("master_ccl_id")["ccl_name"].to_dict()
        
        # Master CPD ID list
        self.master_cpd_id_list = self.curves_df.master_cpd_id.tolist()
        
        # Master CPD ID -> Compound name
        self.cpd_name_mapping = self.per_compound.set_index("master_cpd_id")["cpd_name"].to_dict()
        
        # Master CPD ID -> Broad CPD ID
        self.broad_cpd_id_mapping = self.per_compound.set_index("master_cpd_id")["broad_cpd_id"].to_dict()
        
        # SMILES for curves
        self.per_compound_smiles_curves_ordered = self.per_compound.set_index("master_cpd_id").loc[self.curves_df.master_cpd_id].cpd_smiles
        self.per_compound_smiles_curves_ordered.index = self.curves_df.index
        
        # Master CCL ID (in curves order)
        self.master_ccl_id_list = [self.master_ccl_id_mapping[experiment_id] for experiment_id in self.experiment_id_list]
        
        # CCL name (in curves order)
        self.ccl_name_list = [self.ccl_name_mapping[master_ccl_id] for master_ccl_id in self.master_ccl_id_list]
        
        # CPD name (in curves order)
        self.cpd_name_list = [self.cpd_name_mapping[master_cpd_id] for master_cpd_id in self.master_cpd_id_list]
        
        # Broad CPD ID (in curves order)
        self.broad_cpd_id_list = [self.broad_cpd_id_mapping[master_cpd_id] for master_cpd_id in self.master_cpd_id_list]
        
        # LINCS cell and compound data
        self.cell_mfc_name_list = self.siginfo_df.cell_mfc_name.tolist()
        self.pert_mfc_id_list = self.siginfo_df.pert_mfc_id.tolist()
        self.cmap_name_list = self.siginfo_df.cmap_name.tolist()
        
        logger.info("Mapping creation complete")
        
    def find_intersections(self):
        """Find intersections between LINCS and CTRP data"""
        logger.info("Finding intersections between LINCS and CTRP data")
        
        # (cell line name, compound name) pairs for LINCS
        self.siginfo_1 = list(zip(self.cell_mfc_name_list, self.pert_mfc_id_list))
        self.siginfo_2 = list(zip(self.cell_mfc_name_list, self.cmap_name_list))
        
        # Series for LINCS pairs
        self.siginfo_1_Series = pd.Series(self.siginfo_1)
        self.siginfo_2_Series = pd.Series(self.siginfo_2)
        
        # (cell line name, compound name) pairs for CTRP
        self.ctrp_1 = list(zip(self.ccl_name_list, self.broad_cpd_id_list))
        self.ctrp_2 = list(zip(self.ccl_name_list, self.cmap_name_list))
        
        # Series for CTRP pairs
        self.ctrp_1_Series = pd.Series(self.ctrp_1)
        self.ctrp_2_Series = pd.Series(self.ctrp_2)
        
        # Intersection of pairs
        self.isct_1 = set(self.siginfo_1).intersection(set(self.ctrp_1))
        self.isct_2 = set(self.siginfo_2).intersection(set(self.ctrp_2))
        
        # List of intersection pairs
        self.isct_1_list = list(self.isct_1)
        self.isct_2_list = list(self.isct_2)
        
        # Pairs in intersection with indices
        self.ctrp_1_in_isct = self.ctrp_1_Series.loc[self.ctrp_1_Series.isin(self.isct_1)]
        self.ctrp_2_in_isct = self.ctrp_2_Series.loc[self.ctrp_2_Series.isin(self.isct_2)]
        
        self.siginfo_1_in_isct = self.siginfo_1_Series.loc[self.siginfo_1_Series.isin(self.isct_1)]
        self.siginfo_2_in_isct = self.siginfo_2_Series.loc[self.siginfo_2_Series.isin(self.isct_2)]
        
        # Handle overlaps between matching methods
        self.siginfo_2_idx_in_siginfo_1 = self.siginfo_2_in_isct.index.isin(self.siginfo_1_in_isct.index)
        self.siginfo_2_in_isct = self.siginfo_2_in_isct.loc[~(self.siginfo_2_idx_in_siginfo_1)]
        self.isct_2_not_covered_list = list(self.siginfo_2_in_isct.unique())
        
        logger.info(f"Found {len(self.isct_1)} + {len(self.isct_2_not_covered_list)} intersection pairs")
        
    def match_data(self):
        """Match LINCS and CTRP data based on intersections"""
        logger.info("Matching LINCS and CTRP data")
        
        # Get CTRP indices mapping
        self.ctrp_1_idx_mapping = get_ctrp_idx_mapping(
            self.siginfo_1_in_isct, self.ctrp_1_in_isct, self.isct_1_list
        )
        self.ctrp_2_idx_mapping = get_ctrp_idx_mapping(
            self.siginfo_2_in_isct, self.ctrp_2_in_isct, self.isct_2_not_covered_list
        )
        
        # First curve index only
        self.ctrp_1_idx_mapping_first = [i[0] for i in self.ctrp_1_idx_mapping.values()]
        self.ctrp_2_idx_mapping_first = [i[0] for i in self.ctrp_2_idx_mapping.values()]
        
        # Get SMILES using curve indices
        self.per_compound_smiles_series_1 = pd.Series(
            self.per_compound_smiles_curves_ordered.loc[self.ctrp_1_idx_mapping_first]
        )
        self.per_compound_smiles_series_1.index = list(self.ctrp_1_idx_mapping.keys())
        
        self.per_compound_smiles_series_2 = pd.Series(
            self.per_compound_smiles_curves_ordered.loc[self.ctrp_2_idx_mapping_first]
        )
        self.per_compound_smiles_series_2.index = list(self.ctrp_2_idx_mapping.keys())
        
        # Concatenate SMILES
        self.per_compound_smiles_concat = pd.concat(
            [self.per_compound_smiles_series_1, self.per_compound_smiles_series_2]
        ).tolist()
        
        # Get viability mappings
        self.viability_mapping_1 = get_viability_mapping(
            self.ctrp_1_idx_mapping, self.curves_df, self.siginfo_df
        )
        self.viability_mapping_2 = get_viability_mapping(
            self.ctrp_2_idx_mapping, self.curves_df, self.siginfo_df
        )
        
        # Calculate mean viabilities
        self.viability_mapping_1_mean = {
            siginfo_i: np.array(viability_i).mean()
            for siginfo_i, viability_i in tqdm(self.viability_mapping_1.items(), desc="Calculating mean viabilities (1)")
        }
        self.viability_mapping_2_mean = {
            siginfo_i: np.array(viability_i).mean()
            for siginfo_i, viability_i in tqdm(self.viability_mapping_2.items(), desc="Calculating mean viabilities (2)")
        }
        
        # Concatenate viabilities
        self.viability_concat = list(self.viability_mapping_1_mean.values()) + list(
            self.viability_mapping_2_mean.values()
        )
        
        # Concatenate siginfo subsets
        self.siginfo_concat = pd.concat(
            [
                self.siginfo_df.loc[self.ctrp_1_idx_mapping.keys()],
                self.siginfo_df.loc[self.ctrp_2_idx_mapping.keys()],
            ]
        )
        
        logger.info("Data matching complete")
        
    def get_minimum_concentrations(self):
        """Get minimum concentrations for each CTRP curve"""
        logger.info("Calculating minimum concentrations")
        
        # Drop duplicates for optimized value fetching
        self.ctrp_1_in_isct_drop_dup_reset_index = pd.DataFrame(
            self.ctrp_1_in_isct.drop_duplicates().reset_index(drop=False)
        )
        self.ctrp_2_in_isct_drop_dup_reset_index = pd.DataFrame(
            self.ctrp_2_in_isct.drop_duplicates().reset_index(drop=False)
        )
        
        # Get minimum values
        self.ctrp_1_min = get_min(
            self.ctrp_1_in_isct_drop_dup_reset_index["index"].tolist(), 
            self.curves_df, 
            self.per_cpd_post_qc
        )
        self.ctrp_2_min = get_min(
            self.ctrp_2_in_isct_drop_dup_reset_index["index"].tolist(), 
            self.curves_df, 
            self.per_cpd_post_qc
        )
        
        # Add minimum values to dataframes
        self.ctrp_1_in_isct_drop_dup_reset_index["min"] = self.ctrp_1_min
        self.ctrp_2_in_isct_drop_dup_reset_index["min"] = self.ctrp_2_min
        
        # Get ordered minimum values
        self.ctrp_1_min_list = list(
            self.ctrp_1_in_isct_drop_dup_reset_index.set_index(0).loc[list(self.siginfo_1_in_isct)]["min"]
        )
        self.ctrp_2_min_list = list(
            self.ctrp_2_in_isct_drop_dup_reset_index.set_index(0).loc[list(self.siginfo_2_in_isct)]["min"]
        )
        
        # Concatenate minimum concentrations
        self.ctrp_min = self.ctrp_1_min_list + self.ctrp_2_min_list
        
        logger.info("Minimum concentration calculation complete")
        
    def prepare_final_data(self):
        """Prepare final data for output"""
        logger.info("Preparing final data for output")
        
        # Add SMILES, viability, and concentration to siginfo
        # Use canonical_smiles as the field name instead of smiles
        self.siginfo_concat["canonical_smiles"] = self.per_compound_smiles_concat
        self.siginfo_concat["viability_unclipped"] = self.viability_concat  # Store as viability_unclipped directly
        self.siginfo_concat["lowest_conc_log2"] = self.ctrp_min
        
        # Split into slices for better handling
        self.siginfo_slice_1 = self.siginfo_concat.loc[self.ctrp_1_idx_mapping.keys()]
        self.siginfo_slice_2 = self.siginfo_concat.loc[self.ctrp_2_idx_mapping.keys()]
        
        # Get signature IDs from gctx
        self.slice_1_idx = self.sig_ids.loc[self.sig_ids.isin(self.siginfo_slice_1.sig_id)]
        self.slice_2_idx = self.sig_ids.loc[self.sig_ids.isin(self.siginfo_slice_2.sig_id)]
        
        # Extract expression data from matrix
        logger.info("Extracting expression data")
        self.matrix_slice_1 = self.matrix[self.slice_1_idx.index, :]
        self.matrix_slice_2 = self.matrix[self.slice_2_idx.index, :]
        
        # Create aligned metadata and expression data
        # Ensure consistent ordering between expression data and metadata
        slice_1_sig_ids = self.slice_1_idx.tolist()
        slice_2_sig_ids = self.slice_2_idx.tolist()
        
        # Filter siginfo to match exactly the signatures we have expression data for
        self.siginfo_slice_1_filtered = self.siginfo_slice_1[self.siginfo_slice_1.sig_id.isin(slice_1_sig_ids)]
        self.siginfo_slice_2_filtered = self.siginfo_slice_2[self.siginfo_slice_2.sig_id.isin(slice_2_sig_ids)]
        
        # Concatenate expression data
        self.expression_data = np.concatenate((self.matrix_slice_1, self.matrix_slice_2))
        
        # Create combined row metadata, ensuring alignment with expression data
        self.row_metadata = pd.concat([
            self.siginfo_slice_1_filtered.set_index("sig_id", drop=False),
            self.siginfo_slice_2_filtered.set_index("sig_id", drop=False)
        ])
        
        # Set row and column IDs
        self.row_ids = self.row_metadata.index.tolist()
        self.col_ids = self.gene_ids.tolist()
        
        # Create viability_clipped from viability_unclipped
        self.row_metadata['viability_clipped'] = self.row_metadata.viability_unclipped.clip(0, 1)
        
        # Add log2-transformed dose
        self.row_metadata['pert_dose_log2'] = np.log2(self.row_metadata.pert_dose)
        
        # Double-check that expression data and row metadata have matching dimensions
        assert self.expression_data.shape[0] == len(self.row_metadata), \
            f"Mismatch between expression data rows ({self.expression_data.shape[0]}) and metadata rows ({len(self.row_metadata)})"
        
        logger.info(f"Final data prepared with {self.expression_data.shape[0]} samples and {self.expression_data.shape[1]} genes")
    
    def match_and_prepare(self):
        """Execute the full matching and preparation process"""
        self.load_data()
        self.create_mappings()
        self.find_intersections()
        self.match_data()
        self.get_minimum_concentrations()
        self.prepare_final_data()
        
        return (
            self.expression_data,
            self.row_metadata,
            self.row_ids,
            self.col_ids,
            self.gene_symbols
        )


def write_matched_data_to_gctx(
    expression_data, row_metadata, row_ids, col_ids, gene_symbols, output_path, metadata, geneinfo_ordered
):
    """
    Write matched data to .gctx file.
    
    Args:
        expression_data: Gene expression data matrix
        row_metadata: Row metadata DataFrame
        row_ids: List of row IDs
        col_ids: List of column IDs
        gene_symbols: List of gene symbols
        output_path: Path to output file
        metadata: Dataset metadata
        geneinfo_ordered: Gene info DataFrame with gene metadata in the correct order
    """
    logger.info(f"Writing data to {output_path}")
    
    # Make sure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    n_rows_out = len(row_ids)
    n_cols_out = expression_data.shape[1]
    chunk_shape_out = (256, n_cols_out)
    
    with h5py.File(output_path, "w") as f_out:
        # Write expression data
        grp_data = f_out.create_group("0/DATA/0")
        dset = grp_data.create_dataset(
            "matrix",
            shape=(n_rows_out, n_cols_out),
            dtype="float32",
            compression="gzip",
            chunks=chunk_shape_out,
        )
        
        # Write in chunks to handle large datasets
        chunk_rows_read = 16384
        for start in range(0, n_rows_out, chunk_rows_read):
            end = min(start + chunk_rows_read, n_rows_out)
            chunk_data = expression_data[start:end, :].astype("float32")
            dset[start:end, :] = chunk_data
        
        logger.info("Gene expression data written")
        
        # Write row metadata
        meta_row_grp = f_out.create_group("0/META/ROW")
        for col in row_metadata.columns:
            data = row_metadata[col].values
            if pd.api.types.is_numeric_dtype(row_metadata[col]):
                meta_row_grp.create_dataset(col, data=data)
            else:
                meta_row_grp.create_dataset(col, data=data.astype(str).astype("S"))
        meta_row_grp.create_dataset("id", data=np.array(row_ids, dtype="S"))
        
        # Write column metadata - include all columns from geneinfo
        meta_col_grp = f_out.create_group("0/META/COL")
        
        # Write all columns from the geneinfo dataframe
        for col in geneinfo_ordered.columns:
            data = geneinfo_ordered[col].values
            if pd.api.types.is_numeric_dtype(geneinfo_ordered[col]):
                meta_col_grp.create_dataset(col, data=data)
            else:
                meta_col_grp.create_dataset(col, data=data.astype(str).astype("S"))
        
        # Add gene_id as id
        meta_col_grp.create_dataset("id", data=np.array(geneinfo_ordered.index, dtype="S"))
        
        # Add metadata attributes
        f_out.attrs["version"] = metadata.version
        f_out.attrs["description"] = metadata.description
        f_out.attrs["data_source"] = metadata.data_source
        f_out.attrs["creation_date"] = metadata.creation_date
        f_out.attrs["platform"] = metadata.platform
        
        if metadata.additional_info:
            for key, value in metadata.additional_info.items():
                f_out.attrs[key] = value
    
    logger.info(f"Data successfully written to {output_path}")


def main():
    """Main function to run the matcher"""
    # Load configuration
    with open("./environment.json") as handle:
        env = json.load(handle)
        logger.info(f"Environment '{env.get('name', 'default')}' loaded")
    
    # Create matcher and run matching process
    matcher = LincsCtprMatcher(env)
    expression_data, row_metadata, row_ids, col_ids, gene_symbols = matcher.match_and_prepare()
    
    # Create metadata
    metadata = DatasetMetadata(
        version="1.0",
        description="Matched LINCS transcriptomics and CTRP cell viability data",
        data_source="LINCS-L1000, CTRP",
        creation_date=pd.Timestamp.now().isoformat(),
        platform="L1000",
        additional_info={
            "matched_samples": len(row_ids),
            "num_genes": expression_data.shape[1],
            "num_cell_lines": row_metadata["cell_mfc_name"].nunique(),
            "num_compounds": row_metadata["pert_mfc_id"].nunique(),
        }
    )
    
    # Write matched data to .gctx file
    output_path = "./data/processed/LINCS_CTRP_matched.gctx"
    write_matched_data_to_gctx(
        expression_data,
        row_metadata,
        row_ids,
        col_ids,
        gene_symbols,
        output_path,
        metadata,
        matcher.geneinfo_ordered  # Pass the geneinfo_ordered to the write function
    )
    
    logger.info(f"Matched data written to {output_path}")
    logger.info(f"Total matched samples: {len(row_ids)}")

if __name__ == "__main__":
    main()