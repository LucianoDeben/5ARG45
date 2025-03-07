#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MixSeq to GCTX Converter

This script converts MixSeq expression data and metadata to GCTX format,
adding SMILES strings and aligning the format with LINCS data.
"""

import logging
import os
import numpy as np
import pandas as pd
import h5py
from dataclasses import dataclass
from typing import Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class DatasetMetadata:
    """Metadata for a dataset."""
    version: str
    description: str
    data_source: str
    creation_date: str
    platform: str
    additional_info: Optional[Dict] = None

def convert_mixseq_to_gctx(
    data_file: str,
    metadata_file: str,
    output_path: str,
    metadata_obj: DatasetMetadata
):
    """
    Convert MixSeq data to GCTX format.
    
    Args:
        data_file: Path to MixSeq expression data CSV
        metadata_file: Path to MixSeq metadata CSV
        output_path: Path for output GCTX file
        metadata_obj: Dataset metadata object
    """
    logger.info("Loading MixSeq data...")
    data = pd.read_csv(data_file)
    metadata = pd.read_csv(metadata_file)
    
    # Always remove the first column as it's an index
    logger.info(f"Removing first column '{data.columns[0]}' (index column)")
    data = data.iloc[:, 1:]
    
    # Get gene symbols (all columns are now gene symbols)
    gene_symbols = data.columns.tolist()
    logger.info(f"Extracted {len(gene_symbols)} gene symbols")
    
    # Convert data to numpy array
    expression_data = data.values
    
    # Verify reasonable value range
    min_val = np.min(expression_data)
    max_val = np.max(expression_data)
    logger.info(f"Expression data with shape {expression_data.shape}, value range: [{min_val:.2f}, {max_val:.2f}]")
    
    # Clean metadata
    logger.info("Cleaning metadata...")
    
    # Remove useless columns
    useless_keys = ["Unnamed: 0.1", "Unnamed: 0", "curves_ccl_name_cpd_name"]
    for key in useless_keys:
        if key in metadata.columns:
            metadata = metadata.drop(columns=[key])
    
    # Reset index
    metadata = metadata.reset_index(drop=True)
    
    # Map SMILES strings for known compounds
    smiles_mapping = {
        'Trametinib': "CC(=O)Nc1cccc(c1)-n1c2c(C)c(=O)n(C)c(Nc3ccc(I)cc3F)c2c(=O)n(C2CC2)c1=O",
        'Navitoclax': "CC1(C)CCC(=C(CN2CCN(CC2)c2ccc(cc2)C(=O)NS(=O)(=O)c2ccc(N[C@H](CCN3CCOCC3)CSc3ccccc3)c(c2)S(=O)(=O)C(F)(F)F)C1)c1ccc(Cl)cc1",
        'Afatinib': "CN(C)CC=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1",
        'Gemcitabine': "Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)C2(F)F)c(=O)n1",
        'Dabrafenib': "CC(C)(C)c1nc(c(s1)-c1ccnc(N)n1)-c1cccc(NS(=O)(=O)c2c(F)cccc2F)c1F",
        'Bortezomib': "CC(C)C[C@@H](NC(=O)[C@@H](Cc1ccccc1)NC(=O)C1=CNC=CN1)B(O)O"
    }
    
    # Add canonical SMILES
    def get_canonical_smiles(drug_name):
        # Try different variations of the drug name
        for name in [drug_name, drug_name.lower(), drug_name.upper(), drug_name.capitalize()]:
            if name in smiles_mapping:
                return smiles_mapping[name]
        return None
    
    # Add canonical SMILES column from the drug column
    if 'drug' in metadata.columns:
        metadata['canonical_smiles'] = metadata['drug'].apply(get_canonical_smiles)
    elif 'cpd_name' in metadata.columns:
        metadata['canonical_smiles'] = metadata['cpd_name'].apply(get_canonical_smiles)
    
    # Rename relevant columns to match LINCS format
    column_mapping = {
        'ccl_name': 'cell_mfc_name',
        'cpd_name': 'cmap_name',
        'broad_cpd_id': 'pert_mfc_id', 
        'master_cpd_id': 'pert_id',
        'tp': 'pert_time',
        'drug_concentrations': 'pert_dose',
        'sig3upper_viability': 'viability_unclipped'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in metadata.columns and new_col not in metadata.columns:
            metadata[new_col] = metadata[old_col]
    
    # Add derived columns
    if 'viability_unclipped' in metadata.columns:
        metadata['viability_clipped'] = metadata['viability_unclipped'].clip(0, 1)
    elif 'sig3upper_viability' in metadata.columns:
        metadata['viability_unclipped'] = metadata['sig3upper_viability']
        metadata['viability_clipped'] = metadata['viability_unclipped'].clip(0, 1)
    
    # Add log2 dose if not already present
    if 'pert_dose' in metadata.columns and 'pert_dose_log2' not in metadata.columns:
        # Convert to numeric to handle any string values
        metadata['pert_dose'] = pd.to_numeric(metadata['pert_dose'], errors='coerce')
        
        # Check if values are already in log2 scale
        dose_values = metadata['pert_dose'].dropna().values
        if dose_values.size > 0:
            max_dose = np.max(dose_values)
            if max_dose < 100:  # Assumption: if max dose is small, it's likely already log2
                logger.info("Dose values appear to be already in log2 scale")
                metadata['pert_dose_log2'] = metadata['pert_dose']
            else:
                logger.info("Converting dose values to log2 scale")
                metadata['pert_dose_log2'] = np.log2(metadata['pert_dose'])
    
    # Ensure necessary columns exist
    required_columns = ['cell_mfc_name', 'cmap_name', 'pert_time', 'pert_dose']
    for col in required_columns:
        if col not in metadata.columns:
            logger.warning(f"Required column {col} not found in metadata")
    
    # Make sure rows in metadata match rows in expression data
    if len(metadata) != expression_data.shape[0]:
        logger.warning(f"Mismatch in number of rows: metadata ({len(metadata)}) vs expression data ({expression_data.shape[0]})")
        # If mismatch, use the smaller number
        min_rows = min(len(metadata), expression_data.shape[0])
        metadata = metadata.iloc[:min_rows]
        expression_data = expression_data[:min_rows, :]
    
    # Create gene/column metadata
    gene_metadata = pd.DataFrame({
        'gene_symbol': gene_symbols,
        'gene_id': list(range(len(gene_symbols))),  # Use indices as gene_ids
    })
    
    # Reset index to ensure alignment with data
    metadata.reset_index(drop=True, inplace=True)
    
    # Set up row IDs as simple integers (as strings)
    row_ids = [str(i) for i in range(len(metadata))]
    
    # Set column IDs (gene symbols)
    col_ids = gene_symbols
    
    # Write to GCTX file
    write_to_gctx(
        expression_data=expression_data,
        row_metadata=metadata,
        col_metadata=gene_metadata,
        row_ids=row_ids,
        col_ids=col_ids,
        output_path=output_path,
        metadata_obj=metadata_obj
    )

def write_to_gctx(
    expression_data: np.ndarray,
    row_metadata: pd.DataFrame,
    col_metadata: pd.DataFrame,
    row_ids: list,
    col_ids: list,
    output_path: str,
    metadata_obj: DatasetMetadata
):
    """
    Write data to GCTX format.
    
    Args:
        expression_data: Expression data matrix
        row_metadata: Row metadata DataFrame
        col_metadata: Column metadata DataFrame
        row_ids: List of row IDs
        col_ids: List of column IDs
        output_path: Path to output file
        metadata_obj: Dataset metadata object
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
        
        # Write column metadata
        meta_col_grp = f_out.create_group("0/META/COL")
        for col in col_metadata.columns:
            data = col_metadata[col].values
            if pd.api.types.is_numeric_dtype(col_metadata[col]):
                meta_col_grp.create_dataset(col, data=data)
            else:
                meta_col_grp.create_dataset(col, data=data.astype(str).astype("S"))
        meta_col_grp.create_dataset("id", data=np.array(col_ids, dtype="S"))
        
        # Add metadata attributes
        f_out.attrs["version"] = metadata_obj.version
        f_out.attrs["description"] = metadata_obj.description
        f_out.attrs["data_source"] = metadata_obj.data_source
        f_out.attrs["creation_date"] = metadata_obj.creation_date
        f_out.attrs["platform"] = metadata_obj.platform
        
        if metadata_obj.additional_info:
            for key, value in metadata_obj.additional_info.items():
                f_out.attrs[key] = value
    
    logger.info(f"Data successfully written to {output_path}")

def main():
    """Main function to run the converter"""
    # File paths
    data_file = "./data/raw/MixSeq/mixseq_lfc.csv"
    metadata_file = "./data/raw/MixSeq/mixseq_meta_viability_sig3upper.csv"
    output_path = "./data/processed/MixSeq.gctx"
    
    # Create metadata
    metadata = DatasetMetadata(
        version="1.0",
        description="MixSeq gene expression data with viability measurements",
        data_source="MixSeq, McFarland et al. 2020",
        creation_date=pd.Timestamp.now().isoformat(),
        platform="MixSeq",
        additional_info={
            "format": "GCTX",
            "external_validation": True
        }
    )
    
    # Convert data
    convert_mixseq_to_gctx(data_file, metadata_file, output_path, metadata)
    
    logger.info(f"MixSeq data successfully converted to GCTX format: {output_path}")

if __name__ == "__main__":
    main()