#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Space and Gene Alignment for MixSeq GCTX

This script:
1. Maps feature_space values from LINCS to MixSeq genes
2. Adds missing LINCS genes to the MixSeq matrix (filled with zeros)
3. Updates the MixSeq GCTX file with the complete gene set
"""

import logging
import os
import h5py
import numpy as np
import pandas as pd
from src.data.loaders import GCTXDataLoader
from src.config.config_utils import setup_logging

# Setup logging
logger = setup_logging()

def align_mixseq_with_lincs(lincs_file: str, mixseq_file: str, output_file: str = None):
    if output_file is None:
        output_file = mixseq_file
    
    # Create temporary output file if we're overwriting
    temp_output = output_file + ".temp" if output_file == mixseq_file else output_file
    
    # Ensure the temp file doesn't exist already
    if os.path.exists(temp_output):
        logger.info(f"Removing existing temporary file: {temp_output}")
        try:
            os.remove(temp_output)
        except Exception as e:
            logger.error(f"Failed to remove existing temp file: {str(e)}")
            return False
    
    logger.info("Loading gene information and expression data from LINCS and MixSeq datasets...")
    
    # Load gene information and expression data
    with GCTXDataLoader(lincs_file) as lincs_loader, GCTXDataLoader(mixseq_file) as mixseq_loader:
        # Load column metadata
        lincs_col_meta = lincs_loader.get_column_metadata()
        mixseq_col_meta = mixseq_loader.get_column_metadata()
        
        # Get shape information
        mixseq_n_rows = mixseq_loader.n_rows
        
        logger.info(f"LINCS: {len(lincs_col_meta)} genes, MixSeq: {len(mixseq_col_meta)} genes")
        
        # Ensure both datasets have gene_symbol column
        if "gene_symbol" not in lincs_col_meta.columns:
            logger.error("LINCS dataset missing gene_symbol column")
            return False
        
        if "gene_symbol" not in mixseq_col_meta.columns:
            logger.error("MixSeq dataset missing gene_symbol column")
            return False
        
        # Check for feature_space in LINCS
        if "feature_space" not in lincs_col_meta.columns:
            logger.error("LINCS dataset missing feature_space column")
            return False
        
        # Get gene lists
        lincs_genes = lincs_col_meta["gene_symbol"].tolist()
        mixseq_genes = mixseq_col_meta["gene_symbol"].tolist()
        
        # Create a mapping from gene symbol to feature_space in LINCS
        lincs_gene_to_feature = dict(zip(lincs_col_meta["gene_symbol"], lincs_col_meta["feature_space"]))
        
        # Count unique feature_space values in LINCS
        unique_feature_space = lincs_col_meta["feature_space"].value_counts()
        logger.info(f"LINCS feature_space distribution: {unique_feature_space.to_dict()}")
        
        # Find overlapping and missing genes
        lincs_genes_set = set(lincs_genes)
        mixseq_genes_set = set(mixseq_genes)
        
        common_genes = lincs_genes_set.intersection(mixseq_genes_set)
        missing_genes = lincs_genes_set - mixseq_genes_set
        mixseq_only_genes = mixseq_genes_set - lincs_genes_set
        
        logger.info(f"Common genes: {len(common_genes)}")
        logger.info(f"Missing LINCS genes: {len(missing_genes)}")
        logger.info(f"MixSeq-only genes: {len(mixseq_only_genes)}")
        
        # Count missing genes by feature_space
        missing_by_feature = {
            fs: sum(1 for g in missing_genes if lincs_gene_to_feature[g] == fs)
            for fs in unique_feature_space.index
        }
        logger.info(f"Missing genes by feature_space: {missing_by_feature}")
        
        # Get expression data for MixSeq
        logger.info("Loading MixSeq expression data...")
        mixseq_expr = mixseq_loader.get_expression_data()
        
        # Create new column metadata for the combined dataset
        logger.info("Creating new gene metadata with all LINCS genes plus MixSeq-only genes...")
        
        # Start with all LINCS genes to preserve order
        new_gene_symbols = lincs_genes.copy()
        new_feature_space = [lincs_gene_to_feature[g] for g in new_gene_symbols]
        
        # Add MixSeq-only genes
        for g in mixseq_genes:
            if g not in lincs_genes_set:
                new_gene_symbols.append(g)
                new_feature_space.append("other")
        
        # Create index mapping from mixseq_genes to new_gene_symbols
        mixseq_to_new_idx = {g: new_gene_symbols.index(g) for g in mixseq_genes}
        
        # Create new expression matrix with zeros for missing genes
        logger.info("Creating new expression matrix with all genes...")
        new_n_cols = len(new_gene_symbols)
        new_expr = np.zeros((mixseq_n_rows, new_n_cols), dtype=np.float32)
        
        # Fill in values for genes that exist in MixSeq
        for i, gene in enumerate(mixseq_genes):
            new_idx = mixseq_to_new_idx[gene]
            new_expr[:, new_idx] = mixseq_expr[:, i]
        
        # Create new column metadata
        new_col_meta = pd.DataFrame({
            "gene_symbol": new_gene_symbols,
            "feature_space": new_feature_space,
            "gene_id": list(range(len(new_gene_symbols)))
        })
        
        # Summary of the final gene distribution
        final_feature_counts = pd.Series(new_feature_space).value_counts()
        logger.info(f"Final feature_space distribution: {final_feature_counts.to_dict()}")
        
        # Create new row metadata by copying the original
        logger.info("Preparing row metadata...")
        mixseq_row_meta = mixseq_loader.get_row_metadata()
        
        # Get row IDs
        row_ids = [str(i) for i in range(mixseq_n_rows)]
    
    # Now create the updated GCTX file
    logger.info(f"Writing updated GCTX file to: {temp_output}")
    
    try:
        # Create a brand new file
        with h5py.File(temp_output, 'w') as dst:
            # Create basic group structure
            dst.create_group("0")
            dst.create_group("0/DATA")
            dst.create_group("0/DATA/0")
            dst.create_group("0/META")
            dst.create_group("0/META/ROW")
            dst.create_group("0/META/COL")
            
            # Create the expression matrix dataset
            logger.info(f"Writing expression matrix of shape {new_expr.shape}...")
            chunk_shape = (256, min(new_n_cols, 1000))
            dset = dst.create_dataset(
                "0/DATA/0/matrix",
                shape=(mixseq_n_rows, new_n_cols),
                dtype="float32",
                compression="gzip",
                chunks=chunk_shape
            )
            
            # Write expression data in chunks
            chunk_rows = 16384
            for start in range(0, mixseq_n_rows, chunk_rows):
                end = min(start + chunk_rows, mixseq_n_rows)
                dset[start:end, :] = new_expr[start:end, :]
            
            # Write row metadata - carefully check for duplicates
            logger.info("Writing row metadata...")
            row_meta_grp = dst["0/META/ROW"]
            written_columns = set()
            
            for col in mixseq_row_meta.columns:
                if col in written_columns:
                    logger.warning(f"Skipping duplicate row metadata column: {col}")
                    continue
                    
                data = mixseq_row_meta[col].values
                if pd.api.types.is_numeric_dtype(mixseq_row_meta[col]):
                    row_meta_grp.create_dataset(col, data=data)
                else:
                    row_meta_grp.create_dataset(col, data=data.astype(str).astype("S"))
                written_columns.add(col)
            
            # Write row IDs
            if "id" in written_columns:
                logger.warning("Row ID already written, skipping duplicate")
            else:
                row_meta_grp.create_dataset("id", data=np.array(row_ids, dtype="S"))
            
            # Write column metadata - carefully check for duplicates
            logger.info("Writing column metadata...")
            col_meta_grp = dst["0/META/COL"]
            written_columns = set()
            
            for col in new_col_meta.columns:
                if col in written_columns:
                    logger.warning(f"Skipping duplicate column metadata column: {col}")
                    continue
                    
                data = new_col_meta[col].values
                if pd.api.types.is_numeric_dtype(new_col_meta[col]):
                    col_meta_grp.create_dataset(col, data=data)
                else:
                    col_meta_grp.create_dataset(col, data=data.astype(str).astype("S"))
                written_columns.add(col)
            
            # Write column IDs (gene symbols)
            if "id" in written_columns:
                logger.warning("Column ID already written, skipping duplicate")
            else:
                col_meta_grp.create_dataset("id", data=np.array(new_gene_symbols, dtype="S"))
            
            # Add metadata attributes
            with h5py.File(mixseq_file, 'r') as src:
                # Copy original attributes
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
            
            # Add or update additional info
            dst.attrs["genes_aligned_with_lincs"] = "true"
            dst.attrs["lincs_source_file"] = os.path.basename(lincs_file)
            dst.attrs["total_genes"] = len(new_gene_symbols)
            dst.attrs["missing_genes_added"] = len(missing_genes)
            dst.attrs["alignment_date"] = pd.Timestamp.now().isoformat()
            
        # If we're overwriting the original, replace it
        if output_file == mixseq_file:
            os.replace(temp_output, mixseq_file)
            logger.info(f"Successfully updated {mixseq_file} with aligned genes and feature_space")
        else:
            logger.info(f"Successfully created {output_file} with aligned genes and feature_space")
            
        return True
        
    except Exception as e:
        logger.error(f"Error creating updated GCTX file: {str(e)}")
        # Clean up temp file if it exists
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return False


def main():
    """Main function."""
    lincs_file = "./data/processed/LINCS_CTRP_matched.gctx"
    mixseq_file = "./data/processed/MixSeq.gctx"
    
    # Align MixSeq with LINCS
    success = align_mixseq_with_lincs(lincs_file, mixseq_file)
    
    if success:
        logger.info("Gene alignment and feature space mapping completed successfully")
    else:
        logger.error("Gene alignment and feature space mapping failed")
        exit(1)


if __name__ == "__main__":
    main()