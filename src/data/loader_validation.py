#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GCTX Data Validation Script

Simplified validation script that performs essential checks on GCTX files.
"""

import logging
import os
import numpy as np
import pandas as pd
import h5py
from typing import Dict, List, Optional, Tuple, Union
from src.data.loaders import GCTXLoader
from src.config.config_utils import setup_logging

# Setup logging
logger = setup_logging()

class GCTXValidator:
    """Validator for GCTX files with essential checks."""
    
    def __init__(self, gctx_file: str, name: str = ""):
        self.gctx_file = gctx_file
        self.name = name or os.path.basename(gctx_file)
        self.loader = None
        self.validation_results = {}
        
    def validate_file_structure(self) -> Dict[str, bool]:
        """Validate basic file structure."""
        results = {
            "file_exists": os.path.exists(self.gctx_file),
            "file_readable": False,
            "has_required_groups": False,
        }
        
        if not results["file_exists"]:
            logger.error(f"File does not exist: {self.gctx_file}")
            return results
            
        try:
            with h5py.File(self.gctx_file, "r") as f:
                results["file_readable"] = True
                required_paths = ["0/DATA/0/matrix", "0/META/ROW", "0/META/COL"]
                results["has_required_groups"] = all(path in f for path in required_paths)
        except Exception as e:
            logger.error(f"Error validating file structure: {str(e)}")
            
        return results
        
    def validate_data_integrity(self) -> Dict[str, bool]:
        """Validate data integrity and value ranges."""
        results = {
            "matrix_not_empty": False,
            "reasonable_value_range": False,
            "consistent_dimensions": False,
        }
        
        try:
            if self.loader is None:
                self.loader = GCTXLoader(self.gctx_file)
                
            with self.loader:
                # Check if matrix is not empty
                matrix_shape = (self.loader.n_rows, self.loader.n_cols)
                results["matrix_not_empty"] = matrix_shape[0] > 0 and matrix_shape[1] > 0
                
                # Sample data to check for reasonable range
                sample_rows = min(1000, self.loader.n_rows)
                sample_data = self.loader.get_data_matrix(row_indices=slice(0, sample_rows))
                
                # Check reasonable value range
                min_val = np.min(sample_data)
                max_val = np.max(sample_data)
                results["reasonable_value_range"] = min_val > -50 and max_val < 50
                
                # Check consistent dimensions
                row_metadata = self.loader.get_row_metadata()
                col_metadata = self.loader.get_column_metadata()
                results["consistent_dimensions"] = (
                    len(row_metadata) == matrix_shape[0] and
                    len(col_metadata) == matrix_shape[1]
                )
                
                logger.info(f"Data integrity: Shape = {matrix_shape}, Range = [{min_val:.2f}, {max_val:.2f}]")
                
        except Exception as e:
            logger.error(f"Error validating data integrity: {str(e)}")
            
        return results
        
    def validate_metadata(self) -> Dict[str, bool]:
        """Validate metadata completeness and value ranges."""
        results = {
            "has_essential_metadata": False,
            "viability_in_range": False,
        }
        
        try:
            if self.loader is None:
                self.loader = GCTXLoader(self.gctx_file)
                
            with self.loader:
                # Check row metadata
                row_metadata = self.loader.get_row_metadata()
                col_metadata = self.loader.get_column_metadata()
                
                # Check for essential columns
                essential_row_keys = {"cell_mfc_name", "cmap_name", "pert_dose", "viability_clipped"}
                essential_col_keys = {"gene_symbol", "feature_space"}
                
                found_row_keys = essential_row_keys.intersection(set(row_metadata.columns))
                found_col_keys = essential_col_keys.intersection(set(col_metadata.columns))
                
                results["has_essential_metadata"] = (
                    len(found_row_keys) >= 3 and len(found_col_keys) >= 1
                )
                
                # Check viability range
                if "viability_clipped" in row_metadata.columns:
                    viability = row_metadata["viability_clipped"].dropna()
                    min_viability = viability.min() if not viability.empty else np.nan
                    max_viability = viability.max() if not viability.empty else np.nan
                    results["viability_in_range"] = min_viability >= 0 and max_viability <= 1
                    logger.info(f"Viability range: [{min_viability:.2f}, {max_viability:.2f}]")
                    
        except Exception as e:
            logger.error(f"Error validating metadata: {str(e)}")
            
        return results
        
    def validate(self) -> Dict:
        """Perform all validation checks and return a summary."""
        self.validation_results = {
            "file_structure": self.validate_file_structure(),
            "data_integrity": self.validate_data_integrity(),
            "metadata": self.validate_metadata(),
        }
        
        # Calculate overall success
        total_checks = 0
        passed_checks = 0
        
        for category, checks in self.validation_results.items():
            for check, result in checks.items():
                total_checks += 1
                if result:
                    passed_checks += 1
        
        success_rate = passed_checks / total_checks if total_checks > 0 else 0
        status = "PASS" if success_rate >= 0.8 else "WARNING" if success_rate >= 0.6 else "FAIL"
        
        self.validation_results["summary"] = {
            "passed_checks": passed_checks,
            "total_checks": total_checks,
            "success_rate": success_rate,
            "status": status
        }
        
        logger.info(f"Validation summary for {self.name}: {passed_checks}/{total_checks} checks passed ({success_rate:.2%}) - {status}")
        
        return self.validation_results


class DatasetComparator:
    """Simplified comparator for two GCTX datasets."""
    
    def __init__(self, dataset1_file: str, dataset2_file: str, name1: str = "Dataset1", name2: str = "Dataset2"):
        self.dataset1_file = dataset1_file
        self.dataset2_file = dataset2_file
        self.name1 = name1
        self.name2 = name2
        
    def compare(self) -> Dict:
        """Compare two datasets for compatibility."""
        results = {
            "both_valid": False,
            "common_gene_symbols": 0.0,
            "compatible_value_ranges": False,
        }
        
        try:
            loader1 = GCTXLoader(self.dataset1_file)
            loader2 = GCTXLoader(self.dataset2_file)
            
            with loader1, loader2:
                # Check if both are valid
                results["both_valid"] = True
                
                # Compare gene symbols
                col_metadata1 = loader1.get_column_metadata()
                col_metadata2 = loader2.get_column_metadata()
                
                if "gene_symbol" in col_metadata1.columns and "gene_symbol" in col_metadata2.columns:
                    gene_symbols1 = set(col_metadata1["gene_symbol"])
                    gene_symbols2 = set(col_metadata2["gene_symbol"])
                    common_genes = gene_symbols1.intersection(gene_symbols2)
                    
                    results["common_gene_symbols"] = len(common_genes) / max(len(gene_symbols1), len(gene_symbols2))
                    logger.info(f"Common genes: {len(common_genes)} ({results['common_gene_symbols']:.2%})")
                
                # Compare value ranges
                sample1 = loader1.get_data_matrix(row_indices=slice(0, min(1000, loader1.n_rows)))
                sample2 = loader2.get_data_matrix(row_indices=slice(0, min(1000, loader2.n_rows)))
                
                min1, max1 = np.min(sample1), np.max(sample1)
                min2, max2 = np.min(sample2), np.max(sample2)
                
                # Check if value ranges are comparable
                min_compatible = abs(min1 - min2) < 10
                max_compatible = abs(max1 - max2) < 10
                
                results["compatible_value_ranges"] = min_compatible and max_compatible
                logger.info(f"Value ranges: {self.name1}=[{min1:.2f}, {max1:.2f}], {self.name2}=[{min2:.2f}, {max2:.2f}]")
                
                # Check viability ranges
                row_metadata1 = loader1.get_row_metadata()
                row_metadata2 = loader2.get_row_metadata()
                
                if "viability_clipped" in row_metadata1.columns and "viability_clipped" in row_metadata2.columns:
                    viability1 = row_metadata1["viability_clipped"].dropna()
                    viability2 = row_metadata2["viability_clipped"].dropna()
                    
                    results["viability_mean_difference"] = abs(viability1.mean() - viability2.mean())
                    logger.info(f"Viability means: {self.name1}={viability1.mean():.2f}, {self.name2}={viability2.mean():.2f}")
                
        except Exception as e:
            logger.error(f"Error comparing datasets: {str(e)}")
        
        # Calculate overall compatibility score
        compatibility_score = 0.0
        if results["both_valid"]:
            gene_score = results["common_gene_symbols"]
            range_score = 1.0 if results["compatible_value_ranges"] else 0.0
            viability_score = 1.0 if results.get("viability_mean_difference", 1.0) < 0.2 else 0.0
            
            compatibility_score = 0.5 * gene_score + 0.3 * range_score + 0.2 * viability_score
        
        status = "COMPATIBLE" if compatibility_score >= 0.7 else "PARTIALLY COMPATIBLE" if compatibility_score >= 0.4 else "INCOMPATIBLE"
        
        results["summary"] = {
            "compatibility_score": compatibility_score,
            "status": status
        }
        
        logger.info(f"Compatibility score: {compatibility_score:.2f} - {status}")
        
        return results


def main():
    """Main function to run validation."""
    # File paths
    lincs_file = "./data/processed/LINCS_CTRP_QC2.gctx"
    mixseq_file = "./data/processed/MixSeq.gctx"
    
    # Validate individual datasets
    logger.info("Validating LINCS dataset...")
    lincs_validator = GCTXValidator(lincs_file, "LINCS")
    lincs_results = lincs_validator.validate()
    
    logger.info("Validating MixSeq dataset...")
    mixseq_validator = GCTXValidator(mixseq_file, "MixSeq")
    mixseq_results = mixseq_validator.validate()
    
    # Compare datasets
    logger.info("Comparing datasets...")
    comparator = DatasetComparator(lincs_file, mixseq_file, "LINCS", "MixSeq")
    comparison_results = comparator.compare()
    
    # Log overall results
    overall_valid = (
        lincs_results["summary"]["status"] != "FAIL" and 
        mixseq_results["summary"]["status"] != "FAIL"
    )
    
    logger.info(f"Overall validation result: {'PASS' if overall_valid else 'FAIL'}")
    logger.info(f"Dataset compatibility: {comparison_results['summary']['status']}")
    
    return overall_valid


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)