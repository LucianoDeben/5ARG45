#!/usr/bin/env python
"""
Simplified test script focusing only on the TF inference functionality.
"""
import logging
import os
import torch
import numpy as np
import pandas as pd
import decoupler as dc

from src.data.datasets import DatasetFactory
from src.data.loaders import GCTXLoader
from src.data.preprocessing_transforms import TFInferenceTransform

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=== TF Inference Test ===")
    
    # Clear all caches to start fresh
    print("Clearing all caches...")
    DatasetFactory.clear_cache(memory_cache=True, disk_cache=True)
    
    # Get the regulatory network
    print("Fetching regulatory network...")
    network = dc.get_collectri(organism="human")
    print(f"Network loaded with shape: {network.shape}")
    
    # Create the TF inference transform
    print("Creating TF inference transform...")
    tf_transform = TFInferenceTransform(
        network=network, 
        method="ulm",
        consensus=True,
        min_n=10
    )
    
    # Data path
    data_path = "./data/processed/LINCS_small_updated.gctx"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    # Load the data
    print(f"Loading data from {data_path}...")
    loader = GCTXLoader(data_path)
    print(f"Dataset dimensions: {loader.n_rows} rows, {loader.n_cols} columns")
    
    # Create test datasets - first WITHOUT TF inference
    print("\n1. Creating dataset WITHOUT TF inference...")
    train_ds_raw, val_ds_raw, test_ds_raw = DatasetFactory.create_and_split_datasets(
        dataset_type="transcriptomics",
        gctx_loader=loader,
        nrows=1000,
        feature_space="landmark",
        tf_inference_transform=None,  # No TF inference
        use_cache=False  # Disable cache to ensure fresh creation
    )
    
    # Extract raw dataset information
    raw_gene_count = len(train_ds_raw.feature_names)
    print(f"Raw dataset - Feature count: {raw_gene_count}")
    print(f"Raw dataset - First 5 features: {train_ds_raw.feature_names[:5]}")
    raw_shape = train_ds_raw[0]['transcriptomics'].shape
    print(f"Raw dataset - Tensor shape: {raw_shape}")
    
    # Create test datasets - WITH TF inference
    print("\n2. Creating dataset WITH TF inference...")
    train_ds_tf, val_ds_tf, test_ds_tf = DatasetFactory.create_and_split_datasets(
        dataset_type="transcriptomics",
        gctx_loader=loader,
        nrows=1000,
        feature_space="landmark",
        tf_inference_transform=tf_transform,
        use_cache=False  # Disable cache to ensure fresh creation
    )
    
    # Extract TF dataset information
    tf_feature_count = len(train_ds_tf.feature_names)
    print(f"TF dataset - Feature count: {tf_feature_count}")
    print(f"TF dataset - First 5 features: {train_ds_tf.feature_names[:5]}")
    tf_shape = train_ds_tf[0]['transcriptomics'].shape
    print(f"TF dataset - Tensor shape: {tf_shape}")
    
    # Check for common TF names
    common_tfs = ['STAT3', 'TP53', 'MYC', 'NFKB1', 'JUN']
    found_tfs = [tf for tf in common_tfs if tf in train_ds_tf.feature_names]
    print(f"Found common TFs: {found_tfs}")
    
    # Validation
    print("\n=== VALIDATION RESULTS ===")
    
    # Check tensor shape matches feature count
    tensor_matches_features = tf_shape[0] == tf_feature_count
    print(f"✓ Tensor shape matches feature count: {tensor_matches_features}")
    
    # Check if feature reduction happened
    dimension_reduced = tf_feature_count < raw_gene_count
    if dimension_reduced:
        reduction_percentage = 100 * (1 - tf_feature_count / raw_gene_count)
        print(f"✓ TF inference reduced dimensions from {raw_gene_count} to {tf_feature_count} "
              f"({reduction_percentage:.1f}% reduction)")
    else:
        print(f"✗ WARNING: No dimension reduction detected! "
              f"{raw_gene_count} original features, {tf_feature_count} after TF inference")
    
    # Check that the outputs are different by comparing tensor shapes
    sample_raw = train_ds_raw[0]['transcriptomics']
    sample_tf = train_ds_tf[0]['transcriptomics']
    
    # Can't use torch.allclose() because dimensions are different
    different_size = sample_raw.shape != sample_tf.shape
    print(f"✓ Raw and TF-processed tensors have different sizes: {different_size}")
    
    # Final verdict
    print("\n=== FINAL VERDICT ===")
    if dimension_reduced and tensor_matches_features and different_size:
        print("✅ TF INFERENCE IS WORKING CORRECTLY")
        print(f"Successfully reduced dimensions from {raw_gene_count} genes to {tf_feature_count} TFs")
    else:
        print("❌ TF INFERENCE HAS ISSUES")
        if not dimension_reduced:
            print("  - No dimension reduction detected")
        if not tensor_matches_features:
            print("  - Tensor shape doesn't match feature count")
        if not different_size:
            print("  - TF inference didn't change tensor shape")
    
    # Print detailed results for comparison
    if raw_gene_count > 0 and tf_feature_count > 0:
        print("\n=== DATA SAMPLE COMPARISON ===")
        print(f"First 5 values from raw dataset (out of {raw_gene_count}):")
        print(sample_raw[:5].numpy())
        print(f"\nFirst 5 values from TF dataset (out of {tf_feature_count}):")
        print(sample_tf[:5].numpy())
        
        # Also show some statistical properties
        print("\n=== STATISTICAL PROPERTIES ===")
        print(f"Raw data - mean: {sample_raw.mean().item():.4f}, std: {sample_raw.std().item():.4f}")
        print(f"TF data - mean: {sample_tf.mean().item():.4f}, std: {sample_tf.std().item():.4f}")

if __name__ == "__main__":
    main()