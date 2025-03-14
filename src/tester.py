import logging
import torch
import numpy as np
import pandas as pd
from src.data.datasets import DatasetFactory
from src.data.loaders import GCTXLoader
from src.data.preprocessing_transforms import TFInferenceTransform, TransformPipeline
import decoupler as dc
import inspect

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug information about the transform class
print("Checking transform classes:")
print(f"TFInferenceTransform class name: {TFInferenceTransform.__name__}")
print(f"TFInferenceTransform module: {TFInferenceTransform.__module__}")
print(f"TransformPipeline class name: {TransformPipeline.__name__}")
print(f"TransformPipeline module: {TransformPipeline.__module__}")

# Get the regulatory network
network = dc.get_collectri(organism="human")

# Create the pipeline with the TF transform
transform = TFInferenceTransform(network=network, method="ulm")
transform_pipeline = TransformPipeline([transform])

# Load data and test with the actual dataset
loader = GCTXLoader("./data/processed/LINCS_small_updated.gctx")

# Use the transform pipeline
train_ds, val_ds, test_ds = DatasetFactory.create_and_split_transcriptomics(
    gctx_loader=loader,
    nrows=1000,
    feature_space="landmark",
    transform_transcriptomics=transform_pipeline,
    use_cache=True
)

# Print original gene information
print("\nOriginal gene information:")
print(f"Number of original genes: {len(train_ds.gene_symbols)}")
print(f"First 5 gene symbols: {train_ds.gene_symbols[:5]}")

# Get a sample from the dataset
try:
    sample = train_ds[0]
    
    # Check if TF names are in the sample
    if 'tf_names' in sample:
        print("\nTranscription Factor information:")
        print(f"Number of TFs: {len(sample['tf_names'])}")
        print(f"First 5 TF names: {sample['tf_names'][:5]}")
        
        # Check the dimensions
        print(f"\nTF activity tensor shape: {sample['transcriptomics'].shape}")
        print(f"Original gene count: {len(train_ds.gene_symbols)}")
        print(f"TF count: {len(sample['tf_names'])}")
    else:
        print("\nWARNING: 'tf_names' not found in sample.")
        print(f"Sample keys: {sample.keys()}")
        print(f"Transcriptomics shape: {sample['transcriptomics'].shape}")
except Exception as e:
    print(f"Error getting sample: {str(e)}")