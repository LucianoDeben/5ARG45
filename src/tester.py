import logging
import torch
import numpy as np
import pandas as pd
import decoupler as dc

from src.data.datasets import DatasetFactory
from src.data.loaders import GCTXLoader
from src.data.preprocessing_transforms import (
    TFInferenceTransform, 
    TransformPipeline, 
    create_preprocessing_transform
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Debug information about the transform classes
    print("Checking transform classes:")
    print(f"TFInferenceTransform class name: {TFInferenceTransform.__name__}")
    print(f"TFInferenceTransform module: {TFInferenceTransform.__module__}")
    print(f"TransformPipeline class name: {TransformPipeline.__name__}")
    print(f"TransformPipeline module: {TransformPipeline.__module__}")

    # Get the regulatory network
    try:
        network = dc.get_collectri(organism="human")
    except Exception as e:
        logger.error(f"Failed to fetch regulatory network: {e}")

    # Create the TF inference transform
    tf_transform = TFInferenceTransform(
        network=network, 
        method="ulm",
        consensus=True,
        min_n=10
    )

    # Create the transform pipeline
    transform_pipeline = TransformPipeline([tf_transform])

    # Load data and test with the actual dataset
    try:
        loader = GCTXLoader("./data/processed/LINCS_small_updated.gctx")

        # Use the transform pipeline with TF inference
        train_ds, val_ds, test_ds = DatasetFactory.create_and_split_transcriptomics(
            gctx_loader=loader,
            nrows=1000,
            feature_space="landmark",
            tf_inference_transform=tf_transform,
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

        # Additional dataset validation
        print("\nDataset Validation:")
        print(f"Training dataset size: {len(train_ds)}")
        print(f"Validation dataset size: {len(val_ds)}")
        print(f"Test dataset size: {len(test_ds)}")

    except Exception as e:
        logger.error(f"Error in dataset creation: {e}")
        import traceback
        traceback.print_exc()

    # Cache statistics
    print("\nCache Statistics:")
    cache_stats = DatasetFactory.get_cache_stats()
    for key, value in cache_stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()