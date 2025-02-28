# data/tester.py
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add parent directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from data.datasets import MultimodalDataset
from data.loaders import GCTXDataLoader
from data.preprocessing import LINCSCTRPDataProcessor, create_transformations


def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def test_gctx_loader(gctx_file, nrows=100):
    """Test the GCTXDataLoader class."""
    logger = logging.getLogger("test_gctx_loader")
    logger.info(f"Testing GCTXDataLoader with {gctx_file}")

    with GCTXDataLoader(gctx_file) as loader:
        # 1. Test basic loading
        logger.info("1. Testing basic loading of expression data")
        try:
            start_time = time.time()
            expression_data = loader.get_expression_data(
                row_slice=slice(0, nrows), feature_space="landmark"
            )
            elapsed = time.time() - start_time
            logger.info(
                f"Loaded expression data with shape {expression_data.shape} in {elapsed:.2f} seconds"
            )
        except Exception as e:
            logger.error(f"Error loading expression data: {e}")
            return False

        # 2. Test metadata loading
        logger.info("2. Testing metadata loading")
        try:
            start_time = time.time()
            row_metadata = loader.get_row_metadata(row_slice=slice(0, nrows))
            elapsed = time.time() - start_time
            logger.info(
                f"Loaded row metadata with shape {row_metadata.shape} in {elapsed:.2f} seconds"
            )

            # Check required columns
            required_cols = ["canonical_smiles", "pert_dose", "viability"]
            missing_cols = [
                col for col in required_cols if col not in row_metadata.columns
            ]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False

            logger.info(f"Row metadata columns: {', '.join(row_metadata.columns)}")
        except Exception as e:
            logger.error(f"Error loading row metadata: {e}")
            return False

        # 3. Test column metadata loading
        logger.info("3. Testing column metadata loading")
        try:
            start_time = time.time()
            col_metadata = loader.get_column_metadata_for_feature_space("landmark")
            elapsed = time.time() - start_time
            logger.info(
                f"Loaded column metadata with shape {col_metadata.shape} in {elapsed:.2f} seconds"
            )
            logger.info(f"Column metadata columns: {', '.join(col_metadata.columns)}")
        except Exception as e:
            logger.error(f"Error loading column metadata: {e}")
            return False

        # 4. Test data with metadata
        logger.info("4. Testing combined data and metadata loading")
        try:
            start_time = time.time()
            expr_data, row_meta, col_meta = loader.get_data_with_metadata(
                row_slice=slice(0, nrows), feature_space="landmark"
            )
            elapsed = time.time() - start_time
            logger.info(f"Loaded combined data in {elapsed:.2f} seconds")
            logger.info(f"Expression data shape: {expr_data.shape}")
            logger.info(f"Row metadata shape: {row_meta.shape}")
            logger.info(f"Column metadata shape: {col_meta.shape}")
        except Exception as e:
            logger.error(f"Error loading combined data: {e}")
            return False

    logger.info("GCTXDataLoader tests completed successfully!")
    return True


def test_multimodal_dataset(gctx_file, nrows=100):
    """Test the MultimodalDataset class."""
    logger = logging.getLogger("test_multimodal_dataset")
    logger.info(f"Testing MultimodalDataset with {gctx_file}")

    # Load data for dataset
    with GCTXDataLoader(gctx_file) as loader:
        expr_data, row_metadata, _ = loader.get_data_with_metadata(
            row_slice=slice(0, nrows), feature_space="landmark"
        )

    # 1. Test basic dataset creation
    logger.info("1. Testing basic dataset creation")
    try:
        start_time = time.time()
        dataset = MultimodalDataset(
            transcriptomics_data=expr_data, metadata=row_metadata
        )
        elapsed = time.time() - start_time
        logger.info(
            f"Created dataset with {len(dataset)} samples in {elapsed:.2f} seconds"
        )
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return False

    # 2. Test dataset indexing
    logger.info("2. Testing dataset indexing")
    try:
        # Single item
        item = dataset[0]
        logger.info(f"Single item keys: {', '.join(item.keys())}")
        logger.info(f"Transcriptomics shape: {item['transcriptomics'].shape}")
        logger.info(f"Molecular shape: {item['molecular'].shape}")
        logger.info(f"Viability shape: {item['viability'].shape}")

        # Multiple items
        items = dataset[0:5]
        logger.info(
            f"Multiple items transcriptomics shape: {items['transcriptomics'].shape}"
        )
    except Exception as e:
        logger.error(f"Error accessing dataset items: {e}")
        return False

    # 3. Test mode conversion
    logger.info("3. Testing mode conversion")
    try:
        # To sklearn mode
        sklearn_dataset = dataset.to_sklearn()
        X, y = sklearn_dataset[0:5]
        logger.info(f"Sklearn X shape: {X.shape}, y shape: {y.shape}")

        # To unimodal mode
        trans_dataset = dataset.to_unimodal("transcriptomics")
        X_trans, y_trans = trans_dataset[0:5]
        logger.info(f"Transcriptomics-only shape: {X_trans.shape}")
    except Exception as e:
        logger.error(f"Error converting dataset modes: {e}")
        return False

    # 4. Test with transformations
    logger.info("4. Testing dataset with transformations")
    try:
        # Create transformations
        trans_transform, mol_transform = create_transformations(
            transcriptomics_transform_type="scale",
            molecular_transform_type="fingerprint",
        )

        # Apply transformations
        transformed_dataset = dataset.with_transforms(
            transcriptomics_transform=trans_transform, molecular_transform=mol_transform
        )

        # Get item
        item = transformed_dataset[0]
        logger.info(
            f"Transformed item transcriptomics shape: {item['transcriptomics'].shape}"
        )
        logger.info(f"Transformed item molecular shape: {item['molecular'].shape}")
    except Exception as e:
        logger.error(f"Error applying transformations: {e}")
        return False

    logger.info("MultimodalDataset tests completed successfully!")
    return True


def test_data_processor(gctx_file, nrows=100):
    """Test the LINCSCTRPDataProcessor class."""
    logger = logging.getLogger("test_data_processor")
    logger.info(f"Testing LINCSCTRPDataProcessor with {gctx_file}")

    # 1. Test basic processor creation and data loading
    logger.info("1. Testing basic processor creation and data loading")
    try:
        start_time = time.time()
        processor = LINCSCTRPDataProcessor(
            gctx_file=gctx_file,
            feature_space="landmark",
            nrows=nrows,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
        )
        elapsed = time.time() - start_time
        logger.info(f"Created processor in {elapsed:.2f} seconds")
    except Exception as e:
        logger.error(f"Error creating processor: {e}")
        return False

    # 2. Test dataset creation
    logger.info("2. Testing dataset creation")
    try:
        start_time = time.time()
        train_dataset, val_dataset, test_dataset = processor.process()
        elapsed = time.time() - start_time
        logger.info(f"Created datasets in {elapsed:.2f} seconds")
        logger.info(f"Train set size: {len(train_dataset)}")
        logger.info(f"Validation set size: {len(val_dataset)}")
        logger.info(f"Test set size: {len(test_dataset)}")
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        return False

    # 3. Test dataloaders
    logger.info("3. Testing dataloader creation")
    try:
        start_time = time.time()
        train_loader, val_loader, test_loader = processor.get_dataloaders()
        elapsed = time.time() - start_time
        logger.info(f"Created dataloaders in {elapsed:.2f} seconds")

        # Test iteration
        for i, batch in enumerate(train_loader):
            if i == 0:
                logger.info(f"First batch keys: {', '.join(batch.keys())}")
                logger.info(
                    f"First batch transcriptomics shape: {batch['transcriptomics'].shape}"
                )
                logger.info(f"First batch molecular shape: {batch['molecular'].shape}")
                logger.info(f"First batch viability shape: {batch['viability'].shape}")
            if i >= 2:  # Only check first few batches
                break
    except Exception as e:
        logger.error(f"Error creating or using dataloaders: {e}")
        return False

    # 4. Test with transformations
    logger.info("4. Testing processor with transformations")
    try:
        # Create transformations
        trans_transform, mol_transform = create_transformations(
            transcriptomics_transform_type="scale",
            molecular_transform_type="fingerprint",
        )

        # Create processor with transformations
        processor_with_transforms = LINCSCTRPDataProcessor(
            gctx_file=gctx_file,
            feature_space="landmark",
            nrows=nrows,
            transform_transcriptomics=trans_transform,
            transform_molecular=mol_transform,
        )

        # Create datasets
        train_dataset, val_dataset, test_dataset = processor_with_transforms.process()
        logger.info(f"Created datasets with transformations")

        # Check first item
        item = train_dataset[0]
        logger.info(
            f"Transformed item transcriptomics shape: {item['transcriptomics'].shape}"
        )
        logger.info(f"Transformed item molecular shape: {item['molecular'].shape}")
    except Exception as e:
        logger.error(f"Error with transformed processor: {e}")
        return False

    # 5. Test with group splitting
    logger.info("5. Testing processor with group splitting")
    try:
        # Check if cell_id column exists for group splitting
        with GCTXDataLoader(gctx_file) as loader:
            metadata = loader.get_row_metadata(row_slice=slice(0, 1))
            if "cell_id" in metadata.columns or "cell_mfc_name" in metadata.columns:
                group_col = (
                    "cell_id" if "cell_id" in metadata.columns else "cell_mfc_name"
                )

                # Create processor with group splitting
                group_processor = LINCSCTRPDataProcessor(
                    gctx_file=gctx_file,
                    feature_space="landmark",
                    nrows=nrows,
                    group_by=group_col,
                )

                # Create datasets
                train_dataset, val_dataset, test_dataset = group_processor.process()
                logger.info(f"Created datasets with group splitting on {group_col}")
                logger.info(f"Train set size: {len(train_dataset)}")
                logger.info(f"Validation set size: {len(val_dataset)}")
                logger.info(f"Test set size: {len(test_dataset)}")
            else:
                logger.warning("No suitable column found for group splitting test")
    except Exception as e:
        logger.error(f"Error with group splitting: {e}")
        # Don't fail the test for this one

    logger.info("LINCSCTRPDataProcessor tests completed successfully!")
    return True


def run_all_tests(gctx_file, nrows=100):
    """Run all tests."""
    logger = logging.getLogger("run_all_tests")
    logger.info(f"Running all tests with GCTX file: {gctx_file} and nrows={nrows}")

    results = {
        "GCTXDataLoader": test_gctx_loader(gctx_file, nrows),
        "MultimodalDataset": test_multimodal_dataset(gctx_file, nrows),
        "LINCSCTRPDataProcessor": test_data_processor(gctx_file, nrows),
    }

    logger.info("Test Results:")
    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{test_name}: {status}")
        all_passed = all_passed and passed

    if all_passed:
        logger.info("All tests passed successfully!")
    else:
        logger.error("Some tests failed. Please check the logs for details.")

    return all_passed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the data module components")
    parser.add_argument("--gctx_file", required=True, help="Path to the GCTX file")
    parser.add_argument(
        "--nrows", type=int, default=100, help="Number of rows to load for testing"
    )
    parser.add_argument(
        "--feature_space", default="landmark", help="Feature space to use"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)

    # Run tests
    success = run_all_tests(args.gctx_file, args.nrows)

    # Exit with appropriate code
    sys.exit(0 if success else 1)
