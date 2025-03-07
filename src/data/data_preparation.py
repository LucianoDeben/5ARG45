# src/data/data_preparation.py
import logging
from typing import Tuple

from torch.utils.data import DataLoader

from src.data.datasets import DatasetFactory
from src.data.feature_transforms import create_feature_transform
from src.data.loaders import GCTXDataLoader

logger = logging.getLogger(__name__)


def prepare_datasets(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare training, validation, and test data loaders based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing data and training parameters.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders.
    """
    try:
        # Step 1: Load data using GCTXDataLoader
        gctx_loader = GCTXDataLoader(config["data"]["gctx_file"])

        # Step 2: Create feature transform for molecular data
        transform_molecular = create_feature_transform(
            config["molecular"].get("transform_molecular", "fingerprint"),
            fingerprint_size=config["molecular"].get("fingerprint_size", 1024),
            fingerprint_radius=config["molecular"].get("fingerprint_radius", 2),
        )

        # Step 3: Create and split datasets using DatasetFactory
        train_ds, val_ds, test_ds = DatasetFactory.create_and_split_multimodal(
            gctx_loader=gctx_loader,
            feature_space=config["data"]["feature_space"],
            nrows=config["data"].get("nrows"),
            test_size=config["training"]["test_size"],
            val_size=config["training"]["val_size"],
            random_state=config["data"]["random_seed"],
            group_by=config["training"]["group_by"],
            stratify_by=config["training"]["stratify_by"],
            transform_transcriptomics=None,
            transform_molecular=transform_molecular,
            chunk_size=config["data"].get("chunk_size"),
        )
        
        # Step 5: Create PyTorch DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["data"].get("num_workers", 0),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["data"].get("num_workers", 0),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["data"].get("num_workers", 0),
        )

        logger.info("Data preparation completed successfully.")
        return train_loader, val_loader, test_loader

    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise
