import logging

import torch

import wandb
from config.config_utils import (
    generate_config_id,
    init_wandb,
    load_config,
    merge_configs,
    validate_config,
)
from config.default_config import get_default_config
from data.preprocessing import LINCSCTRPDataProcessor
from models.multimodal_models import (
    MultimodalModel,
)  # Adjust import based on your structure

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    # Load default and custom configs
    default_config = get_default_config()
    custom_config = load_config("../config.yaml")
    config = merge_configs(default_config, custom_config)

    # Validate and set up experiment tracking
    validate_config(config)
    config_id = generate_config_id(config)
    init_wandb(config)

    # Initialize processor with config
    processor = LINCSCTRPDataProcessor(
        gctx_file=config["data"]["gctx_file"],
        feature_space=config["data"]["feature_space"],
        nrows=config["data"]["nrows"],
        test_size=config["training"]["test_size"],
        val_size=config["training"]["val_size"],
        random_state=config["training"]["random_state"],
        batch_size=config["training"]["batch_size"],
    )

    # Get datasets and data loaders
    train_dataset, val_dataset, test_dataset = processor.process()
    train_loader, val_loader, test_loader = processor.get_dataloaders()

    # Log dataset sizes to W&B
    if config["experiment"]["track"]:
        wandb.log(
            {
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "test_size": len(test_dataset),
            }
        )

    # Initialize the model
    model = MultimodalModel(config)

    # Test a batch with the model
    logging.info("Testing data loader and model integration...")
    for batch in train_loader:
        # Reshape to remove the extra dimension
        transcriptomics = batch["transcriptomics"].squeeze(1)  # Shape: [32, 978]
        molecular = batch["molecular"].squeeze(1)  # Shape: [32, 11]
        targets = batch["viability"]

        # Log batch structure
        logging.info(f"Batch keys: {batch.keys()}")
        logging.info(f"Transcriptomics shape: {transcriptomics.shape}")
        logging.info(f"Molecular shape: {molecular.shape}")
        logging.info(f"Targets shape: {targets.shape}")

        # Forward pass through the model
        try:
            outputs = model(transcriptomics, molecular)
            logging.info(f"Model output shape: {outputs.shape}")
            logging.info("Successfully processed batch through the model!")
        except Exception as e:
            logging.error(f"Error during forward pass: {e}")
        break  # Only test the first batch


if __name__ == "__main__":
    main()
