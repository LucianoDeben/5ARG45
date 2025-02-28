import logging

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

    # Test data loading
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


if __name__ == "__main__":
    main()
