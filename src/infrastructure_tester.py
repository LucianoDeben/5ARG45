"""Infrastructure layer testing script."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from config.config_utils import load_config, validate_config
from config.default_config import get_default_config
from data.adapters import DatasetMetadata, LINCSAdapter
from data.datasets import MultimodalDataset
from data.loaders import GCTXDataLoader
from data.preprocessing import LINCSCTRPDataProcessor, create_transformations
from utils.logging import ExperimentLogger
from utils.storage import CacheManager, CheckpointManager, DatasetStorage

logger = logging.getLogger(__name__)


class SimpleNN(nn.Module):
    """Simple feedforward neural network."""

    def __init__(
        self,
        transcriptomics_dim: int,
        chemical_dim: int,
        hidden_dims: list = [256, 128],
    ):
        super().__init__()

        # Input dimensions
        total_input_dim = transcriptomics_dim + chemical_dim

        # Build layers
        layers = []
        in_dim = total_input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.2),
                ]
            )
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(
        self, x_transcriptomics: torch.Tensor, x_chemical: torch.Tensor
    ) -> torch.Tensor:
        # Concatenate inputs
        x = torch.cat([x_transcriptomics, x_chemical], dim=1)
        return self.model(x)


def prepare_data(
    config: Dict,
) -> Tuple[MultimodalDataset, MultimodalDataset, MultimodalDataset]:
    """Prepare datasets using our data infrastructure."""

    # Initialize data processor
    processor = LINCSCTRPDataProcessor(
        gctx_file=config["data"]["gctx_file"],
        feature_space=config["data"]["feature_space"],
        test_size=config["training"]["test_size"],
        val_size=config["training"]["val_size"],
        random_state=config["training"]["random_state"],
        group_by=config["training"]["group_by"],
        batch_size=config["training"]["batch_size"],
    )

    # Create transformations
    transform_transcriptomics, transform_molecular = create_transformations(
        transcriptomics_transform_type="normalize",
        molecular_transform_type="fingerprint",
        fingerprint_size=config["chemical"]["fingerprint_size"],
        fingerprint_radius=config["chemical"]["radius"],
    )

    # Get datasets
    train_dataset, val_dataset, test_dataset = processor.process()

    # Apply transformations
    train_dataset = train_dataset.with_transforms(
        transform_transcriptomics, transform_molecular
    )
    val_dataset = val_dataset.with_transforms(
        transform_transcriptomics, transform_molecular
    )
    test_dataset = test_dataset.with_transforms(
        transform_transcriptomics, transform_molecular
    )

    return train_dataset, val_dataset, test_dataset


def train_sklearn_model(
    train_dataset: MultimodalDataset,
    val_dataset: MultimodalDataset,
    test_dataset: MultimodalDataset,
    exp_logger: ExperimentLogger,
    storage: DatasetStorage,
) -> Ridge:
    """Train and evaluate a sklearn Ridge regression model."""

    logger.info("Training sklearn Ridge regression model...")

    # Convert datasets to sklearn format
    train_data = train_dataset.to_sklearn()
    val_data = val_dataset.to_sklearn()
    test_data = test_dataset.to_sklearn()

    # Initialize and train model
    model = Ridge(alpha=1.0)
    model.fit(train_data[0], train_data[1])

    # Evaluate
    for name, (X, y) in [("train", train_data), ("val", val_data), ("test", test_data)]:
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        metrics = {"mse": mse, "r2": r2}
        exp_logger.log_metrics(metrics, step=0, phase=name)

    # Save model
    storage.save_processed(
        model, name="ridge_model", version="v1", metadata={"type": "sklearn_ridge"}
    )

    return model


def train_pytorch_model(
    train_dataset: MultimodalDataset,
    val_dataset: MultimodalDataset,
    test_dataset: MultimodalDataset,
    config: Dict,
    exp_logger: ExperimentLogger,
    checkpoint_manager: CheckpointManager,
) -> SimpleNN:
    """Train and evaluate a PyTorch neural network."""

    logger.info("Training PyTorch neural network...")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset.to_pytorch(),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset.to_pytorch(), batch_size=config["training"]["batch_size"]
    )
    test_loader = DataLoader(
        test_dataset.to_pytorch(), batch_size=config["training"]["batch_size"]
    )

    # Initialize model
    model = SimpleNN(
        transcriptomics_dim=config["model"]["transcriptomics_input_dim"],
        chemical_dim=config["model"]["chemical_input_dim"],
        hidden_dims=config["model"]["predictor_hidden_dims"],
    )

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(config["training"]["epochs"]):
        # Training phase
        model.train()
        train_loss = 0

        for batch in train_loader:
            x_trans = batch["transcriptomics"].to(device)
            x_chem = batch["molecular"].to(device)
            y = batch["viability"].to(device)

            optimizer.zero_grad()
            output = model(x_trans, x_chem)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                x_trans = batch["transcriptomics"].to(device)
                x_chem = batch["molecular"].to(device)
                y = batch["viability"].to(device)

                output = model(x_trans, x_chem)
                loss = criterion(output, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Log metrics
        metrics = {"loss": train_loss, "val_loss": val_loss}
        exp_logger.log_metrics(metrics, step=epoch)

        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_manager.save(
                model=model, epoch=epoch, metrics=metrics, optimizer=optimizer
            )

    # Test evaluation
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            x_trans = batch["transcriptomics"].to(device)
            x_chem = batch["molecular"].to(device)
            y = batch["viability"].to(device)

            output = model(x_trans, x_chem)
            loss = criterion(output, y)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    exp_logger.log_metrics({"test_loss": test_loss}, step=epoch)

    return model


def main():
    """Main testing function."""
    # Load and validate configuration
    config = get_default_config()
    validate_config(config)

    # Initialize experiment logger
    exp_logger = ExperimentLogger(
        experiment_name="infrastructure_test", use_tensorboard=True, use_wandb=False
    )

    # Initialize storage
    base_dir = Path("data")
    cache_manager = CacheManager(
        cache_dir=base_dir / "cache",
        max_memory_size=2.0,  # GB
        max_disk_size=20.0,  # GB
    )

    dataset_storage = DatasetStorage(base_dir=base_dir, cache_manager=cache_manager)

    checkpoint_manager = CheckpointManager(
        dirpath=base_dir / "checkpoints",
        filename="model_{epoch:02d}_{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
    )

    # Log configuration
    exp_logger.log_config(config)

    try:
        # Prepare data
        train_dataset, val_dataset, test_dataset = prepare_data(config)

        # Train sklearn model
        sklearn_model = train_sklearn_model(
            train_dataset, val_dataset, test_dataset, exp_logger, dataset_storage
        )

        # Train PyTorch model
        pytorch_model = train_pytorch_model(
            train_dataset,
            val_dataset,
            test_dataset,
            config,
            exp_logger,
            checkpoint_manager,
        )

        logger.info("Infrastructure testing completed successfully!")

    except Exception as e:
        logger.error(f"Testing failed: {e}")
        raise

    finally:
        # Cleanup
        exp_logger.close()
        cache_manager.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
