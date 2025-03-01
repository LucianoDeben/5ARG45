"""Infrastructure testing script."""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from config.config_utils import load_config, validate_config
from config.default_config import get_default_config
from data.preprocessing import LINCSCTRPDataProcessor, create_transformations
from utils.logging import ExperimentLogger
from utils.storage import CacheManager, CheckpointManager, DatasetStorage

logger = logging.getLogger(__name__)


class SimpleNN(nn.Module):
    """Simple feedforward neural network with separate processing of modalities."""

    def __init__(
        self,
        transcriptomics_dim: int,
        chemical_dim: int,
        hidden_dims: list = [256, 128],
    ):
        super().__init__()

        # Calculate intermediate dimensions
        trans_hidden = hidden_dims[0] // 2
        chem_hidden = hidden_dims[0] // 2

        # Process transcriptomics data
        self.transcriptomics_net = nn.Sequential(
            nn.Linear(transcriptomics_dim, trans_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(trans_hidden),  # Specify number of features
            nn.Dropout(0.2),
        )

        # Process chemical data
        self.chemical_net = nn.Sequential(
            nn.Linear(chemical_dim, chem_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(chem_hidden),  # Specify number of features
            nn.Dropout(0.2),
        )

        # Joint processing after concatenation
        joint_layers = []
        in_dim = trans_hidden + chem_hidden  # Size after concatenation

        for hidden_dim in hidden_dims[1:]:
            joint_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),  # Specify number of features
                    nn.Dropout(0.2),
                ]
            )
            in_dim = hidden_dim

        joint_layers.append(nn.Linear(in_dim, 1))
        self.joint_net = nn.Sequential(*joint_layers)

    def forward(
        self, x_transcriptomics: torch.Tensor, x_chemical: torch.Tensor
    ) -> torch.Tensor:
        # Process each modality separately
        h_trans = self.transcriptomics_net(x_transcriptomics)
        h_chem = self.chemical_net(x_chemical)

        # Concatenate the processed features
        h_joint = torch.cat([h_trans, h_chem], dim=1)

        # Joint processing
        return self.joint_net(h_joint)


def train_sklearn_model(
    train_dataset,
    val_dataset,
    test_dataset,
    exp_logger: ExperimentLogger,
    storage: DatasetStorage,
) -> Ridge:
    """Train and evaluate a sklearn Ridge regression model."""

    logger.info("Training sklearn Ridge regression model...")

    # Get numpy arrays from transcriptomics datasets
    X_train, y_train = train_dataset.get_data()
    X_val, y_val = val_dataset.get_data()
    X_test, y_test = test_dataset.get_data()

    # Initialize and train model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Evaluate
    for name, (X, y) in [
        ("train", (X_train, y_train)),
        ("val", (X_val, y_val)),
        ("test", (X_test, y_test)),
    ]:
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
    train_loader,
    val_loader,
    test_loader,
    config: dict,
    exp_logger: ExperimentLogger,
    checkpoint_manager: CheckpointManager,
) -> SimpleNN:
    """Train and evaluate a PyTorch neural network."""

    logger.info("Training PyTorch neural network...")

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

        # Save checkpoint if best
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

    # Initialize storage systems
    base_dir = Path("data")

    dataset_storage = DatasetStorage(
        base_dir=base_dir, compress=True  # Enable compression for stored data
    )

    cache_manager = CacheManager(
        cache_dir=base_dir / "cache",
        max_memory_size=2.0,  # GB
        max_disk_size=20.0,  # GB
    )

    checkpoint_manager = CheckpointManager(
        dirpath=base_dir / "checkpoints",
        filename="model_{epoch:02d}_{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
    )

    # Log configuration
    exp_logger.log_config(config)

    try:
        # Create transformations
        trans_transform, mol_transform = create_transformations(
            transcriptomics_transform_type="scale",
            molecular_transform_type="fingerprint",
            fingerprint_size=config["chemical"]["fingerprint_size"],
            fingerprint_radius=config["chemical"]["radius"],
        )

        # Initialize data processor
        data_processor = LINCSCTRPDataProcessor(
            gctx_file=config["data"]["gctx_file"],
            feature_space=config["data"]["feature_space"],
            test_size=config["training"]["test_size"],
            val_size=config["training"]["val_size"],
            batch_size=config["training"]["batch_size"],
            transform_transcriptomics=trans_transform,
            transform_molecular=mol_transform,
        )

        # Get datasets for sklearn model
        logger.info("Preparing transcriptomics data for sklearn model...")
        train_dataset, val_dataset, test_dataset = (
            data_processor.get_transcriptomics_data()
        )

        # Train sklearn model
        sklearn_model = train_sklearn_model(
            train_dataset, val_dataset, test_dataset, exp_logger, dataset_storage
        )

        # Get dataloaders for PyTorch model
        logger.info("Preparing multimodal data for PyTorch model...")
        train_loader, val_loader, test_loader = (
            data_processor.get_multimodal_dataloaders()
        )

        # Train PyTorch model
        pytorch_model = train_pytorch_model(
            train_loader,
            val_loader,
            test_loader,
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
