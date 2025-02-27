import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from data.datasets import MultimodalDataset
from data.loaders import GCTXDataLoader
from data.preprocessing import LINCSCTRPDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Define a small neural network for transcriptomics only
class SimpleNN(nn.Module):
    def __init__(self, input_dim: int):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Output: viability score
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_neural_net(
    train_loader, val_loader, input_dim: int, epochs: int = 5, device: str = "cpu"
):
    """Train a simple neural network on transcriptomics data."""
    model = SimpleNN(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            transcriptomics = batch["transcriptomics"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad()
            outputs = model(transcriptomics)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * transcriptomics.size(0)
        train_loss /= len(train_loader.dataset)
        logger.info(
            f"Neural Net - Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}"
        )

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                transcriptomics = batch["transcriptomics"].to(device)
                targets = batch["target"].to(device)
                outputs = model(transcriptomics)
                val_loss += criterion(
                    outputs.squeeze(), targets
                ).item() * transcriptomics.size(0)
            val_loss /= len(val_loader.dataset)
            logger.info(
                f"Neural Net - Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}"
            )

    return model


def train_sklearn_model(train_dataset, val_dataset):
    """Train a scikit-learn Random Forest Regressor on transcriptomics data."""
    # Use only transcriptomics features for simplicity
    X_train, y_train = train_dataset.to_numpy()
    X_val, y_val = val_dataset.to_numpy()
    X_train_transcriptomics = X_train[
        :, : train_dataset.transcriptomics.shape[1]
    ]  # Exclude chemicals
    X_val_transcriptomics = X_val[:, : val_dataset.transcriptomics.shape[1]]

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train_transcriptomics, y_train)
    y_pred = model.predict(X_val_transcriptomics)
    mse = mean_squared_error(y_val, y_pred)
    logger.info(f"Scikit-learn RF - Validation MSE: {mse:.4f}")
    return model


def main():
    # Path to your .gctx file (update this)
    gctx_file = "../data/processed/LINCS2.gctx"

    # Configure processor for quick debugging
    processor = LINCSCTRPDataProcessor(
        gctx_file=gctx_file,
        feature_space="landmark",  # Use only landmark genes
        nrows=100,  # Limit to 100 rows
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        batch_size=32,
    )

    # Test data loading and preprocessing
    logger.info("Testing data loading and preprocessing...")
    train_dataset, val_dataset, test_dataset = processor.process()

    # Test PyTorch DataLoader with neural net
    logger.info("Training PyTorch Neural Network...")
    train_loader, val_loader, test_loader = processor.get_dataloaders()
    input_dim = train_dataset.transcriptomics.shape[1]  # Number of landmark genes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nn_model = train_neural_net(
        train_loader, val_loader, input_dim, epochs=5, device=device
    )

    # Test scikit-learn model
    logger.info("Training scikit-learn Random Forest...")
    rf_model = train_sklearn_model(train_dataset, val_dataset)

    # Quick evaluation on test set (optional)
    logger.info("Evaluating on test set...")
    X_test, y_test = test_dataset.to_numpy()
    X_test_transcriptomics = X_test[:, : test_dataset.transcriptomics.shape[1]]
    rf_pred = rf_model.predict(X_test_transcriptomics)
    rf_test_mse = mean_squared_error(y_test, rf_pred)
    logger.info(f"Random Forest Test MSE: {rf_test_mse:.4f}")

    # Evaluate neural net on test set
    nn_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            transcriptomics = batch["transcriptomics"].to(device)
            targets = batch["target"].to(device)
            outputs = nn_model(transcriptomics)
            test_loss += nn.MSELoss()(
                outputs.squeeze(), targets
            ).item() * transcriptomics.size(0)
        test_loss /= len(test_loader.dataset)
        logger.info(f"Neural Net Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
