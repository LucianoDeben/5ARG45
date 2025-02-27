import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.datasets import MultimodalDataset
from data.loaders import GCTXDataLoader
from data.preprocessing import LINCSCTRPDataProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TranscriptomicNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(TranscriptomicNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> TranscriptomicNet:
    model = TranscriptomicNet(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

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
        logger.info(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")

    return model


def main():
    gctx_file = "../data/processed/LINCS.gctx"  # Update this path
    test_size = 0.2
    val_size = 0.1
    random_state = 42
    batch_size = 32
    epochs = 10

    # Random Splitting
    logger.info("Training with Random Splitting")
    random_processor = LINCSCTRPDataProcessor(
        lincs_file=gctx_file,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        batch_size=batch_size,
    )
    train_loader_random, val_loader_random, test_loader_random = (
        random_processor.get_dataloaders()
    )

    sample_batch = next(iter(train_loader_random))
    input_dim = sample_batch["transcriptomics"].shape[1]
    model_random = train_model(
        train_loader=train_loader_random,
        val_loader=val_loader_random,
        input_dim=input_dim,
        epochs=epochs,
    )

    # Grouped Splitting
    logger.info("Training with Grouped Splitting by 'cell_mfc_name'")
    grouped_processor = LINCSCTRPDataProcessor(
        lincs_file=gctx_file,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        group_by="cell_mfc_name",
        batch_size=batch_size,
    )
    train_loader_grouped, val_loader_grouped, test_loader_grouped = (
        grouped_processor.get_dataloaders()
    )

    model_grouped = train_model(
        train_loader=train_loader_grouped,
        val_loader=val_loader_grouped,
        input_dim=input_dim,
        epochs=epochs,
    )


if __name__ == "__main__":
    main()
