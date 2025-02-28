import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
from models.chemical.smiles_processing import SMILESProcessor
from models.multimodal_models import MultimodalModel
from models.prediction.viability_prediction import ViabilityPredictor
from models.transcriptomics.encoders import TranscriptomicEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_multimodal_pipeline():
    # Load and merge configs
    default_config = get_default_config()
    custom_config = load_config("../config.yaml")
    config = merge_configs(default_config, custom_config)
    validate_config(config)
    config_id = generate_config_id(config)
    init_wandb(config)

    # Initialize processor for SMILES sequences
    processor = LINCSCTRPDataProcessor(
        gctx_file=config["data"]["gctx_file"],
        feature_space=config["data"]["feature_space"],
        nrows=100,
        test_size=config["training"]["test_size"],
        val_size=config["training"]["val_size"],
        random_state=config["training"]["random_state"],
        batch_size=config["training"]["batch_size"],
    )
    train_loader, val_loader, _ = processor.get_dataloaders()

    # Initialize models
    sample = next(iter(train_loader))
    input_dim = sample["transcriptomics"].shape[1]
    transcriptomic_encoder = TranscriptomicEncoder(
        input_dim, config["model"]["hidden_dims"], config["model"]["dropout"]
    )
    smiles_processor = SMILESProcessor(
        vocab_size=100,
        embedding_dim=64,
        hidden_dim=config["model"]["hidden_dims"][0],
        output_dim=config["model"]["hidden_dims"][0],
        is_cnn=True,
    )
    multimodal_model = MultimodalModel(input_dim, 64, config["model"]["hidden_dims"])
    predictor = ViabilityPredictor(config["model"]["hidden_dims"][-1] * 2)

    # Combine into full pipeline
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        multimodal_model.parameters(), lr=config["training"]["learning_rate"]
    )

    multimodal_model.train()
    for epoch in range(config["training"]["epochs"]):
        epoch_loss = 0.0
        for batch in train_loader:
            trans_data = batch["transcriptomics"]
            smiles_seq = batch["chemicals"]  # Now a padded tensor
            targets = batch["target"]
            optimizer.zero_grad()
            outputs = multimodal_model(trans_data, smiles_seq)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * trans_data.size(0)
        epoch_loss /= len(train_loader.dataset)
        logger.info(
            f"Epoch {epoch+1}/{config['training']['epochs']}, Train Loss: {epoch_loss:.4f}"
        )
        if config["experiment"]["track"]:
            wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss})

    logger.info("Multimodal pipeline test completed")
    wandb.finish()


if __name__ == "__main__":
    test_multimodal_pipeline()
