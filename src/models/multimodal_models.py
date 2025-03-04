# src/models/multimodal_models.py

import logging
from typing import Dict, List

import torch
import torch.nn as nn

from src.models.chemical.smiles_processing import MolecularEncoder, SMILESEncoder

from ..models.chemical.descriptors import MolecularDescriptorEncoder
from ..models.integration.fusion import FeatureFusion
from ..models.prediction.viability_prediction import ViabilityPredictor
from ..models.transcriptomics.encoders import TranscriptomicEncoder

logger = logging.getLogger(__name__)


class MultimodalDrugResponseModel(nn.Module):
    """
    Multimodal neural network for drug response prediction.

    Combines transcriptomics and chemical feature encoders with a fusion module
    and prediction head to predict cell viability.
    """

    def __init__(
        self,
        transcriptomics_input_dim: int = 978,
        chemical_input_dim: int = 1025,
        transcriptomics_hidden_dims: list = [512, 256],
        chemical_hidden_dims: list = [256, 128],
        transcriptomics_output_dim: int = 128,
        chemical_output_dim: int = 128,
        fusion_output_dim: int = 128,
        fusion_strategy: str = "concat",
        predictor_hidden_dims: list = [64, 32],
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        """Initialize the multimodal network."""
        super(MultimodalDrugResponseModel, self).__init__()

        # Store configuration
        self.config = {
            "transcriptomics_input_dim": transcriptomics_input_dim,
            "chemical_input_dim": chemical_input_dim,
            "transcriptomics_hidden_dims": transcriptomics_hidden_dims,
            "chemical_hidden_dims": chemical_hidden_dims,
            "transcriptomics_output_dim": transcriptomics_output_dim,
            "chemical_output_dim": chemical_output_dim,
            "fusion_output_dim": fusion_output_dim,
            "fusion_strategy": fusion_strategy,
            "predictor_hidden_dims": predictor_hidden_dims,
            "dropout": dropout,
            "activation": activation,
        }

        # Transcriptomics encoder
        self.transcriptomics_encoder = TranscriptomicEncoder(
            input_dim=transcriptomics_input_dim,
            hidden_dims=transcriptomics_hidden_dims,
            output_dim=transcriptomics_output_dim,
            dropout=dropout,
            activation=activation,
            normalize=True,
        )

        # Chemical encoder (handles molecular descriptors with dosage)
        self.chemical_encoder = MolecularDescriptorEncoder(
            input_dim=chemical_input_dim,
            hidden_dims=chemical_hidden_dims,
            output_dim=chemical_output_dim,
            dropout=dropout,
            activation=activation,
        )

        # Fusion module
        if fusion_strategy == "concat":
            fusion_input_dim = transcriptomics_output_dim + chemical_output_dim
        else:
            fusion_input_dim = max(transcriptomics_output_dim, chemical_output_dim)

        self.fusion = FeatureFusion(
            t_dim=transcriptomics_output_dim,
            c_dim=chemical_output_dim,
            output_dim=fusion_output_dim,
            strategy=fusion_strategy,
            dropout=dropout,
        )

        # Prediction head
        self.predictor = ViabilityPredictor(
            input_dim=fusion_output_dim,
            hidden_dims=predictor_hidden_dims,
            dropout=dropout,
            activation=activation,
            output_activation="sigmoid",  # Use sigmoid for viability prediction
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for multimodal input."""
        # Extract inputs
        transcriptomics = inputs["transcriptomics"]
        molecular = inputs["molecular"]

        # Process through encoders
        transcriptomics_features = self.transcriptomics_encoder(transcriptomics)
        chemical_features = self.chemical_encoder(molecular)

        # Fuse features
        fused_features = self.fusion(transcriptomics_features, chemical_features)

        # Make prediction
        predictions = self.predictor(fused_features)

        return predictions


class MultimodalViabilityPredictor(nn.Module):
    def _create_encoder(
        self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float
    ) -> nn.Module:
        """
        Create a flexible encoder with multiple hidden layers
        """
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def __init__(
        self,
        transcriptomics_input_dim: int,
        molecular_input_dim: int,
        transcriptomics_hidden_dims: List[int] = [512, 256],
        molecular_hidden_dims: List[int] = [256, 128],
        transcriptomics_output_dim: int = 128,
        molecular_output_dim: int = 128,
        fusion_strategy: str = "concat",
        predictor_hidden_dims: List[int] = [64, 32],
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        super().__init__()

        # Transcriptomics Encoder
        self.transcriptomics_encoder = self._create_encoder(
            input_dim=transcriptomics_input_dim,
            hidden_dims=transcriptomics_hidden_dims,
            output_dim=transcriptomics_output_dim,
            dropout=dropout,
        )

        # Molecular Encoder
        # Dynamically choose encoder based on input type
        self.molecular_encoder = MolecularEncoder(
            input_dim=molecular_input_dim,
            hidden_dims=molecular_hidden_dims,
            output_dim=molecular_output_dim,
            dropout=dropout,
        )

        # SMILES Encoder (fallback)
        self.smiles_encoder = SMILESEncoder(
            embedding_dim=molecular_output_dim,
            hidden_dim=molecular_output_dim,
            output_dim=molecular_output_dim,
            architecture="cnn",  # or 'rnn' based on your preference
        )

        # Fusion Strategy
        fusion_input_dim = transcriptomics_output_dim + molecular_output_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 2),
            nn.BatchNorm1d(fusion_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Predictor
        predictor_input_dim = fusion_input_dim // 2
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, predictor_hidden_dims[0]),
            nn.BatchNorm1d(predictor_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden_dims[0], predictor_hidden_dims[1]),
            nn.BatchNorm1d(predictor_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(predictor_hidden_dims[1], 1),
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with flexible input handling
        """
        transcriptomics = inputs["transcriptomics"]

        # Handle molecular features flexibly
        if "molecular" in inputs:
            molecular_features = self.molecular_encoder(inputs["molecular"])
        elif "smiles" in inputs:
            molecular_features = self.smiles_encoder({"smiles": inputs["smiles"]})
        else:
            raise ValueError("No molecular features or SMILES string provided")

        # Encode features
        transcriptomics_features = self.transcriptomics_encoder(transcriptomics)

        # Concatenate features
        fused_features = torch.cat(
            [transcriptomics_features, molecular_features], dim=1
        )

        # Apply fusion
        fused_features = self.fusion(fused_features)

        # Predict drug response
        prediction = self.predictor(fused_features)

        return prediction.squeeze()
