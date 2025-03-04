# models/multimodal_models.py
import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from config.constants import Activation, FusionStrategy
from models.chemical.descriptors import (
    IntegratedMolecularEncoder,
    MolecularDescriptorEncoder,
)
from models.chemical.graph_networks import MolecularGraphEncoder
from models.chemical.smiles_processing import SMILESEncoder, SMILESWithDosageEncoder
from models.integration.attention import CrossModalAttention, ModalityAttentionFusion
from models.integration.fusion import FeatureFusion, MultimodalFusion
from models.prediction.viability_prediction import (
    MultiTaskViabilityPredictor,
    ViabilityPredictor,
)
from models.transcriptomics.encoders import (
    CNNTranscriptomicEncoder,
    TranscriptomicEncoder,
)

logger = logging.getLogger(__name__)


class MultimodalDrugResponseModel(nn.Module):
    """
    End-to-end multimodal model for drug response prediction.

    Combines transcriptomics and chemical data processing components to predict
    cancer cell viability in response to drug treatments. Supports various encoder
    architectures, fusion strategies, and prediction heads, configured via a
    configuration dictionary.

    Attributes:
        transcriptomics_encoder: Encoder for gene expression data
        chemical_encoder: Encoder for chemical/molecular data
        fusion_module: Module for combining modality features
        predictor: Prediction head for viability estimation
        config: Model configuration dictionary
        modality_dims: Dictionary mapping modality names to feature indices for interpretation
    """

    def __init__(
        self,
        config: Dict,
        transcriptomics_encoder: Optional[nn.Module] = None,
        chemical_encoder: Optional[nn.Module] = None,
        fusion_module: Optional[nn.Module] = None,
        predictor: Optional[nn.Module] = None,
    ):
        """
        Initialize the MultimodalDrugResponseModel.

        Args:
            config: Configuration dictionary containing model specifications
            transcriptomics_encoder: Optional pre-initialized transcriptomics encoder
            chemical_encoder: Optional pre-initialized chemical encoder
            fusion_module: Optional pre-initialized fusion module
            predictor: Optional pre-initialized predictor
        """
        super(MultimodalDrugResponseModel, self).__init__()
        self.config = config
        model_config = config["model"]

        # Use provided components or create from config
        self.transcriptomics_encoder = (
            transcriptomics_encoder
            or self._create_transcriptomics_encoder(model_config)
        )
        self.chemical_encoder = chemical_encoder or self._create_chemical_encoder(
            model_config
        )
        self.fusion_module = fusion_module or self._create_fusion_module(model_config)
        self.predictor = predictor or self._create_predictor(model_config)

        # Store feature dimensions for interpretability
        self.modality_dims = {
            "transcriptomics": (0, model_config["transcriptomics_output_dim"]),
            "chemical": (
                model_config["transcriptomics_output_dim"],
                model_config["transcriptomics_output_dim"]
                + model_config["chemical_output_dim"],
            ),
        }

        logger.info(
            f"Initialized MultimodalDrugResponseModel with "
            f"transcriptomics: {type(self.transcriptomics_encoder).__name__}, "
            f"chemical: {type(self.chemical_encoder).__name__}, "
            f"fusion: {type(self.fusion_module).__name__}, "
            f"predictor: {type(self.predictor).__name__}"
        )

    def _create_transcriptomics_encoder(self, model_config: Dict) -> nn.Module:
        """Create transcriptomics encoder based on config."""
        encoder_type = model_config.get("transcriptomics_encoder_type", "mlp")

        if encoder_type == "mlp":
            return TranscriptomicEncoder(
                input_dim=model_config["transcriptomics_input_dim"],
                hidden_dims=model_config["transcriptomics_hidden_dims"],
                output_dim=model_config["transcriptomics_output_dim"],
                normalize=model_config.get("normalize", True),
                dropout=model_config.get("dropout", 0.3),
                activation=Activation[
                    model_config.get("activation", "RELU").upper()
                ].value,
                residual=model_config.get("use_residual", False),
            )
        elif encoder_type == "cnn":
            return CNNTranscriptomicEncoder(
                input_dim=model_config["transcriptomics_input_dim"],
                hidden_dims=model_config["transcriptomics_hidden_dims"],
                output_dim=model_config["transcriptomics_output_dim"],
                kernel_sizes=model_config.get("kernel_sizes", [5, 5, 3]),
                normalize=model_config.get("normalize", True),
                dropout=model_config.get("dropout", 0.3),
            )
        else:
            raise ValueError(
                f"Unsupported transcriptomics encoder type: {encoder_type}"
            )

    def _create_chemical_encoder(self, model_config: Dict) -> nn.Module:
        """Create chemical encoder based on config."""
        encoder_type = model_config.get("chemical_encoder_type", "descriptors")

        if encoder_type == "descriptors":
            return IntegratedMolecularEncoder(
                descriptor_dim=model_config["chemical_input_dim"],
                hidden_dims=model_config["chemical_hidden_dims"],
                output_dim=model_config["chemical_output_dim"],
                dosage_integration=model_config.get("dosage_integration", "concat"),
                normalize=model_config.get("normalize", True),
                dropout=model_config.get("dropout", 0.3),
            )
        elif encoder_type == "graph":
            return MolecularGraphEncoder(
                input_dim=model_config["chemical_input_dim"],
                hidden_dims=model_config["chemical_hidden_dims"],
                output_dim=model_config["chemical_output_dim"],
                gnn_type=model_config.get("gnn_type", "gcn"),
                pooling=model_config.get("graph_pooling", "mean"),
                dropout=model_config.get("dropout", 0.3),
            )
        elif encoder_type == "smiles":
            return SMILESWithDosageEncoder(
                embedding_dim=model_config.get("smiles_embedding_dim", 64),
                hidden_dim=model_config["chemical_hidden_dims"][0],
                output_dim=model_config["chemical_output_dim"],
                architecture=model_config.get("smiles_architecture", "cnn"),
                dosage_integration=model_config.get("dosage_integration", "concat"),
                vocab_file=model_config.get("vocab_file", "data/raw/vocab.txt"),
                max_length=model_config.get("max_smiles_length", 128),
                dropout=model_config.get("dropout", 0.3),
            )
        else:
            raise ValueError(f"Unsupported chemical encoder type: {encoder_type}")

    def _create_fusion_module(self, model_config: Dict) -> nn.Module:
        """Create fusion module based on config."""
        fusion_type = model_config.get("fusion_type", "simple")

        if fusion_type == "simple":
            return FeatureFusion(
                t_dim=model_config["transcriptomics_output_dim"],
                c_dim=model_config["chemical_output_dim"],
                output_dim=model_config["fusion_output_dim"],
                strategy=FusionStrategy[
                    model_config.get("fusion_strategy", "CONCAT").upper()
                ].value,
                projection=model_config.get("fusion_projection", True),
                dropout=model_config.get("dropout", 0.3),
            )
        elif fusion_type == "attention":
            return CrossModalAttention(
                transcriptomics_dim=model_config["transcriptomics_output_dim"],
                chemical_dim=model_config["chemical_output_dim"],
                hidden_dim=model_config["fusion_output_dim"],
                num_heads=model_config.get("attention_heads", 4),
                dropout=model_config.get("dropout", 0.1),
                use_projection=True,
            )
        elif fusion_type == "multimodal":
            return ModalityAttentionFusion(
                modality_dims={
                    "transcriptomics": model_config["transcriptomics_output_dim"],
                    "chemical": model_config["chemical_output_dim"],
                },
                hidden_dim=model_config["fusion_output_dim"],
                output_dim=model_config["fusion_output_dim"],
                num_heads=model_config.get("attention_heads", 4),
                dropout=model_config.get("dropout", 0.1),
                aggregation=model_config.get("attention_aggregation", "mean"),
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

    def _create_predictor(self, model_config: Dict) -> nn.Module:
        """Create predictor based on config."""
        predictor_type = model_config.get("predictor_type", "standard")

        if predictor_type == "standard":
            return ViabilityPredictor(
                input_dim=model_config["fusion_output_dim"],
                hidden_dims=model_config["predictor_hidden_dims"],
                dropout=model_config.get("dropout", 0.3),
                activation=Activation[
                    model_config.get("activation", "RELU").upper()
                ].value,
                use_batch_norm=model_config.get("use_batch_norm", True),
                output_activation="sigmoid",
                uncertainty=model_config.get("uncertainty", False),
            )
        elif predictor_type == "multitask":
            return MultiTaskViabilityPredictor(
                input_dim=model_config["fusion_output_dim"],
                task_names=model_config["prediction_tasks"],
                shared_hidden_dims=model_config["predictor_hidden_dims"],
                task_specific_dims=model_config.get("task_specific_dims", [64]),
                dropout=model_config.get("dropout", 0.3),
                use_batch_norm=model_config.get("use_batch_norm", True),
            )
        else:
            raise ValueError(f"Unsupported predictor type: {predictor_type}")

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for end-to-end drug response prediction.

        Args:
            batch: Dictionary with 'transcriptomics' and 'molecular' keys containing tensors
                  from MultimodalDrugDataset

        Returns:
            Predicted viability (Tensor) or dictionary of predictions for multi-task model
        """
        if (
            not isinstance(batch, dict)
            or "transcriptomics" not in batch
            or "molecular" not in batch
        ):
            raise ValueError(
                "Input must be a dictionary with 'transcriptomics' and 'molecular' keys"
            )

        # Process transcriptomics data
        t_features = self.transcriptomics_encoder(batch["transcriptomics"])

        # Process chemical data
        c_features = self.chemical_encoder(batch["molecular"])

        # Handle different fusion types
        fusion_type = self.config["model"].get("fusion_type", "simple")

        if fusion_type == "simple":
            fused_features = self.fusion_module(t_features, c_features)
        elif fusion_type == "attention":
            direction = self.config["model"].get("attention_direction", "both")
            if direction == "both":
                t_updated, c_updated = self.fusion_module(
                    t_features, c_features, direction="both"
                )
                fused_features = torch.cat([t_updated, c_updated], dim=-1)
            else:
                fused_features = self.fusion_module(
                    t_features, c_features, direction=direction
                )
        elif fusion_type == "multimodal":
            fused_features = self.fusion_module(
                {"transcriptomics": t_features, "chemical": c_features}
            )

        # Make predictions
        return self.predictor(fused_features)


class ModelFactory:
    """
    Factory class for creating multimodal drug response models.

    Simplifies model creation by instantiating model components based on configuration
    settings, ensuring compatibility between components, and handling necessary
    preprocessing.
    """

    @staticmethod
    def create_model(config: Dict) -> MultimodalDrugResponseModel:
        """
        Create a MultimodalDrugResponseModel based on configuration.

        Args:
            config: Configuration dictionary with model specifications

        Returns:
            Initialized MultimodalDrugResponseModel

        Raises:
            ValueError: If configuration is invalid
        """
        ModelFactory._validate_config(config)
        return MultimodalDrugResponseModel(config)

    @staticmethod
    def _validate_config(config: Dict) -> None:
        """
        Validate model configuration against infrastructure constants and requirements.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if "model" not in config:
            raise ValueError("Configuration must contain 'model' section")

        model_config = config["model"]

        # Required parameters
        required_params = [
            "transcriptomics_input_dim",
            "transcriptomics_hidden_dims",
            "transcriptomics_output_dim",
            "chemical_input_dim",
            "chemical_hidden_dims",
            "chemical_output_dim",
            "fusion_output_dim",
            "predictor_hidden_dims",
        ]

        missing_params = [p for p in required_params if p not in model_config]
        if missing_params:
            raise ValueError(f"Missing required model parameters: {missing_params}")

        # Validate activation functions
        for param in ["activation"]:
            if param in model_config:
                try:
                    Activation[model_config[param].upper()]
                except KeyError:
                    raise ValueError(f"Invalid {param}: {model_config[param]}")

        # Validate fusion strategy
        if "fusion_strategy" in model_config:
            try:
                FusionStrategy[model_config["fusion_strategy"].upper()]
            except KeyError:
                raise ValueError(
                    f"Invalid fusion_strategy: {model_config['fusion_strategy']}"
                )

        # Special case for multitask predictor
        if (
            model_config.get("predictor_type") == "multitask"
            and "prediction_tasks" not in model_config
        ):
            raise ValueError("prediction_tasks required for multitask predictor")

        # Validate optional parameters (e.g., ensure they exist if specified)
        optional_params = {
            "normalize": bool,
            "dropout": float,
            "use_batch_norm": bool,
            "use_residual": bool,
            "dosage_integration": str,
            "gnn_type": str,
            "graph_pooling": str,
            "smiles_architecture": str,
            "vocab_file": str,
            "max_smiles_length": int,
            "smiles_embedding_dim": int,
            "kernel_sizes": List[int],
            "attention_heads": int,
            "attention_direction": str,
            "attention_aggregation": str,
            "fusion_projection": bool,
            "task_specific_dims": List[int],
            "uncertainty": bool,
        }
        for param, expected_type in optional_params.items():
            if param in model_config and not isinstance(
                model_config[param], expected_type
            ):
                raise ValueError(
                    f"Invalid type for {param}: expected {expected_type}, got {type(model_config[param])}"
                )


class TranscriptomicsOnlyModel(nn.Module):
    """
    Unimodal model for drug response prediction using only transcriptomics data.

    A simplification of the full multimodal model, useful for ablation studies
    and baselines. Supports configuration-driven encoder selection.

    Attributes:
        encoder: Encoder for transcriptomics data
        predictor: Prediction head for viability
    """

    def __init__(self, config: Dict):
        """
        Initialize TranscriptomicsOnlyModel.

        Args:
            config: Configuration dictionary
        """
        super(TranscriptomicsOnlyModel, self).__init__()
        model_config = config["model"]

        # Use factory method for encoder creation
        self.encoder = TranscriptomicEncoder(
            input_dim=model_config["transcriptomics_input_dim"],
            hidden_dims=model_config["transcriptomics_hidden_dims"],
            output_dim=model_config["transcriptomics_output_dim"],
            normalize=model_config.get("normalize", True),
            dropout=model_config.get("dropout", 0.3),
            activation=Activation[model_config.get("activation", "RELU").upper()].value,
            residual=model_config.get("use_residual", False),
        )

        # Create predictor
        self.predictor = ViabilityPredictor(
            input_dim=model_config["transcriptomics_output_dim"],
            hidden_dims=model_config["predictor_hidden_dims"],
            dropout=model_config.get("dropout", 0.3),
            activation=Activation[model_config.get("activation", "RELU").upper()].value,
            use_batch_norm=model_config.get("use_batch_norm", True),
            output_activation="sigmoid",
        )

        logger.info("Initialized TranscriptomicsOnlyModel")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transcriptomics-only model.

        Args:
            x: Transcriptomics data tensor [batch_size, input_dim]

        Returns:
            Predicted viability [batch_size, 1]
        """
        features = self.encoder(x)
        return self.predictor(features)


class ChemicalOnlyModel(nn.Module):
    """
    Unimodal model for drug response prediction using only chemical data.

    A simplification of the full multimodal model, useful for ablation studies
    and baselines. Supports configuration-driven encoder selection.

    Attributes:
        encoder: Encoder for chemical data
        predictor: Prediction head for viability
    """

    def __init__(self, config: Dict):
        """
        Initialize ChemicalOnlyModel.

        Args:
            config: Configuration dictionary
        """
        super(ChemicalOnlyModel, self).__init__()
        model_config = config["model"]

        # Use factory method for encoder creation
        self.encoder = self._create_chemical_encoder(model_config)

        # Create predictor
        self.predictor = ViabilityPredictor(
            input_dim=model_config["chemical_output_dim"],
            hidden_dims=model_config["predictor_hidden_dims"],
            dropout=model_config.get("dropout", 0.3),
            activation=Activation[model_config.get("activation", "RELU").upper()].value,
            use_batch_norm=model_config.get("use_batch_norm", True),
            output_activation="sigmoid",
        )

        logger.info(
            f"Initialized ChemicalOnlyModel with encoder type {model_config.get('chemical_encoder_type', 'descriptors')}"
        )

    def _create_chemical_encoder(self, model_config: Dict) -> nn.Module:
        """Create chemical encoder based on config (shared with MultimodalDrugResponseModel)."""
        encoder_type = model_config.get("chemical_encoder_type", "descriptors")

        if encoder_type == "descriptors":
            return IntegratedMolecularEncoder(
                descriptor_dim=model_config["chemical_input_dim"],
                hidden_dims=model_config["chemical_hidden_dims"],
                output_dim=model_config["chemical_output_dim"],
                dosage_integration=model_config.get("dosage_integration", "concat"),
                normalize=model_config.get("normalize", True),
                dropout=model_config.get("dropout", 0.3),
            )
        elif encoder_type == "graph":
            return MolecularGraphEncoder(
                input_dim=model_config["chemical_input_dim"],
                hidden_dims=model_config["chemical_hidden_dims"],
                output_dim=model_config["chemical_output_dim"],
                gnn_type=model_config.get("gnn_type", "gcn"),
                pooling=model_config.get("graph_pooling", "mean"),
                dropout=model_config.get("dropout", 0.3),
            )
        elif encoder_type == "smiles":
            return SMILESWithDosageEncoder(
                embedding_dim=model_config.get("smiles_embedding_dim", 64),
                hidden_dim=model_config["chemical_hidden_dims"][0],
                output_dim=model_config["chemical_output_dim"],
                architecture=model_config.get("smiles_architecture", "cnn"),
                dosage_integration=model_config.get("dosage_integration", "concat"),
                vocab_file=model_config.get("vocab_file", "data/raw/vocab.txt"),
                max_length=model_config.get("max_smiles_length", 128),
                dropout=model_config.get("dropout", 0.3),
            )
        else:
            raise ValueError(f"Unsupported chemical encoder type: {encoder_type}")

    def forward(self, x: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """
        Forward pass for chemical-only model.

        Args:
            x: Chemical data tensor or dictionary with 'molecular' key [batch_size, input_dim] or {'molecular': tensor, 'dosage': tensor}

        Returns:
            Predicted viability [batch_size, 1]
        """
        features = self.encoder(x)
        return self.predictor(features)
