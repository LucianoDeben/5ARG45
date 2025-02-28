import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

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
    BiologicallyInformedEncoder,
    CNNTranscriptomicEncoder,
    TranscriptomicEncoder,
)

logger = logging.getLogger(__name__)


class MultimodalDrugResponseModel(nn.Module):
    """
    End-to-end multimodal model for drug response prediction.

    This model combines transcriptomics and chemical data processing
    components to predict cancer cell viability in response to drug
    treatments. It supports various encoder architectures, fusion
    strategies, and prediction heads.

    Attributes:
        transcriptomics_encoder: Encoder for gene expression data
        chemical_encoder: Encoder for chemical/molecular data
        fusion_module: Module for combining modality features
        predictor: Prediction head for viability estimation
        config: Model configuration dictionary
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
            config: Configuration dictionary
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
                activation=model_config.get("activation", "relu"),
                residual=model_config.get("use_residual", False),
            )
        elif encoder_type == "cnn":
            return CNNTranscriptomicEncoder(
                input_dim=model_config["transcriptomics_input_dim"],
                hidden_dims=model_config["transcriptomics_hidden_dims"],
                output_dim=model_config["transcriptomics_output_dim"],
                kernel_sizes=model_config.get("kernel_sizes", None),
                normalize=model_config.get("normalize", True),
                dropout=model_config.get("dropout", 0.3),
            )
        elif encoder_type == "biological":
            # For biological encoder, we need pathway groups
            if "pathway_groups" not in model_config:
                raise ValueError("pathway_groups required for biological encoder")

            return BiologicallyInformedEncoder(
                input_dim=model_config["transcriptomics_input_dim"],
                pathway_groups=model_config["pathway_groups"],
                hidden_dims=model_config["transcriptomics_hidden_dims"],
                output_dim=model_config["transcriptomics_output_dim"],
                normalize=model_config.get("normalize", True),
                dropout=model_config.get("dropout", 0.3),
                aggregation=model_config.get("pathway_aggregation", "attention"),
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
                strategy=model_config.get("fusion_strategy", "concat"),
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
                activation=model_config.get("activation", "relu"),
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
        self, batch: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for end-to-end drug response prediction.

        Args:
            batch: Either a dictionary with 'transcriptomics' and 'molecular' keys,
                  or a tuple of (transcriptomics_data, chemical_data)

        Returns:
            Predicted viability or dictionary of predictions for multi-task model
        """
        # Handle different input formats
        if isinstance(batch, dict):
            transcriptomics_data = batch["transcriptomics"]
            chemical_data = batch["molecular"]
        elif isinstance(batch, tuple) and len(batch) == 2:
            transcriptomics_data, chemical_data = batch
        else:
            raise ValueError(
                "Input must be either a dictionary with 'transcriptomics' and 'molecular' keys "
                "or a tuple of (transcriptomics_data, chemical_data)"
            )

        # Process transcriptomics data
        t_features = self.transcriptomics_encoder(transcriptomics_data)

        # Process chemical data
        c_features = self.chemical_encoder(chemical_data)

        # Handle different fusion types
        fusion_type = self.config["model"].get("fusion_type", "simple")

        if fusion_type == "simple":
            fused_features = self.fusion_module(t_features, c_features)
        elif fusion_type == "attention":
            # CrossModalAttention can do bi-directional or uni-directional attention
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
            # ModalityAttentionFusion takes a dictionary of features
            fused_features = self.fusion_module(
                {
                    "transcriptomics": t_features,
                    "chemical": c_features,
                }
            )

        # Make predictions
        return self.predictor(fused_features)


class ModelFactory:
    """
    Factory class for creating multimodal drug response models.

    This class simplifies model creation by instantiating model components
    based on configuration settings, ensuring compatibility between components,
    and handling any necessary preprocessing.
    """

    @staticmethod
    def create_model(config: Dict) -> MultimodalDrugResponseModel:
        """
        Create a MultimodalDrugResponseModel based on configuration.

        Args:
            config: Configuration dictionary with model specifications

        Returns:
            Initialized MultimodalDrugResponseModel
        """
        # Validate configuration
        ModelFactory._validate_config(config)

        # Create model
        return MultimodalDrugResponseModel(config)

    @staticmethod
    def _validate_config(config: Dict) -> None:
        """
        Validate model configuration.

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

        # Special case for biological encoder which needs pathway information
        if (
            model_config.get("transcriptomics_encoder_type") == "biological"
            and "pathway_groups" not in model_config
        ):
            raise ValueError("pathway_groups required for biological encoder")

        # Special case for multitask predictor which needs task names
        if (
            model_config.get("predictor_type") == "multitask"
            and "prediction_tasks" not in model_config
        ):
            raise ValueError("prediction_tasks required for multitask predictor")


class TranscriptomicsOnlyModel(nn.Module):
    """
    Unimodal model for drug response prediction using only transcriptomics data.

    This model is a simplification of the full multimodal model, using only
    the transcriptomics modality. It's useful for ablation studies and as a baseline.
    """

    def __init__(self, config: Dict):
        """
        Initialize TranscriptomicsOnlyModel.

        Args:
            config: Configuration dictionary
        """
        super(TranscriptomicsOnlyModel, self).__init__()
        model_config = config["model"]

        # Create transcriptomics encoder
        self.encoder = TranscriptomicEncoder(
            input_dim=model_config["transcriptomics_input_dim"],
            hidden_dims=model_config["transcriptomics_hidden_dims"],
            output_dim=model_config["transcriptomics_output_dim"],
            normalize=model_config.get("normalize", True),
            dropout=model_config.get("dropout", 0.3),
            activation=model_config.get("activation", "relu"),
        )

        # Create predictor
        self.predictor = ViabilityPredictor(
            input_dim=model_config["transcriptomics_output_dim"],
            hidden_dims=model_config["predictor_hidden_dims"],
            dropout=model_config.get("dropout", 0.3),
            activation=model_config.get("activation", "relu"),
            use_batch_norm=model_config.get("use_batch_norm", True),
            output_activation="sigmoid",
        )

        logger.info("Initialized TranscriptomicsOnlyModel")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transcriptomics-only model.

        Args:
            x: Transcriptomics data tensor

        Returns:
            Predicted viability
        """
        features = self.encoder(x)
        return self.predictor(features)


class ChemicalOnlyModel(nn.Module):
    """
    Unimodal model for drug response prediction using only chemical data.

    This model is a simplification of the full multimodal model, using only
    the chemical modality. It's useful for ablation studies and as a baseline.
    """

    def __init__(self, config: Dict):
        """
        Initialize ChemicalOnlyModel.

        Args:
            config: Configuration dictionary
        """
        super(ChemicalOnlyModel, self).__init__()
        model_config = config["model"]

        # Create chemical encoder based on type
        encoder_type = model_config.get("chemical_encoder_type", "descriptors")

        if encoder_type == "descriptors":
            self.encoder = IntegratedMolecularEncoder(
                descriptor_dim=model_config["chemical_input_dim"],
                hidden_dims=model_config["chemical_hidden_dims"],
                output_dim=model_config["chemical_output_dim"],
                dosage_integration=model_config.get("dosage_integration", "concat"),
                normalize=model_config.get("normalize", True),
                dropout=model_config.get("dropout", 0.3),
            )
        elif encoder_type == "smiles":
            self.encoder = SMILESWithDosageEncoder(
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
            raise ValueError(
                f"Unsupported chemical encoder type for unimodal model: {encoder_type}"
            )

        # Create predictor
        self.predictor = ViabilityPredictor(
            input_dim=model_config["chemical_output_dim"],
            hidden_dims=model_config["predictor_hidden_dims"],
            dropout=model_config.get("dropout", 0.3),
            activation=model_config.get("activation", "relu"),
            use_batch_norm=model_config.get("use_batch_norm", True),
            output_activation="sigmoid",
        )

        logger.info(f"Initialized ChemicalOnlyModel with encoder type {encoder_type}")

    def forward(self, x: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """
        Forward pass for chemical-only model.

        Args:
            x: Chemical data tensor or dictionary with 'smiles' and 'dosage'

        Returns:
            Predicted viability
        """
        features = self.encoder(x)
        return self.predictor(features)
