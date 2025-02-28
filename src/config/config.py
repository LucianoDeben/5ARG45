import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TranscriptomicsConfig:
    """Configuration for transcriptomics encoder."""

    input_dim: int = 12328  # Default based on LINCS data
    hidden_dims: list = field(default_factory=lambda: [1024, 512, 256])
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    dimensionality_reduction: Optional[str] = None  # 'pca', 'umap', 'tf_activity', etc.
    dim_reduction_components: int = 128


@dataclass
class ChemicalConfig:
    """Configuration for chemical encoder."""

    input_type: str = "descriptors"  # 'descriptors', 'fingerprints', 'smiles'
    input_dim: int = 10  # Default for basic descriptors
    hidden_dims: list = field(default_factory=lambda: [128, 64])
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    fingerprint_size: int = 2048
    fingerprint_radius: int = 3


@dataclass
class FusionConfig:
    """Configuration for fusion module."""

    fusion_type: str = "concat"  # 'concat', 'attention', 'gated'
    hidden_dims: list = field(default_factory=lambda: [256, 128])
    dropout_rate: float = 0.3
    use_batch_norm: bool = True


@dataclass
class PredictionConfig:
    """Configuration for prediction head."""

    hidden_dims: list = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.3
    use_batch_norm: bool = True


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    type: str = "adam"  # 'adam', 'sgd', 'adamw'
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9  # Only for SGD


@dataclass
class LossConfig:
    """Configuration for loss function."""

    type: str = "mse"  # 'mse', 'mae', 'huber'
    delta: float = 1.0  # For Huber loss


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    type: Optional[str] = None  # 'step', 'cosine', 'plateau'
    step_size: int = 10  # For StepLR
    gamma: float = 0.1  # For StepLR
    T_max: int = 100  # For CosineAnnealingLR
    eta_min: float = 0.0  # For CosineAnnealingLR
    patience: int = 5  # For ReduceLROnPlateau
    factor: float = 0.1  # For ReduceLROnPlateau


@dataclass
class DataConfig:
    """Configuration for data processing."""

    lincs_file: str = "data/processed/LINCS.gctx"
    ctrp_file: str = "data/raw/CTRP_viability.csv"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    batch_size: int = 32
    num_workers: int = 4
    max_samples: Optional[int] = None  # Limit number of samples (for debugging)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    experiment_name: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"  # 'cuda' or 'cpu'
    num_epochs: int = 100
    metrics: list = field(default_factory=lambda: ["mse", "mae", "r2"])
    early_stopping_patience: int = 10
    grad_clip_value: Optional[float] = None
    mixed_precision: bool = False
    checkpoint_interval: int = 1
    log_interval: int = 10


@dataclass
class Config:
    """Main configuration for the multimodal drug response model."""

    transcriptomics: TranscriptomicsConfig = field(
        default_factory=TranscriptomicsConfig
    )
    chemical: ChemicalConfig = field(default_factory=ChemicalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Metadata
    version: str = "1.0.0"
    description: str = "Multimodal Drug Response Prediction Model"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save(self, path: str) -> None:
        """
        Save configuration to file.

        Args:
            path: Path to save configuration
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        elif path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=4)
        else:
            raise ValueError(f"Unsupported file format: {path}")

        logger.info(f"Configuration saved to {path}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config instance
        """
        # Create nested dataclass instances
        transcriptomics = TranscriptomicsConfig(
            **config_dict.get("transcriptomics", {})
        )
        chemical = ChemicalConfig(**config_dict.get("chemical", {}))
        fusion = FusionConfig(**config_dict.get("fusion", {}))
        prediction = PredictionConfig(**config_dict.get("prediction", {}))
        optimizer = OptimizerConfig(**config_dict.get("optimizer", {}))
        loss = LossConfig(**config_dict.get("loss", {}))
        scheduler = SchedulerConfig(**config_dict.get("scheduler", {}))
        data = DataConfig(**config_dict.get("data", {}))
        training = TrainingConfig(**config_dict.get("training", {}))

        # Get metadata
        version = config_dict.get("version", "1.0.0")
        description = config_dict.get(
            "description", "Multimodal Drug Response Prediction Model"
        )
        created_at = config_dict.get("created_at", datetime.now().isoformat())
        id = config_dict.get("id", str(uuid.uuid4()))

        return cls(
            transcriptomics=transcriptomics,
            chemical=chemical,
            fusion=fusion,
            prediction=prediction,
            optimizer=optimizer,
            loss=loss,
            scheduler=scheduler,
            data=data,
            training=training,
            version=version,
            description=description,
            created_at=created_at,
            id=id,
        )

    @classmethod
    def load(cls, path: str) -> "Config":
        """
        Load configuration from file.

        Args:
            path: Path to configuration file

        Returns:
            Config instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
        elif path.endswith(".json"):
            with open(path, "r") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path}")

        logger.info(f"Configuration loaded from {path}")
        return cls.from_dict(config_dict)


class ConfigValidator:
    """Validator for configuration."""

    @staticmethod
    def validate(config: Config) -> bool:
        """
        Validate configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid, raises ValueError otherwise
        """
        # Validate transcriptomics config
        if config.transcriptomics.input_dim <= 0:
            raise ValueError(
                f"Invalid transcriptomics input dimension: {config.transcriptomics.input_dim}"
            )

        # Validate chemical config
        if config.chemical.input_type not in ["descriptors", "fingerprints", "smiles"]:
            raise ValueError(
                f"Invalid chemical input type: {config.chemical.input_type}"
            )

        # Validate fusion config
        if config.fusion.fusion_type not in ["concat", "attention", "gated"]:
            raise ValueError(f"Invalid fusion type: {config.fusion.fusion_type}")

        # Validate optimizer config
        if config.optimizer.type not in ["adam", "sgd", "adamw"]:
            raise ValueError(f"Invalid optimizer type: {config.optimizer.type}")

        # Validate loss config
        if config.loss.type not in ["mse", "mae", "huber"]:
            raise ValueError(f"Invalid loss type: {config.loss.type}")

        # Validate scheduler config
        if config.scheduler.type is not None and config.scheduler.type not in [
            "step",
            "cosine",
            "plateau",
        ]:
            raise ValueError(f"Invalid scheduler type: {config.scheduler.type}")

        # Validate training config
        if config.training.device not in ["cuda", "cpu"]:
            raise ValueError(f"Invalid device: {config.training.device}")

        # Validate data config
        if not os.path.exists(config.data.lincs_file):
            logger.warning(f"LINCS file not found: {config.data.lincs_file}")

        if not os.path.exists(config.data.ctrp_file):
            logger.warning(f"CTRP file not found: {config.data.ctrp_file}")

        return True


def get_default_config() -> Config:
    """
    Get default configuration.

    Returns:
        Default Config instance
    """
    return Config()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create default configuration
    config = get_default_config()

    # Modify configuration
    config.transcriptomics.hidden_dims = [1024, 512, 256]
    config.chemical.input_type = "fingerprints"
    config.fusion.fusion_type = "attention"
    config.optimizer.type = "adamw"
    config.optimizer.learning_rate = 1e-3
    config.scheduler.type = "cosine"
    config.scheduler.T_max = 100
    config.training.mixed_precision = True
    config.training.experiment_name = "multimodal_drug_response_v1"

    # Validate configuration
    ConfigValidator.validate(config)

    # Save configuration
    config.save("config/default_config.yaml")

    # Load configuration
    loaded_config = Config.load("config/default_config.yaml")

    print("Configuration loaded successfully.")
    print(f"Experiment: {loaded_config.training.experiment_name}")
    print(
        f"Transcriptomics hidden dimensions: {loaded_config.transcriptomics.hidden_dims}"
    )
