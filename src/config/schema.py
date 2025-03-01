from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Literal

from config.constants import (
    VALID_ACTIVATIONS,
    VALID_CHEMICAL_REPRESENTATIONS,
    VALID_FEATURE_SPACES,
    VALID_FUSION_STRATEGIES,
    VALID_LOSS_FUNCTIONS,
    VALID_NORMALIZATIONS,
    VALID_OPTIMIZERS,
    Activation,
    ChemicalRepresentation,
    FeatureSpace,
    FusionStrategy,
    LossFunction,
    Normalization,
    Optimizer,
)


class PathsConfig(BaseModel):
    """File path configuration."""

    data_dir: str = Field("data", description="Base directory for data files")
    model_dir: str = Field("models/saved", description="Directory for saved models")
    log_dir: str = Field("logs", description="Directory for log files")
    results_dir: str = Field(
        "results", description="Directory for results and analysis"
    )


class DataConfig(BaseModel):
    """Data configuration settings."""

    gctx_file: str = Field(
        ..., description="Path to GCTX data file containing gene expression data"
    )
    geneinfo_file: str = Field(..., description="Path to gene information file")
    siginfo_file: str = Field(..., description="Path to signature information file")
    feature_space: Union[str, List[str]] = Field(
        "landmark",
        description="Gene feature space(s) to use (landmark, best inferred, or inferred)",
    )
    nrows: Optional[int] = Field(
        None, description="Number of rows to load (None for all)"
    )
    normalize: Optional[str] = Field(
        None, description="Normalization strategy (zscore, minmax, robust)"
    )
    random_seed: int = Field(42, description="Random seed for reproducibility")
    cache_data: bool = Field(True, description="Whether to cache preprocessed data")
    use_multiprocessing: bool = Field(
        True, description="Whether to use multiprocessing for data loading"
    )
    num_workers: int = Field(
        4, description="Number of worker processes for data loading"
    )

    @field_validator("feature_space")
    def validate_feature_space(cls, v):
        if isinstance(v, str):
            if v not in VALID_FEATURE_SPACES:
                raise ValueError(f"Invalid feature_space: {v}")
        elif isinstance(v, list):
            if not all(fs in VALID_FEATURE_SPACES for fs in v):
                raise ValueError(f"Invalid feature_space in list: {v}")
        return v

    @field_validator("normalize")
    def validate_normalize(cls, v):
        if v is not None and v not in VALID_NORMALIZATIONS:
            raise ValueError(f"Invalid normalization: {v}")
        return v


class LRSchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    type: str = Field(
        "cosine", description="Learning rate scheduler type (cosine, step, exponential)"
    )
    warmup_epochs: int = Field(5, description="Number of warmup epochs")
    min_lr: float = Field(1e-6, description="Minimum learning rate")


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    transcriptomics_input_dim: int = Field(
        ..., gt=0, description="Input dimension for transcriptomics data"
    )
    transcriptomics_hidden_dims: List[int] = Field(
        ...,
        min_items=1,
        description="Hidden layer dimensions for transcriptomics encoder",
    )
    transcriptomics_output_dim: int = Field(
        ..., gt=0, description="Output dimension for transcriptomics encoder"
    )
    chemical_input_dim: int = Field(
        ..., gt=0, description="Input dimension for chemical data"
    )
    chemical_hidden_dims: List[int] = Field(
        ..., min_items=1, description="Hidden layer dimensions for chemical encoder"
    )
    chemical_output_dim: int = Field(
        ..., gt=0, description="Output dimension for chemical encoder"
    )
    fusion_output_dim: int = Field(
        ..., gt=0, description="Output dimension after fusion"
    )
    fusion_strategy: str = Field(
        ..., description="Feature fusion strategy (concat, attention, etc.)"
    )
    predictor_hidden_dims: List[int] = Field(
        ..., min_items=1, description="Hidden layer dimensions for predictor network"
    )
    normalize: bool = Field(True, description="Whether to normalize inputs")
    dropout: float = Field(..., ge=0.0, le=1.0, description="Dropout probability")
    activation: str = Field(..., description="Activation function (relu, gelu, etc.)")
    use_batch_norm: bool = Field(True, description="Whether to use batch normalization")
    layer_norm: bool = Field(True, description="Whether to use layer normalization")
    residual_connections: bool = Field(
        True, description="Whether to use residual connections"
    )

    @field_validator("fusion_strategy")
    def validate_fusion(cls, v):
        if v not in VALID_FUSION_STRATEGIES:
            raise ValueError(f"Invalid fusion strategy: {v}")
        return v

    @field_validator("activation")
    def validate_activation(cls, v):
        if v not in VALID_ACTIVATIONS:
            raise ValueError(f"Invalid activation: {v}")
        return v


class TrainingConfig(BaseModel):
    """Training configuration settings."""

    batch_size: int = Field(..., gt=0, description="Training batch size")
    epochs: int = Field(..., gt=0, description="Number of training epochs")
    learning_rate: float = Field(..., gt=0.0, description="Initial learning rate")
    optimizer: str = Field(..., description="Optimizer name (adam, sgd, adamw, etc.)")
    loss: str = Field(..., description="Loss function name (mse, mae, huber, etc.)")
    test_size: float = Field(
        ..., gt=0.0, lt=1.0, description="Proportion of data to use for testing"
    )
    val_size: float = Field(
        ..., gt=0.0, lt=1.0, description="Proportion of data to use for validation"
    )
    random_state: int = Field(42, description="Random seed for data splitting")
    group_by: Optional[str] = Field(
        None, description="Column name to use for group-based splitting"
    )
    stratify_by: Optional[str] = Field(
        None, description="Column name to use for stratified splitting"
    )
    lr_scheduler: Optional[LRSchedulerConfig] = Field(
        None, description="Learning rate scheduler configuration"
    )
    early_stopping: bool = Field(True, description="Whether to use early stopping")
    patience: int = Field(
        10, description="Number of epochs to wait for improvement before early stopping"
    )
    min_delta: float = Field(
        0.001, description="Minimum change to qualify as improvement"
    )
    clip_grad_norm: bool = Field(True, description="Whether to clip gradient norms")
    max_grad_norm: float = Field(1.0, description="Maximum gradient norm for clipping")
    use_amp: bool = Field(True, description="Whether to use automatic mixed precision")
    weight_decay: float = Field(0.01, description="Weight decay (L2 penalty)")
    label_smoothing: float = Field(0.1, description="Label smoothing factor")

    @field_validator("optimizer")
    def validate_optimizer(cls, v):
        if v not in VALID_OPTIMIZERS:
            raise ValueError(f"Invalid optimizer: {v}")
        return v

    @field_validator("loss")
    def validate_loss(cls, v):
        if v not in VALID_LOSS_FUNCTIONS:
            raise ValueError(f"Invalid loss function: {v}")
        return v

    @model_validator(mode="after")
    def validate_split_sizes(self):
        if self.test_size + self.val_size >= 1.0:
            raise ValueError("Sum of test_size and val_size must be less than 1.0")
        return self


class ChemicalConfig(BaseModel):
    """Chemical processing configuration."""

    representation: str = Field(
        ...,
        description="Molecular representation type (fingerprint, smiles_sequence, etc.)",
    )
    fingerprint_size: int = Field(
        ..., gt=0, description="Size of molecular fingerprints"
    )
    radius: int = Field(
        ..., ge=0, le=5, description="Radius for Morgan/ECFP fingerprints"
    )
    use_chirality: bool = Field(
        True, description="Whether to use stereochemical information"
    )
    use_features: bool = Field(
        True, description="Whether to use additional chemical features"
    )
    sanitize: bool = Field(
        True, description="Whether to sanitize molecules before processing"
    )

    @field_validator("representation")
    def validate_representation(cls, v):
        if v not in VALID_CHEMICAL_REPRESENTATIONS:
            raise ValueError(f"Invalid chemical representation: {v}")
        return v


class ExperimentConfig(BaseModel):
    """Experiment tracking configuration."""

    project_name: str = Field(..., description="Project name for experiment tracking")
    run_name: Optional[str] = Field(
        None, description="Run name for experiment tracking"
    )
    track: bool = Field(True, description="Whether to track experiments")
    tags: Optional[List[str]] = Field(None, description="Tags for experiment tracking")
    version: str = Field("1.0.0", description="Configuration schema version")
    save_checkpoints: bool = Field(
        True, description="Whether to save model checkpoints"
    )
    checkpoint_freq: int = Field(
        5, description="Frequency (in epochs) for saving checkpoints"
    )
    keep_n_checkpoints: int = Field(
        3, description="Number of latest checkpoints to keep"
    )


class CompleteConfig(BaseModel):
    """Complete configuration with all sections."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    chemical: ChemicalConfig
    experiment: ExperimentConfig
    paths: Optional[PathsConfig] = Field(None, description="Path configuration")
