# config/schema.py
from typing import List, Optional, Union

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
)


class DataConfig(BaseModel):
    """Data configuration settings."""

    gctx_file: str = Field(..., description="Path to GCTX data file")
    feature_space: Union[str, List[str]] = Field(
        "landmark", description="Gene feature space(s) to use"
    )
    nrows: Optional[int] = Field(
        None, description="Number of rows to load (None for all)"
    )
    normalize: Optional[str] = Field(None, description="Normalization strategy")

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


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    transcriptomics_input_dim: int = Field(..., gt=0)
    transcriptomics_hidden_dims: List[int] = Field(..., min_items=1)
    transcriptomics_output_dim: int = Field(..., gt=0)
    chemical_input_dim: int = Field(..., gt=0)
    chemical_hidden_dims: List[int] = Field(..., min_items=1)
    chemical_output_dim: int = Field(..., gt=0)
    fusion_output_dim: int = Field(..., gt=0)
    fusion_strategy: str = Field(..., description="Feature fusion strategy")
    predictor_hidden_dims: List[int] = Field(..., min_items=1)
    normalize: bool = True
    dropout: float = Field(..., ge=0.0, le=1.0)
    activation: str = Field(..., description="Activation function")
    use_batch_norm: bool = True

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

    batch_size: int = Field(..., gt=0)
    epochs: int = Field(..., gt=0)
    learning_rate: float = Field(..., gt=0.0)
    optimizer: str = Field(..., description="Optimizer name")
    loss: str = Field(..., description="Loss function name")
    test_size: float = Field(..., gt=0.0, lt=1.0)
    val_size: float = Field(..., gt=0.0, lt=1.0)
    random_state: int = 42
    group_by: Optional[str] = None

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

    representation: str = Field(..., description="Molecular representation type")
    fingerprint_size: int = Field(..., gt=0)
    radius: int = Field(..., ge=0, le=5)

    @field_validator("representation")
    def validate_representation(cls, v):
        if v not in VALID_CHEMICAL_REPRESENTATIONS:
            raise ValueError(f"Invalid chemical representation: {v}")
        return v


class ExperimentConfig(BaseModel):
    """Experiment tracking configuration."""

    project_name: str
    run_name: Optional[str] = None
    track: bool = True
    tags: Optional[List[str]] = None
    version: str = "1.0.0"


class CompleteConfig(BaseModel):
    """Complete configuration with all sections."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    chemical: ChemicalConfig
    experiment: ExperimentConfig
