# src/config/schema.py
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from ..config.constants import (
    Activation,
    ChemicalRepresentation,
    FeatureSpace,
    FusionStrategy,
    LossFunction,
    Normalization,
    Optimizer,
)


class PathsConfig(BaseModel):
    data_dir: str = Field("data", description="Base directory for data files")
    model_dir: str = Field("models/saved", description="Directory for saved models")
    log_dir: str = Field("logs", description="Directory for log files")
    results_dir: str = Field(
        "results", description="Directory for results and analysis"
    )

    @field_validator("data_dir", "model_dir", "log_dir", "results_dir")
    def validate_path(cls, v):
        if not isinstance(v, str):
            raise ValueError(f"Path must be a string, got {type(v)}")
        return v


class DataConfig(BaseModel):
    gctx_file: str = Field(..., description="Path to GCTX data file")
    geneinfo_file: str = Field(..., description="Path to gene information file")
    siginfo_file: str = Field(..., description="Path to signature information file")
    curves_post_qc: str = Field(..., description="Path to curves post QC file")
    per_cpd_post_qc: str = Field(..., description="Path to per compound post QC file")
    per_experiment: str = Field(..., description="Path to per experiment file")
    per_compound: str = Field(..., description="Path to per compound file")
    per_cell_line: str = Field(..., description="Path to per cell line file")
    output_path: Optional[str] = Field(None, description="Path for output dataset")
    feature_space: Union[FeatureSpace, List[FeatureSpace]] = Field(
        FeatureSpace.LANDMARK, description="Gene feature space"
    )
    nrows: Optional[int] = Field(None, description="Number of rows to load")
    normalize: Optional[Normalization] = Field(
        None, description="Normalization strategy"
    )
    random_seed: int = Field(42, description="Random seed for reproducibility")
    cache_data: bool = Field(True, description="Whether to cache preprocessed data")
    use_multiprocessing: bool = Field(True, description="Use multiprocessing")
    num_workers: int = Field(4, description="Number of worker processes")
    matching_strategy: Optional[str] = Field(
        "parallel", description="Strategy for matching LINCS and CTRP data"
    )
    max_workers: Optional[int] = Field(
        None, description="Max workers for multiprocessing; null uses cpu_count * 2"
    )
    chunk_size: Optional[int] = Field(
        10000, description="Chunk size for data processing"
    )


class LRSchedulerConfig(BaseModel):
    type: str = Field("cosine", description="Learning rate scheduler type")
    warmup_epochs: int = Field(5, description="Number of warmup epochs")
    min_lr: float = Field(1e-6, description="Minimum learning rate")


class ModelConfig(BaseModel):
    transcriptomics_input_dim: int = Field(
        ..., gt=0, description="Input dimension for transcriptomics data"
    )
    transcriptomics_hidden_dims: List[int] = Field(
        ..., min_items=1, description="Hidden layer dimensions"
    )
    transcriptomics_output_dim: int = Field(
        ..., gt=0, description="Output dimension for transcriptomics encoder"
    )
    chemical_input_dim: int = Field(
        ..., gt=0, description="Input dimension for chemical data"
    )
    chemical_hidden_dims: List[int] = Field(
        ..., min_items=1, description="Hidden layer dimensions"
    )
    chemical_output_dim: int = Field(
        ..., gt=0, description="Output dimension for chemical encoder"
    )
    fusion_output_dim: int = Field(
        ..., gt=0, description="Output dimension after fusion"
    )
    fusion_strategy: FusionStrategy = Field(
        FusionStrategy.CONCAT, description="Feature fusion strategy"
    )
    predictor_hidden_dims: List[int] = Field(
        ..., min_items=1, description="Hidden layer dimensions for predictor"
    )
    normalize: bool = Field(True, description="Whether to normalize inputs")
    dropout: float = Field(0.3, ge=0.0, le=1.0, description="Dropout probability")
    activation: Activation = Field(Activation.RELU, description="Activation function")
    use_batch_norm: bool = Field(True, description="Whether to use batch normalization")
    layer_norm: bool = Field(True, description="Whether to use layer normalization")
    residual_connections: bool = Field(
        True, description="Whether to use residual connections"
    )


class TrainingConfig(BaseModel):
    batch_size: int = Field(32, gt=0, description="Training batch size")
    epochs: int = Field(100, gt=0, description="Number of training epochs")
    learning_rate: float = Field(0.001, gt=0.0, description="Initial learning rate")
    optimizer: Optimizer = Field(Optimizer.ADAM, description="Optimizer name")
    loss: LossFunction = Field(LossFunction.MSE, description="Loss function name")
    test_size: float = Field(0.2, gt=0.0, lt=1.0, description="Proportion for testing")
    val_size: float = Field(
        0.1, gt=0.0, lt=1.0, description="Proportion for validation"
    )
    random_state: int = Field(42, description="Random seed for data splitting")
    group_by: Optional[str] = Field(
        None, description="Column for group-based splitting"
    )
    stratify_by: Optional[str] = Field(
        None, description="Column for stratified splitting"
    )
    lr_scheduler: LRSchedulerConfig = Field(
        default_factory=lambda: LRSchedulerConfig(),
        description="Learning rate scheduler",
    )
    early_stopping: bool = Field(True, description="Whether to use early stopping")
    patience: int = Field(10, description="Epochs to wait for improvement")
    min_delta: float = Field(0.001, description="Minimum change for improvement")
    clip_grad_norm: bool = Field(True, description="Whether to clip gradient norms")
    max_grad_norm: float = Field(1.0, description="Maximum gradient norm")
    use_amp: bool = Field(True, description="Whether to use automatic mixed precision")
    weight_decay: float = Field(0.01, description="Weight decay (L2 penalty)")
    label_smoothing: float = Field(0.1, description="Label smoothing factor")

    @model_validator(mode="after")
    def validate_split_sizes(self):
        if self.test_size + self.val_size >= 1.0:
            raise ValueError("Sum of test_size and val_size must be less than 1.0")
        return self


class ChemicalConfig(BaseModel):
    representation: ChemicalRepresentation = Field(
        ChemicalRepresentation.FINGERPRINT, description="Molecular representation type"
    )
    fingerprint_size: int = Field(
        2048, gt=0, description="Size of molecular fingerprints"
    )
    radius: int = Field(
        2, ge=0, le=5, description="Radius for Morgan/ECFP fingerprints"
    )
    use_chirality: bool = Field(
        True, description="Whether to use stereochemical information"
    )
    use_features: bool = Field(
        True, description="Whether to use additional chemical features"
    )
    sanitize: bool = Field(True, description="Whether to sanitize molecules")


class ExperimentConfig(BaseModel):
    project_name: str = Field(
        "multimodal_drug_response", description="Project name for experiment tracking"
    )
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


class EvaluationConfig(BaseModel):
    metrics: List[str] = Field(
        ["r2", "rmse", "mae", "pearson"], description="Evaluation metrics"
    )
    loss: str = Field("mse", description="Evaluation loss function")
    output_dir: str = Field(
        "results/eval", description="Directory for evaluation results"
    )
    visualization: Dict[str, Any] = Field(
        default_factory=lambda: {"dpi": 300, "figsize": [10, 8]}
    )


class InferenceConfig(BaseModel):
    device: str = Field("cuda", description="Inference device")
    max_ensemble: int = Field(5, description="Maximum ensemble models")
    output_path: str = Field(
        "results/predictions.csv", description="Output predictions path"
    )
    export_formats: List[str] = Field(
        ["pytorch", "onnx", "torchscript"], description="Model export formats"
    )


class DeploymentConfig(BaseModel):
    quantization: List[str] = Field(
        ["static", "dynamic"], description="Quantization strategies"
    )


class CompleteConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    chemical: ChemicalConfig
    experiment: ExperimentConfig
    paths: Optional[PathsConfig] = Field(
        default_factory=lambda: PathsConfig(), description="Path configuration"
    )
    evaluation: Optional[EvaluationConfig] = Field(
        default_factory=lambda: EvaluationConfig()
    )
    inference: Optional[InferenceConfig] = Field(
        default_factory=lambda: InferenceConfig()
    )
    deployment: Optional[DeploymentConfig] = Field(
        default_factory=lambda: DeploymentConfig()
    )


@model_validator(mode="after")
def validate_model_dimensions(self):
    """Validate that model dimensions are compatible."""
    if self.fusion_strategy == FusionStrategy.CONCAT:
        expected_dim = self.transcriptomics_output_dim + self.chemical_output_dim
        if self.fusion_output_dim != expected_dim:
            raise ValueError(
                f"For concat fusion, fusion_output_dim should be equal to "
                f"transcriptomics_output_dim + chemical_output_dim "
                f"({self.transcriptomics_output_dim} + {self.chemical_output_dim} = {expected_dim})"
            )
    return self
