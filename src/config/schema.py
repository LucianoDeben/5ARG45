# src/config/schema.py
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from ..config.constants import (
    Activation,
    Device,
    FeatureSpace,
    FusionStrategy,
    ImputationStrategy,
    LRScheduler,
    LossFunction,
    MolecularRepresentation,
    Normalization,
    Optimizer,
    RegressionMetrics,
)


class PathsConfig(BaseModel):
    data_dir: str = Field("data", description="Base directory for data files")
    model_dir: str = Field("models/saved", description="Directory for saved models")
    log_dir: str = Field("logs", description="Directory for log files")
    results_dir: str = Field(
        "results", description="Directory for results and analysis"
    )
    checkpoint_dir: str = Field(
        "models/saved/checkpoints", description="Directory for model checkpoints"
    )

    @field_validator(
        "data_dir", "model_dir", "log_dir", "results_dir", "checkpoint_dir"
    )
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
    matching_strategy: str = Field(
        "parallel", description="Strategy for matching LINCS and CTRP data"
    )
    max_workers: Optional[int] = Field(
        None, description="Max workers for multiprocessing; null uses cpu_count * 2"
    )
    chunk_size: int = Field(10000, description="Chunk size for data processing")
    feature_space: Union[FeatureSpace, List[FeatureSpace]] = Field(
        FeatureSpace.LANDMARK, description="Gene feature space"
    )
    nrows: Optional[int] = Field(1000, description="Number of rows to load")
    normalize: Optional[Normalization] = Field(
        Normalization.NONE, description="Normalization strategy"
    )
    random_seed: int = Field(42, description="Random seed for reproducibility")
    cache_data: bool = Field(True, description="Whether to cache preprocessed data")
    use_multiprocessing: bool = Field(True, description="Use multiprocessing")
    num_workers: int = Field(0, description="Number of worker processes")
    imputation_strategy: ImputationStrategy = Field(
        ImputationStrategy.MEAN, description="Strategy for imputing missing values"
    )
    handle_outliers: bool = Field(False, description="Whether to handle outliers")
    outlier_threshold: float = Field(3.0, description="Threshold for outlier detection")


class LRSchedulerConfig(BaseModel):
    type: LRScheduler = Field(LRScheduler.COSINE, description="Learning rate scheduler type")
    warmup_epochs: int = Field(5, description="Number of warmup epochs")
    min_lr: float = Field(1e-6, description="Minimum learning rate")
    step_size: int = Field(10, description="Step size for StepLR scheduler")
    gamma: float = Field(0.1, description="Gamma factor for StepLR scheduler")


class TrainingConfig(BaseModel):
    batch_size: int = Field(128, gt=0, description="Training batch size")
    epochs: int = Field(10, gt=0, description="Number of training epochs")
    num_runs: int = Field(5, gt=0, description="Number of training runs")
    learning_rate: float = Field(0.001, gt=0.0, description="Initial learning rate")
    optimizer: Optimizer = Field(Optimizer.ADAM, description="Optimizer name")
    loss: LossFunction = Field(LossFunction.MSE, description="Loss function name")
    test_size: float = Field(0.4, gt=0.0, lt=1.0, description="Proportion for testing")
    val_size: float = Field(
        0.1, gt=0.0, lt=1.0, description="Proportion for validation"
    )
    random_state: int = Field(42, description="Random seed for data splitting")
    group_by: Optional[str] = Field(
        "cell_mfc_name", description="Column for group-based splitting"
    )
    stratify_by: Optional[str] = Field(
        "viability", description="Column for stratified splitting"
    )
    lr_scheduler: LRSchedulerConfig = Field(
        default_factory=LRSchedulerConfig, description="Learning rate scheduler"
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


class MolecularConfig(BaseModel):
    representation: MolecularRepresentation = Field(
        MolecularRepresentation.FINGERPRINT, description="Molecular representation type"
    )
    fingerprint_size: int = Field(
        1024, gt=0, description="Size of molecular fingerprints"
    )
    radius: int = Field(
        2, ge=0, le=5, description="Radius for Morgan/ECFP fingerprints"
    )
    use_chirality: bool = Field(
        True, description="Whether to use stereochemical information"
    )
    use_features: bool = Field(
        True, description="Whether to use additional molecular features"
    )
    sanitize: bool = Field(True, description="Whether to sanitize molecules")


class ModelConfig(BaseModel):
    transcriptomics_input_dim: int = Field(
        978, gt=0, description="Input dimension for transcriptomics data"
    )
    transcriptomics_hidden_dims: List[int] = Field(
        [512, 256],
        min_items=1,
        description="Hidden layer dimensions for transcriptomics encoder",
    )
    transcriptomics_output_dim: int = Field(
        128, gt=0, description="Output dimension for transcriptomics encoder"
    )
    molecular_input_dim: int = Field(
        1025, gt=0, description="Input dimension for molecular data"
    )  
    molecular_hidden_dims: List[int] = Field(
        [256, 128], min_items=1, description="Hidden layer dimensions for molecular encoder"
    )  
    molecular_output_dim: int = Field(
        128, gt=0, description="Output dimension for molecular encoder"
    )  
    fusion_output_dim: int = Field(
        256, gt=0, description="Output dimension after fusion"
    )
    predictor_hidden_dims: List[int] = Field(
        [128, 64], min_items=1, description="Hidden layer dimensions for predictor"
    )
    fusion_strategy: FusionStrategy = Field(
        FusionStrategy.CONCAT, description="Feature fusion strategy"
    )
    activation: Activation = Field(Activation.RELU, description="Activation function")
    dropout: float = Field(0.3, ge=0.0, le=1.0, description="Dropout probability")
    use_batch_norm: bool = Field(True, description="Whether to use batch normalization")
    layer_norm: bool = Field(True, description="Whether to use layer normalization")
    residual_connections: bool = Field(
        True, description="Whether to use residual connections"
    )

    @model_validator(mode="after")
    def validate_fusion_output_dim(self):
        if self.fusion_strategy == FusionStrategy.CONCAT:
            expected_dim = self.transcriptomics_output_dim + self.molecular_output_dim
            if self.fusion_output_dim != expected_dim:
                raise ValueError(
                    f"For concat fusion, fusion_output_dim should be equal to "
                    f"transcriptomics_output_dim + molecular_output_dim "
                    f"({self.transcriptomics_output_dim} + {self.molecular_output_dim} = {expected_dim})"
                )
        return self


class EvaluationConfig(BaseModel):
    metrics: List[str] = Field(
        [RegressionMetrics.R2, RegressionMetrics.RMSE,RegressionMetrics.MAE, RegressionMetrics.PEARSON], description="Evaluation metrics"
    )
    loss: str = Field(LossFunction.MSE, description="Evaluation loss function")
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


class ExperimentConfig(BaseModel):
    project_name: str = Field(
        "multimodal_drug_response", description="Project name for experiment tracking"
    )
    run_name: Optional[str] = Field(
        None, description="Run name for experiment tracking"
    )
    track: bool = Field(True, description="Whether to track experiments")
    tags: List[str] = Field(
        ["multimodal", "drug-response"], description="Tags for experiment tracking"
    )
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
    log_level: str = Field("WARNING", description="Logging level")

class InferenceConfig(BaseModel):
    device: str = Field(Device.GPU, description="Inference device")
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

class InterpretationConfig(BaseModel):
    enabled: bool = Field(True, description="Whether to enable model interpretation")
    attribution_methods: List[str] = Field(
        ["integrated_gradients", "feature_ablation", "deep_lift"],
        description="Attribution methods",
    )
    n_samples: int = Field(5, description="Number of samples for interpretation")
    top_k: int = Field(20, description="Number of top features to highlight")
    visualize_features: bool = Field(True, description="Whether to visualize features")
    feature_naming: Dict[str, Optional[str]] = Field(
        {"transcriptomics": "data/raw/LINCS/geneinfo_beta.txt", "molecular": None},
        description="Paths to feature name mappings",
    )
    save_attributions: bool = Field(
        True, description="Whether to save raw attribution values"
    )
    generate_figures: bool = Field(True, description="Whether to generate figures")


class CompleteConfig(BaseModel):
    paths: PathsConfig = Field(
        default_factory=PathsConfig, description="Path configuration"
    )
    data: DataConfig
    training: TrainingConfig
    molecular: MolecularConfig
    model: ModelConfig
    experiment: ExperimentConfig
    evaluation: Optional[EvaluationConfig] = Field(default_factory=EvaluationConfig)
    inference: Optional[InferenceConfig] = Field(default_factory=InferenceConfig)
    deployment: Optional[DeploymentConfig] = Field(default_factory=DeploymentConfig)
    interpretation: Optional[InterpretationConfig] = Field(
        default_factory=InterpretationConfig
    )
