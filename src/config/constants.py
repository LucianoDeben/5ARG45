# config/constants.py
from enum import Enum
from typing import Dict

# Schema version for backward compatibility
CURRENT_SCHEMA_VERSION = "1.0.0"

# Required configuration sections
REQUIRED_CONFIG_SECTIONS = {"data", "model", "training", "chemical", "experiment"}


class FeatureSpace(str, Enum):
    """Valid gene feature spaces for L1000 data."""

    LANDMARK = "landmark"
    BEST_INFERRED = "best inferred"
    INFERRED = "inferred"


class Normalization(str, Enum):
    """Data normalization strategies."""

    ZSCORE = "zscore"
    MINMAX = "minmax"
    ROBUST = "robust"


class Optimizer(str, Enum):
    """Valid optimization algorithms."""

    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"
    RADAM = "radam"


class LossFunction(str, Enum):
    """Valid loss functions."""

    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    QUANTILE = "quantile"


class ChemicalRepresentation(str, Enum):
    """Valid molecular representation types."""

    FINGERPRINT = "fingerprint"
    SMILES_SEQUENCE = "smiles_sequence"
    GRAPH = "graph"
    DESCRIPTORS = "descriptors"


class Activation(str, Enum):
    """Valid activation functions."""

    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    GELU = "gelu"
    SELU = "selu"
    ELU = "elu"


class FusionStrategy(str, Enum):
    """Valid multimodal fusion strategies."""

    CONCAT = "concat"
    ATTENTION = "attention"
    CROSS_ATTENTION = "cross_attention"
    GATED = "gated"
    BILINEAR = "bilinear"


# Convert enums to sets for backward compatibility
VALID_FEATURE_SPACES = {e.value for e in FeatureSpace}
VALID_NORMALIZATIONS = {e.value for e in Normalization}
VALID_OPTIMIZERS = {e.value for e in Optimizer}
VALID_LOSS_FUNCTIONS = {e.value for e in LossFunction}
VALID_CHEMICAL_REPRESENTATIONS = {e.value for e in ChemicalRepresentation}
VALID_ACTIVATIONS = {e.value for e in Activation}
VALID_FUSION_STRATEGIES = {e.value for e in FusionStrategy}

# Default paths with environment variable references
DEFAULT_PATHS = {
    "data_dir": "${DATA_DIR:-data}",
    "model_dir": "${MODEL_DIR:-models/saved}",
    "log_dir": "${LOG_DIR:-logs}",
    "results_dir": "${RESULTS_DIR:-results}",
}

# Logging configuration
LOGGING_CONFIG = {
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "level": "INFO",
}
