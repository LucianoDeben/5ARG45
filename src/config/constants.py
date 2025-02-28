# config/constants.py
from typing import Dict, Set

# Required configuration sections
REQUIRED_CONFIG_SECTIONS = {"data", "model", "training", "chemical", "experiment"}

# Valid options for different parameters
VALID_FEATURE_SPACES: Set[str] = {"landmark", "best inferred", "inferred"}

VALID_NORMALIZATIONS: Set[str] = {"zscore", "minmax", "robust"}

VALID_OPTIMIZERS: Set[str] = {"adam", "sgd", "adamw", "radam"}

VALID_LOSS_FUNCTIONS: Set[str] = {"mse", "mae", "huber", "quantile"}

VALID_CHEMICAL_REPRESENTATIONS: Set[str] = {
    "fingerprint",
    "smiles_sequence",
    "graph",
    "descriptors",
}

VALID_ACTIVATIONS: Set[str] = {"relu", "leaky_relu", "gelu", "selu", "elu"}

VALID_FUSION_STRATEGIES: Set[str] = {
    "concat",
    "attention",
    "cross_attention",
    "gated",
    "bilinear",
}

# Schema version for backward compatibility
CURRENT_SCHEMA_VERSION = "1.0.0"

# Default paths
DEFAULT_PATHS = {
    "data_dir": "data/processed",
    "model_dir": "models/saved",
    "log_dir": "logs",
    "results_dir": "results",
}

# Logging configuration
LOGGING_CONFIG = {
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "level": "INFO",
}
