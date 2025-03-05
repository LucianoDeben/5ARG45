# src/config/constants.py
import logging
from enum import Enum
from typing import Any, Dict
from venv import logger


def check_config_types(config: Dict[str, Any], schema_dict: Dict[str, type]) -> bool:
    """Validate configuration value types against a schema dictionary."""
    for key, expected_type in schema_dict.items():
        if key in config:
            if isinstance(expected_type, dict) and isinstance(config[key], dict):
                if not check_config_types(config[key], expected_type):
                    return False
            elif not isinstance(config[key], expected_type):
                logger.error(
                    f"Config key '{key}' has wrong type: {type(config[key])}, expected {expected_type}"
                )
                return False
    return True


CURRENT_SCHEMA_VERSION = "1.0.0"
REQUIRED_CONFIG_SECTIONS = {"data", "model", "training", "chemical", "experiment"}


class FeatureSpace(str, Enum):
    LANDMARK = "landmark"
    BEST_INFERRED = "best inferred"
    INFERRED = "inferred"


class Normalization(str, Enum):
    ZSCORE = "zscore"
    MINMAX = "minmax"
    ROBUST = "robust"


class Optimizer(str, Enum):
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"
    RADAM = "radam"


class LossFunction(str, Enum):
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    QUANTILE = "quantile"


class ChemicalRepresentation(str, Enum):
    FINGERPRINT = "fingerprint"
    SMILES_SEQUENCE = "smiles_sequence"
    GRAPH = "graph"
    DESCRIPTORS = "descriptors"


class Activation(str, Enum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    GELU = "gelu"
    SELU = "selu"
    ELU = "elu"


class FusionStrategy(str, Enum):
    CONCAT = "concat"
    ATTENTION = "attention"
    CROSS_ATTENTION = "cross_attention"
    GATED = "gated"
    BILINEAR = "bilinear"


DEFAULT_PATHS = {
    "data_dir": "${DATA_DIR:-data}",
    "model_dir": "${MODEL_DIR:-models/saved}",
    "log_dir": "${LOG_DIR:-logs}",
    "results_dir": "${RESULTS_DIR:-results}",
}

LOGGING_CONFIG = {
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "level": "INFO",
    "handlers": [logging.StreamHandler(), logging.FileHandler("app.log")],
}
