# src/config/constants.py
import logging
from enum import Enum
from typing import Any, Dict

logger = logging.getLogger(__name__)


def check_config_types(config, schema):
    for key, expected_type in schema.items():
        if key in config:
            value = config[key]
            if isinstance(expected_type, dict):
                assert isinstance(value, dict), f"{key} must be a dict"
                check_config_types(value, expected_type)
            elif isinstance(expected_type, tuple):
                assert any(
                    isinstance(value, t) for t in expected_type
                ), f"{key} must be one of {expected_type}"
            else:
                assert isinstance(
                    value, expected_type
                ), f"{key} must be {expected_type}"


# Schema version and required sections
CURRENT_SCHEMA_VERSION = "1.0.0"
REQUIRED_CONFIG_SECTIONS = {
    "paths",
    "data",
    "training",
    "model",
    "experiment",
}


# Enums for valid options
class FeatureSpace(str, Enum):
    LANDMARK = "landmark"
    BEST_INFERRED = "best inferred"
    INFERRED = "inferred"
    ALL = "all"


class ImputationStrategy(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    KNN = "knn"


class TransformType(str, Enum):
    FINGERPRINT = "fingerprint"
    DESCRIPTORS = "descriptors"
    GRAPH = "graph"


class Normalization(str, Enum):
    ZSCORE = "zscore"
    MINMAX = "minmax"
    ROBUST = "robust"


class Optimizer(str, Enum):
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"
    RADAM = "radam"


class MolecularRepresentation(str, Enum):
    FINGERPRINT = "fingerprint"
    DESCRIPTORS = "descriptors"
    GRAPH = "graph"


class LossFunction(str, Enum):
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    QUANTILE = "quantile"


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


# Default paths
DEFAULT_PATHS = {
    "data_dir": "data",
    "model_dir": "models/saved",
    "log_dir": "logs",
    "results_dir": "results",
    "checkpoint_dir": "models/saved/checkpoints",
}

# Configuration schema for type validation
CONFIG_SCHEMA = {
    "paths": {
        "data_dir": str,
        "model_dir": str,
        "log_dir": str,
        "results_dir": str,
        "checkpoint_dir": str,
    },
    "data": {
        "gctx_file": str,
        "geneinfo_file": str,
        "siginfo_file": str,
        "curves_post_qc": str,
        "per_cpd_post_qc": str,
        "per_experiment": str,
        "per_compound": str,
        "per_cell_line": str,
        "output_path": str,
        "matching_strategy": str,
        "max_workers": (int, type(None)),
        "chunk_size": int,
        "feature_space": str,
        "nrows": int,
        "normalize": str,
        "random_seed": int,
        "cache_data": bool,
        "use_multiprocessing": bool,
        "num_workers": int,
        "batch_size": int,
        "imputation_strategy": str,
        "handle_outliers": bool,
        "outlier_threshold": float,
    },
    "training": {
        "batch_size": int,
        "epochs": int,
        "num_runs": int,
        "learning_rate": float,
        "optimizer": str,
        "loss": str,
        "test_size": float,
        "val_size": float,
        "random_state": int,
        "group_by": (str, type(None)),
        "stratify_by": (str, type(None)),
        "lr_scheduler": {
            "type": str,
            "warmup_epochs": int,
            "min_lr": float,
            "step_size": int,
            "gamma": float,
        },
        "early_stopping": bool,
        "patience": int,
        "min_delta": float,
        "clip_grad_norm": bool,
        "max_grad_norm": float,
        "use_amp": bool,
        "weight_decay": float,
        "label_smoothing": float,
    },
    "molecular": {
        "representation": str,
        "fingerprint_size": int,
        "radius": int,
        "use_chirality": bool,
        "use_features": bool,
        "sanitize": bool,
    },
    "model": {
        "transcriptomics_input_dim": int,
        "transcriptomics_hidden_dims": list,
        "transcriptomics_output_dim": int,
        "molecular_input_dim": int,
        "molecular_hidden_dims": list,
        "molecular_output_dim": int,
        "fusion_output_dim": int,
        "predictor_hidden_dims": list,
        "transcriptomics_encoder_type": str,
        "molecular_encoder_type": str,
        "fusion_type": str,
        "fusion_strategy": str,
        "activation": str,
        "dropout": float,
        "normalize": bool,
        "use_batch_norm": bool,
        "predictor_type": str,
    },
    "evaluation": {
        "metrics": list,
        "loss": str,
        "output_dir": str,
        "visualization": {
            "dpi": int,
            "figsize": list,
        },
    },
    "inference": {
        "device": str,
        "max_ensemble": int,
        "output_path": str,
        "export_formats": list,
    },
    "deployment": {
        "quantization": list,
    },
    "experiment": {
        "project_name": str,
        "run_name": str,
        "track": bool,
        "tags": list,
        "version": str,
        "save_checkpoints": bool,
        "checkpoint_freq": int,
        "keep_n_checkpoints": int,
        "log_level": str,
    },
    "interpretation": {
        "enabled": bool,
        "attribution_methods": list,
        "n_samples": int,
        "top_k": int,
        "visualize_features": bool,
        "feature_naming": {
            "transcriptomics": str,
            "molecular": (str, type(None)),
        },
        "save_attributions": bool,
        "generate_figures": bool,
    },
}

LOGGING_CONFIG = {
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "level": "INFO",
    "handlers": [logging.StreamHandler(), logging.FileHandler("app.log")],
}
