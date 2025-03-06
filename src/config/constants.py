# src/config/constants.py
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# Schema version and required sections
CURRENT_SCHEMA_VERSION = "1.0.0"
REQUIRED_CONFIG_SECTIONS = {
    "paths",
    "data",
    "training",
    "molecular",
    "model",
    "evaluation",
    "inference",
    "deployment",
    "experiment",
    "interpretation",
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
    NONE = "none"

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

class RegressionMetrics(str, Enum):
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2 = "r2"
    PEARSON = "pearson"

class Device(str, Enum):
    CPU = "cpu"
    GPU = "cuda"
    TPU = "tpu"

class FusionStrategy(str, Enum):
    CONCAT = "concat"
    ATTENTION = "attention"
    CROSS_ATTENTION = "cross_attention"
    GATED = "gated"
    BILINEAR = "bilinear"
    
class LRScheduler(str, Enum):
    STEP = "step"
    MULTI_STEP = "multi_step"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    PLATEAU = "plateau"

# Default paths
DEFAULT_PATHS = {
    "data_dir": "data",
    "model_dir": "models/saved",
    "log_dir": "logs",
    "results_dir": "results",
    "checkpoint_dir": "models/saved/checkpoints",
}

# Logging configuration
LOGGING_CONFIG = {
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "level": "INFO",
    "handlers": [logging.StreamHandler(), logging.FileHandler("app.log")],
}