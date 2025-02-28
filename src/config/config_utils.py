# config/config_utils.py
import logging
import uuid
from typing import Any, Dict, Optional

import yaml

import wandb
from config.default_config import get_default_config

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict containing the loaded configuration.

    Raises:
        FileNotFoundError: If the config file is not found.
        yaml.YAMLError: If the YAML file is invalid.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration dictionary for required sections and parameters.

    Args:
        config: Configuration dictionary to validate.

    Raises:
        ValueError: If required sections or parameters are missing or invalid.
    """
    required_keys = ["data", "model", "training", "chemical", "experiment"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required section: {key}")

    # Data section
    if "gctx_file" not in config["data"]:
        raise ValueError("Missing 'gctx_file' in data section")
    if config["data"].get("nrows") is not None and config["data"]["nrows"] <= 0:
        raise ValueError("'nrows' must be positive")
    fs = config["data"].get("feature_space")
    allowed_fs = {"landmark", "best inferred", "inferred", None}
    if isinstance(fs, str) and fs not in allowed_fs:
        raise ValueError(f"Invalid 'feature_space': {fs}. Allowed: {allowed_fs}")
    elif isinstance(fs, list) and not all(x in allowed_fs - {None} for x in fs):
        raise ValueError(
            f"Invalid 'feature_space' list: {fs}. Allowed: {allowed_fs - {None}}"
        )
    if config["data"].get("normalize") not in [None, "zscore"]:
        raise ValueError("Invalid 'normalize' value; must be None or 'zscore'")

    # Model section
    model_required = ["transcriptomics_input_dim", "chemical_input_dim"]
    for key in model_required:
        if key not in config["model"]:
            raise ValueError(f"Missing required model parameter: {key}")
    if config["model"]["transcriptomics_input_dim"] <= 0:
        raise ValueError("'transcriptomics_input_dim' must be positive")
    if config["model"]["chemical_input_dim"] <= 0:
        raise ValueError("'chemical_input_dim' must be positive")

    # Training section
    if config["training"]["batch_size"] <= 0:
        raise ValueError("'batch_size' must be positive")
    if config["training"]["epochs"] <= 0:
        raise ValueError("'epochs' must be positive")
    if config["training"]["learning_rate"] <= 0:
        raise ValueError("'learning_rate' must be positive")
    if config["training"]["test_size"] <= 0 or config["training"]["test_size"] >= 1:
        raise ValueError("'test_size' must be between 0 and 1")
    if config["training"]["val_size"] <= 0 or config["training"]["val_size"] >= 1:
        raise ValueError("'val_size' must be between 0 and 1")

    # Chemical section
    if config["chemical"]["representation"] not in ["fingerprint", "smiles_sequence"]:
        raise ValueError("Invalid 'chemical_representation'")
    if config["chemical"]["fingerprint_size"] <= 0:
        raise ValueError("'fingerprint_size' must be positive")

    logger.info("Configuration validated successfully")


def generate_config_id(config: Dict[str, Any]) -> str:
    """
    Generate a unique configuration ID and update the config with a run name.

    Args:
        config: Configuration dictionary.

    Returns:
        str: Unique configuration ID.
    """
    config_id = str(uuid.uuid4())[:8]
    config["experiment"]["run_name"] = config_id
    return config_id


def init_wandb(config: Dict[str, Any]) -> None:
    """
    Initialize Weights & Biases (W&B) for experiment tracking if enabled.

    Args:
        config: Configuration dictionary containing experiment settings.

    Raises:
        Exception: If W&B initialization fails.
    """
    if config["experiment"]["track"]:
        try:
            wandb.init(
                project=config["experiment"]["project_name"],
                name=config["experiment"]["run_name"],
                config=config,
            )
            logger.info(f"Initialized W&B run: {config['experiment']['run_name']}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            raise
    else:
        logger.info("W&B tracking disabled")


def merge_configs(
    default_config: Dict[str, Any], custom_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge default and custom configurations, preserving nested structures.

    Args:
        default_config: Default configuration dictionary.
        custom_config: Custom configuration dictionary to override defaults.

    Returns:
        Dict: Merged configuration dictionary.
    """

    def deep_merge(d1: Dict, d2: Dict) -> Dict:
        result = d1.copy()
        for key, value in d2.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    merged = deep_merge(default_config, custom_config)
    logger.debug("Merged default and custom configurations")
    return merged
