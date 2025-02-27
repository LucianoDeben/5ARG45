import logging
import uuid
from typing import Any, Dict, Optional

import wandb
import yaml

from config.default_config import get_default_config

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary containing the loaded configuration.
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
    Validate the configuration dictionary.

    Args:
        config: Configuration dictionary to validate.

    Raises:
        ValueError: If validation fails.
    """
    # Required top-level keys
    required_keys = ["data", "model", "training", "chemical", "experiment"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required section: {key}")

    # Data section
    if "gctx_file" not in config["data"]:
        raise ValueError("Missing 'gctx_file' in data section")
    if config["data"].get("nrows") is not None and config["data"]["nrows"] <= 0:
        raise ValueError("'nrows' must be positive")
    if config["data"].get("feature_space") not in [
        None,
        "landmark",
        "best inferred",
        "inferred",
    ] and not isinstance(config["data"].get("feature_space"), list):
        raise ValueError("Invalid 'feature_space' value")

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
    Generate a unique configuration ID and update the config.

    Args:
        config: Configuration dictionary to update.

    Returns:
        Unique configuration ID.
    """
    config_id = str(uuid.uuid4())[:8]  # Shortened UUID
    config["experiment"]["run_name"] = config_id
    return config_id


def init_wandb(config: Dict[str, Any]) -> None:
    """
    Initialize Weights & Biases with the configuration.

    Args:
        config: Configuration dictionary to log.
    """
    if config["experiment"]["track"]:
        wandb.init(
            project=config["experiment"]["project_name"],
            name=config["experiment"]["run_name"],
            config=config,
        )
        logger.info(f"Initialized W&B run: {config['experiment']['run_name']}")
    else:
        logger.info("W&B tracking disabled")


def merge_configs(
    default_config: Dict[str, Any], custom_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge a custom configuration with the default configuration.

    Args:
        default_config: Default configuration dictionary.
        custom_config: Custom configuration dictionary to override defaults.

    Returns:
        Merged configuration dictionary.
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
