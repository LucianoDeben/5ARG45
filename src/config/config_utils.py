# config/config_utils.py
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import ValidationError

import wandb
from config.constants import CURRENT_SCHEMA_VERSION, DEFAULT_PATHS, LOGGING_CONFIG
from config.default_config import get_default_config
from config.schema import CompleteConfig

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging with default settings."""
    logging.basicConfig(**LOGGING_CONFIG)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file with validation.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dict containing the validated configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValidationError: If configuration is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Merge with defaults
        merged_config = merge_configs(get_default_config(), config)

        # Validate against schema
        validated_config = validate_config(merged_config)

        logger.info(f"Successfully loaded and validated config from {config_path}")
        return validated_config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML from {config_path}: {e}")
        raise
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration using pydantic schema.

    Args:
        config: Configuration dictionary to validate

    Returns:
        Validated configuration dictionary

    Raises:
        ValidationError: If configuration is invalid
    """
    try:
        validated = CompleteConfig(**config)
        return validated.dict()
    except ValidationError as e:
        logger.error("Configuration validation failed:")
        for error in e.errors():
            logger.error(f"- {error['loc']}: {error['msg']}")
        raise


def merge_configs(
    default_config: Dict[str, Any],
    custom_config: Dict[str, Any],
    allow_new_keys: bool = False,
) -> Dict[str, Any]:
    """
    Merge default and custom configurations with nested support.

    Args:
        default_config: Default configuration dictionary
        custom_config: Custom configuration to override defaults
        allow_new_keys: Whether to allow new keys not in default config

    Returns:
        Merged configuration dictionary
    """

    def deep_merge(d1: Dict, d2: Dict, path=None) -> Dict:
        path = path or []
        result = d1.copy()

        for key, value in d2.items():
            if not allow_new_keys and key not in d1:
                logger.warning(f"Unknown configuration key: {'.'.join(path + [key])}")
                continue

            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = deep_merge(result[key], value, path + [key])
            else:
                result[key] = value
        return result

    merged = deep_merge(default_config, custom_config)
    return merged


def generate_run_name(config: Dict[str, Any]) -> str:
    """
    Generate a unique run name for the experiment.

    Args:
        config: Configuration dictionary

    Returns:
        Generated run name
    """
    # Extract key components for the run name
    model_type = config["model"]["fusion_strategy"]
    chemical_rep = config["chemical"]["representation"]
    feature_space = config["data"]["feature_space"]

    # Generate unique ID
    unique_id = str(uuid.uuid4())[:8]

    # Construct run name
    run_name = f"{model_type}-{chemical_rep}-{feature_space}-{unique_id}"

    return run_name


def init_wandb(
    config: Dict[str, Any], group: Optional[str] = None, tags: Optional[list] = None
) -> None:
    """
    Initialize Weights & Biases (W&B) for experiment tracking.

    Args:
        config: Configuration dictionary
        group: Optional group name for the run
        tags: Optional tags for the run

    Raises:
        Exception: If W&B initialization fails
    """
    if not config["experiment"]["track"]:
        logger.info("W&B tracking disabled")
        return

    # Generate run name if not provided
    if not config["experiment"]["run_name"]:
        config["experiment"]["run_name"] = generate_run_name(config)

    try:
        # Set up W&B environment
        os.environ["WANDB_SILENT"] = "true"

        # Initialize run
        wandb.init(
            project=config["experiment"]["project_name"],
            name=config["experiment"]["run_name"],
            group=group,
            tags=tags or config["experiment"].get("tags"),
            config=config,
            settings=wandb.Settings(start_method="thread"),
        )

        logger.info(
            f"Initialized W&B run: {config['experiment']['run_name']}"
            + (f" in group: {group}" if group else "")
        )

    except Exception as e:
        logger.error(f"Failed to initialize W&B: {e}")
        raise


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary to save
        save_path: Path where to save the configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved configuration to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise


def get_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Get dictionary of paths for data, models, logs etc.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of paths
    """
    paths = {}
    for key, default_path in DEFAULT_PATHS.items():
        # Check if path is defined in config, otherwise use default
        path = config.get("paths", {}).get(key, default_path)
        paths[key] = Path(path)

        # Create directory if it doesn't exist
        if not paths[key].exists():
            paths[key].mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {paths[key]}")

    return paths


def check_compatibility(config: Dict[str, Any]) -> bool:
    """
    Check if configuration version is compatible with current schema.

    Args:
        config: Configuration dictionary

    Returns:
        True if compatible, False otherwise
    """
    config_version = config["experiment"].get("version", "0.0.0")

    # Parse versions
    curr_major, curr_minor, curr_patch = map(int, CURRENT_SCHEMA_VERSION.split("."))
    conf_major, conf_minor, conf_patch = map(int, config_version.split("."))

    # Major version must match, minor version must be <= current
    if conf_major != curr_major or conf_minor > curr_minor:
        logger.warning(
            f"Configuration version {config_version} may be incompatible with "
            f"current schema version {CURRENT_SCHEMA_VERSION}"
        )
        return False

    return True


def upgrade_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upgrade configuration to be compatible with current schema version.

    Args:
        config: Configuration dictionary to upgrade

    Returns:
        Upgraded configuration dictionary
    """
    if check_compatibility(config):
        return config

    logger.info("Upgrading configuration to current schema version...")

    # Load default config for current version
    current_config = get_default_config()

    # Merge while preserving custom values where possible
    upgraded_config = merge_configs(current_config, config, allow_new_keys=True)

    # Update version
    upgraded_config["experiment"]["version"] = CURRENT_SCHEMA_VERSION

    logger.info(f"Configuration upgraded to version {CURRENT_SCHEMA_VERSION}")
    return upgraded_config
