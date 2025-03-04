# config/config_utils.py
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import ValidationError

import wandb

from ..config.constants import CURRENT_SCHEMA_VERSION, DEFAULT_PATHS, LOGGING_CONFIG
from ..config.default_config import get_default_config
from ..config.schema import CompleteConfig

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging with default settings."""
    logging.basicConfig(**LOGGING_CONFIG)


def resolve_path_variables(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve environment variables in configuration paths.

    Supports format: ${ENV_VAR} or ${ENV_VAR:-default_value}

    Args:
        config: Configuration dictionary with path references

    Returns:
        Configuration with resolved paths
    """
    import os
    import re

    # Pattern matches ${VAR_NAME} or ${VAR_NAME:-default}
    pattern = re.compile(r"\${([A-Za-z0-9_]+)(?::-([^}]+))?}")

    def _resolve(value):
        if isinstance(value, str):
            # Find all matches in the string
            matches = list(pattern.finditer(value))
            if not matches:
                return value

            # Process each match
            result = value
            for match in matches:
                var_name = match.group(1)
                default = match.group(2)

                # Get value from environment or use default
                env_value = os.environ.get(var_name)
                if env_value is None and default is not None:
                    env_value = default
                elif env_value is None:
                    env_value = ""

                # Replace in the string
                placeholder = match.group(0)  # The entire ${...} expression
                result = result.replace(placeholder, env_value)

            return result
        elif isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_resolve(item) for item in value]
        return value

    return _resolve(config)


def load_config(
    config_path: Union[str, Path], resolve_paths: bool = True
) -> Dict[str, Any]:
    """
    Load configuration from a YAML file with validation.

    Args:
        config_path: Path to the YAML configuration file
        resolve_paths: Whether to resolve path variables

    Returns:
        Dict containing the validated configuration
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Merge with defaults
        merged_config = merge_configs(get_default_config(), config)

        # Resolve path variables if requested
        if resolve_paths:
            merged_config = resolve_path_variables(merged_config)

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


def load_secrets() -> Dict[str, Any]:
    """Load secrets from environment variables or secrets file."""
    secrets = {}

    # Try loading from secrets file
    secrets_path = Path("config/secrets.yaml")
    if secrets_path.exists():
        try:
            with open(secrets_path) as f:
                file_secrets = yaml.safe_load(f) or {}
                secrets.update(file_secrets)
        except Exception as e:
            logger.warning(f"Failed to load secrets file: {e}")

    # Environment variables override file
    # Look for variables with FORMAT: CONFIG_SECTION_KEY
    import re

    pattern = re.compile(r"CONFIG_([A-Z0-9_]+)")

    for env_var, value in os.environ.items():
        match = pattern.match(env_var)
        if match:
            # Convert format: CONFIG_SECTION_KEY to section.key
            path = match.group(1).lower().replace("_", ".")

            # Build nested dict from path
            keys = path.split(".")
            current = secrets
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = value

    return secrets


def load_env_config() -> Dict[str, Any]:
    """Load environment-specific configuration based on ENV variable."""
    env = os.environ.get("APP_ENV", "development")
    env_config_path = Path(f"config/environments/{env}.yaml")

    if env_config_path.exists():
        with open(env_config_path) as f:
            return yaml.safe_load(f) or {}
    else:
        logger.warning(f"No configuration found for environment: {env}")
        return {}


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


def parse_cli_overrides(args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse command-line arguments for configuration overrides."""
    import argparse

    parser = argparse.ArgumentParser(description="Override configuration values")
    parser.add_argument(
        "--config-override",
        action="append",
        help="Override config value (e.g., 'training.batch_size=64')",
    )

    args = parser.parse_args(args)
    overrides = {}

    if args.config_override:
        for override in args.config_override:
            if "=" not in override:
                logger.warning(f"Invalid override format: {override}")
                continue

            path, value = override.split("=", 1)
            # Convert value string to appropriate type
            try:
                value = yaml.safe_load(value)
            except Exception:
                pass  # Keep as string if conversion fails

            # Build nested dict from path
            keys = path.split(".")
            current = overrides
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = value

    return overrides
