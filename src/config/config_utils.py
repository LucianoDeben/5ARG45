# src/config/config_utils.py
import logging
import os
import uuid
from os.path import expandvars
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import ValidationError

import wandb

from .constants import CURRENT_SCHEMA_VERSION, DEFAULT_PATHS, LOGGING_CONFIG
from .default_config import get_default_config
from .schema import CompleteConfig

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging with default settings."""
    logging.basicConfig(**LOGGING_CONFIG)


def resolve_path_variables(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve environment variables and dynamic values in configuration paths."""
    import datetime

    def _resolve(value):
        if isinstance(value, str):
            # Handle ${VAR:-default} syntax
            resolved = expandvars(value)

            # Handle dynamic timestamps
            if "${timestamp}" in resolved:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                resolved = resolved.replace("${timestamp}", timestamp)

            return resolved
        elif isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_resolve(item) for item in value]
        return value

    return _resolve(config)


def load_config(
    config_path: Union[str, Path], resolve_paths: bool = True
) -> Dict[str, Any]:
    """Load configuration from a YAML file with validation."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        merged_config = merge_configs(get_default_config(), config)
        if resolve_paths:
            merged_config = resolve_path_variables(merged_config)
        validated_config = validate_config(merged_config)
        logger.info(f"Loaded and validated config from {config_path}")
        return validated_config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML: {e}")
        raise
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def load_secrets() -> Dict[str, Any]:
    """Load secrets from environment variables or secrets file."""
    secrets = {}
    secrets_path = Path("config/secrets.yaml")
    if secrets_path.exists():
        try:
            with open(secrets_path) as f:
                secrets.update(yaml.safe_load(f) or {})
        except Exception as e:
            logger.warning(f"Failed to load secrets file: {e}")

    import re

    pattern = re.compile(r"CONFIG_([A-Z0-9_]+)")
    for env_var, value in os.environ.items():
        if match := pattern.match(env_var):
            path = match.group(1).lower().replace("_", ".")
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
    logger.warning(f"No config for environment: {env}")
    return {}


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration using Pydantic schema."""
    try:
        validated = CompleteConfig(**config)
        return validated.dict()
    except ValidationError as e:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"- {err['loc']}: {err['msg']}" for err in e.errors()
        )
        logger.error(error_msg)
        raise


def merge_configs(
    default_config: Dict[str, Any],
    custom_config: Dict[str, Any],
    allow_new_keys: bool = True,
) -> Dict[str, Any]:
    """Merge default and custom configurations with nested support."""

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

    return deep_merge(default_config, custom_config)


def generate_run_name(config: Dict[str, Any]) -> str:
    """Generate a unique run name for the experiment."""
    model_type = config["model"]["fusion_strategy"]
    chemical_rep = config["chemical"]["representation"]
    feature_space = config["data"]["feature_space"]
    unique_id = str(uuid.uuid4())[:8]
    return f"{model_type}-{chemical_rep}-{feature_space}-{unique_id}"


def init_wandb(
    config: Dict[str, Any],
    group: Optional[str] = None,
    tags: Optional[List] = None,
    wandb_settings: Optional[Dict] = None,
) -> None:
    """Initialize Weights & Biases for experiment tracking."""
    if not config["experiment"]["track"]:
        logger.info("W&B tracking disabled")
        return

    if not config["experiment"]["run_name"]:
        config["experiment"]["run_name"] = generate_run_name(config)

    try:
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(
            project=config["experiment"]["project_name"],
            name=config["experiment"]["run_name"],
            group=group,
            tags=tags or config["experiment"].get("tags"),
            config=config,
            settings=wandb.Settings(**(wandb_settings or {"start_method": "thread"})),
        )
        logger.info(
            f"Initialized W&B run: {config['experiment']['run_name']}"
            + (f" in group: {group}" if group else "")
        )
    except Exception as e:
        logger.error(f"Failed to initialize W&B: {e}")
        raise


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save configuration to a YAML file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved configuration to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise


def resolve_path(path: str, base_dir: Path = None) -> Path:
    """
    Resolve path with environment variables and default values.

    Args:
        path: Path string potentially containing environment variable placeholders
        base_dir: Base directory to use if no absolute path is provided

    Returns:
        Resolved absolute path
    """
    # First, expand environment variables
    resolved_path = os.path.expandvars(path)

    # If the path still contains unresolved variables, use the default part
    if "${" in resolved_path:
        # Extract the default value after the ':-'
        default_path = path.split(":-")[-1].rstrip("}")
        resolved_path = default_path

    # If base_dir is provided and the path is not absolute, join with base_dir
    if base_dir and not os.path.isabs(resolved_path):
        resolved_path = str(base_dir / resolved_path)

    # Convert to absolute path
    return Path(resolved_path).resolve()


def get_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Get dictionary of paths for data, models, logs, etc."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent.absolute()

    paths = {}
    default_paths = {
        "data_dir": "data",
        "model_dir": "models/saved",
        "log_dir": "logs",
        "results_dir": "results",
    }

    for key, default_path in default_paths.items():
        # Get path from config or use default
        raw_path = config.get("paths", {}).get(key, default_path)

        # Resolve the path
        try:
            resolved_path = resolve_path(raw_path, project_root)

            # Create directory
            resolved_path.mkdir(parents=True, exist_ok=True)

            paths[key] = resolved_path
            logger.info(f"Created directory for {key}: {resolved_path}")
        except Exception as e:
            logger.error(f"Failed to create directory for {key}: {e}")
            # Fallback to project root + default path
            fallback_path = project_root / default_path
            fallback_path.mkdir(parents=True, exist_ok=True)
            paths[key] = fallback_path
            logger.warning(f"Used fallback directory for {key}: {fallback_path}")

    return paths


def check_compatibility(config: Dict[str, Any]) -> bool:
    """Check if configuration version is compatible with current schema."""
    config_version = config["experiment"].get("version", "0.0.0")
    curr_major, curr_minor, curr_patch = map(int, CURRENT_SCHEMA_VERSION.split("."))
    conf_major, conf_minor, conf_patch = map(int, config_version.split("."))
    if conf_major != curr_major or conf_minor > curr_minor:
        logger.warning(
            f"Config version {config_version} may be incompatible with {CURRENT_SCHEMA_VERSION}"
        )
        return False
    return True


def upgrade_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade configuration to current schema version."""
    if check_compatibility(config):
        return config
    logger.info("Upgrading configuration to current schema version...")
    upgraded_config = merge_configs(get_default_config(), config, allow_new_keys=True)
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
            try:
                value = yaml.safe_load(value)
            except Exception:
                pass  # Keep as string if conversion fails
            keys = path.split(".")
            current = overrides
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = value
    return overrides
