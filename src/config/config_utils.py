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

from .constants import CURRENT_SCHEMA_VERSION, DEFAULT_PATHS, LOGGING_CONFIG, REQUIRED_CONFIG_SECTIONS
from .default_config import get_default_config
from .schema import CompleteConfig

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Configure logging with customizable log level."""
    logging_config = LOGGING_CONFIG.copy()
    logging_config["level"] = log_level.upper()
    logging.basicConfig(**logging_config)
    logging.info(f"Logging configured with level: {log_level}")


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
        logger.warning(f"Config file not found: {config_path}. Using default configuration.")
        user_config = {}
    else:
        try:
            with open(config_path) as f:
                user_config = yaml.safe_load(f) or {}
            logger.info(f"Loaded user configuration from {config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML: {e}")
            raise
    
    # Get default configuration
    default_config = get_default_config()
    
    # Load environment-specific config and secrets
    env_config = load_environment_config()
    
    # Track which sections are using defaults
    default_sections = set()
    for section in REQUIRED_CONFIG_SECTIONS:
        if section not in user_config:
            default_sections.add(section)
    
    # Merge configurations with priority: default < env < file < cli
    merged_config = merge_configs(default_config, env_config)
    merged_config = merge_configs(merged_config, user_config)
    
    # Add CLI overrides
    cli_overrides = parse_cli_overrides()
    if cli_overrides:
        merged_config = merge_configs(merged_config, cli_overrides)
        
    if resolve_paths:
        merged_config = resolve_path_variables(merged_config)
    
    # Notify user about sections using defaults
    if default_sections:
        logger.warning(f"Using default configuration for sections: {', '.join(default_sections)}")
        
    try:
        validated_config = validate_config(merged_config)
        return validated_config
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def load_environment_config() -> Dict[str, Any]:
    """Load environment-specific configuration and secrets."""
    config = {}
    
    # Load environment-specific config
    env = os.environ.get("APP_ENV", "development")
    env_config_path = Path(f"config/environments/{env}.yaml")
    if env_config_path.exists():
        with open(env_config_path) as f:
            env_config = yaml.safe_load(f) or {}
            config.update(env_config)
    
    # Load secrets
    secrets_path = Path("config/secrets.yaml")
    if secrets_path.exists():
        try:
            with open(secrets_path) as f:
                secrets = yaml.safe_load(f) or {}
                config.update(secrets)
        except Exception as e:
            logger.warning(f"Failed to load secrets file: {e}")
    
    # Parse environment variables with CONFIG_ prefix
    import re
    pattern = re.compile(r"CONFIG_([A-Z0-9_]+)")
    env_config = {}
    for env_var, value in os.environ.items():
        if match := pattern.match(env_var):
            path = match.group(1).lower().replace("_", ".")
            keys = path.split(".")
            current = env_config
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = value
    
    config.update(env_config)
    return config


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
    molecular_rep = config["molecular"]["representation"]
    feature_space = config["data"]["feature_space"]
    unique_id = str(uuid.uuid4())[:8]
    return f"{model_type}-{molecular_rep}-{feature_space}-{unique_id}"


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


def get_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Get dictionary of paths for data, models, logs, etc."""
    paths = {}
    for key, default_path in DEFAULT_PATHS.items():
        path = config.get("paths", {}).get(key, default_path)
        paths[key] = Path(path)
        if not paths[key].exists():
            paths[key].mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {paths[key]}")
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

#TODO: Simplify this function by using argparse.ArgumentParser
#TODO: Simplify this function for multiple overrides
def parse_cli_overrides(args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse command-line arguments for configuration overrides."""
    import argparse

    parser = argparse.ArgumentParser(description="Override configuration values")
    parser.add_argument(
        "--config-override",
        action="append",
        help="Override config value (e.g., 'training.batch_size=64')",
    )
    parsed_args = parser.parse_args(args)
    overrides = {}
    if parsed_args.config_override:
        for override in parsed_args.config_override:
            if "=" not in override:
                logger.warning(f"Invalid override format: {override}")
                continue
            path, value = override.split("=", 1)
            try:
                value = yaml.safe_load(value)
            except Exception:
                pass  
            keys = path.split(".")
            current = overrides
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = value
    return overrides