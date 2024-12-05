import os

import yaml


def load_config(config_path: str) -> dict:
    """
    Load the configuration file and resolve paths.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary with resolved paths.
    """
    # Load the configuration file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Resolve paths based on the location of the configuration file
    config_dir = os.path.dirname(config_path)
    for key, value in config["data_paths"].items():
        config["data_paths"][key] = os.path.join(config_dir, value)

    return config
