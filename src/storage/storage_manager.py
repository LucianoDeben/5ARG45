import json
import os
from datetime import datetime

import torch


class StorageManager:
    """
    Manages storage operations for model checkpoints and configurations.

    Attributes:
        config (dict): Configuration dictionary containing storage settings.
        base_path (str): Base directory path for storing artifacts.
    """

    def __init__(self, config: dict):
        """
        Initializes the StorageManager with the provided configuration.

        Args:
            config (dict): Configuration dictionary with storage settings.
        """
        self.config = config
        self.base_path = config["storage"]["base_path"]
        os.makedirs(self.base_path, exist_ok=True)

    def save_model(self, model: torch.nn.Module, run_id: str) -> str:
        """
        Saves the model's state dictionary to a file.

        Args:
            model (torch.nn.Module): The PyTorch model to save.
            run_id (str): Unique identifier for the run.

        Returns:
            str: Path to the saved model file.
        """
        path = os.path.join(self.base_path, f"model_{run_id}.pt")
        torch.save(model.state_dict(), path)
        return path

    def load_model(self, model: torch.nn.Module, run_id: str) -> torch.nn.Module:
        """
        Loads the model's state dictionary from a file into the provided model.

        Args:
            model (torch.nn.Module): The PyTorch model to load the state into.
            run_id (str): Unique identifier for the run.

        Returns:
            torch.nn.Module: The model with the loaded state.
        """
        path = os.path.join(self.base_path, f"model_{run_id}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found for run_id {run_id}")
        model.load_state_dict(torch.load(path))
        return model

    def save_config(self, config: dict, run_id: str) -> str:
        """
        Saves the configuration dictionary to a JSON file.

        Args:
            config (dict): Configuration dictionary to save.
            run_id (str): Unique identifier for the run.

        Returns:
            str: Path to the saved configuration file.
        """
        path = os.path.join(self.base_path, f"config_{run_id}.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
        return path

    def load_config(self, run_id: str) -> dict:
        """
        Loads the configuration from a JSON file.

        Args:
            run_id (str): Unique identifier for the run.

        Returns:
            dict: The loaded configuration dictionary.
        """
        path = os.path.join(self.base_path, f"config_{run_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No config found for run_id {run_id}")
        with open(path, "r") as f:
            return json.load(f)

    def generate_run_id(self) -> str:
        """
        Generates a unique run ID based on the current timestamp.

        Returns:
            str: A unique run ID (e.g., "20231015_143022").
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
