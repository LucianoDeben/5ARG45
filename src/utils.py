import logging
import os
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset


def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a YAML configuration file and resolve relative paths.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Optional[Dict[str, Any]]: Configuration dictionary with resolved paths, or None if an error occurs.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_path}: {e}")

    if not isinstance(config, dict):
        raise TypeError(
            f"Invalid configuration format in {config_path}. Expected a dictionary."
        )

    # Resolve relative paths
    config_dir = os.path.dirname(config_path)
    if "data_paths" in config and isinstance(config["data_paths"], dict):
        for key, value in config["data_paths"].items():
            if isinstance(value, str):  # Ensure only string paths are modified
                config["data_paths"][key] = os.path.abspath(
                    os.path.join(config_dir, value)
                )

    return config


def load_sampled_data(
    file_path: str,
    sample_size: Optional[int] = None,
    use_chunks: bool = False,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a dataset from a CSV file with optional sampling and chunked loading.

    Parameters:
    - file_path (str): Path to the CSV file.
    - sample_size (Optional[int]): Number of rows to sample. If None, loads the entire file.
    - use_chunks (bool): Whether to use chunked loading for large files.
    - chunk_size (Optional[int]): Size of chunks when using chunked loading.

    Returns:
    - pd.DataFrame: The loaded dataset, potentially sampled, with reset indices if sampled.
    """
    if sample_size is None:
        # Load entire dataset if no sample size is provided
        return pd.read_csv(file_path)

    if use_chunks:
        # Load dataset in chunks and sample iteratively
        chunks, total_loaded = [], 0

        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            if total_loaded >= sample_size:
                break
            sampled_chunk = chunk.sample(
                min(sample_size - total_loaded, len(chunk)), random_state=42
            )
            chunks.append(sampled_chunk)
            total_loaded += len(sampled_chunk)

        sampled_data = pd.concat(chunks, axis=0, ignore_index=True)
    else:
        # Load entire file up to sample_size and then sample from it
        full_data = pd.read_csv(file_path, nrows=sample_size)
        sampled_data = full_data.sample(n=sample_size, random_state=42).reset_index(
            drop=True
        )

    return sampled_data


def sanity_check(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: Union[str, torch.device],
    max_iters: int = 100,
    lr: float = 1e-3,
    loss_threshold: float = 0.1,
) -> bool:
    """
    Perform a sanity check by training the model for a few iterations and checking if the loss decreases.

    Args:
        model (nn.Module): The PyTorch model to check.
        loader (torch.utils.data.DataLoader): DataLoader for loading a batch of data.
        device (Union[str, torch.device]): The device to run the model on.
        max_iters (int): Number of training iterations for the sanity check.
        lr (float): Learning rate for the optimizer.
        loss_threshold (float): Fraction of initial loss to compare final loss against.

    Returns:
        bool: True if the loss reduces significantly, False otherwise.
    """
    model.train()
    X, y = next(iter(loader))
    X, y = X.to(device), y.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    initial_loss = None
    for i in range(max_iters):
        optimizer.zero_grad()
        outputs = model(X).squeeze()
        loss = criterion(outputs, y)

        if any(param.requires_grad for param in model.parameters()):
            loss.backward()
        else:
            logging.warning("Model parameters are frozen. Skipping backpropagation.")

        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()
            logging.info(f"Initial loss: {initial_loss}")

        logging.info(f"Iteration {i+1}/{max_iters}, Loss: {loss.item()}")

        if loss.item() < loss_threshold * initial_loss:
            logging.info("Sanity check passed.")
            return True

    logging.info("Sanity check failed.")
    return False


def create_dataloader(
    X: Union[pd.DataFrame, pd.Series],
    y: pd.Series,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader from input features and labels.

    Args:
        X (Union[pd.DataFrame, pd.Series]): Feature dataset with samples as rows.
        y (pd.Series): Target variable.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: PyTorch DataLoader with a TensorDataset.

    Raises:
        ValueError: If X or y are empty or have mismatched lengths.
    """
    if X.empty or y.empty:
        raise ValueError("Input features and labels cannot be empty.")

    if len(X) != len(y):
        raise ValueError(
            "Feature matrix X and target variable y must have the same number of samples."
        )

    # Convert pandas DataFrame/Series to PyTorch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
