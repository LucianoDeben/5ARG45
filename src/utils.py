import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, train_test_split
from data_sets import LINCSDataset
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
        file_path (str): Path to the CSV file.
        sample_size (Optional[int]): Number of rows to sample. If None, loads the entire file.
        use_chunks (bool): Whether to use chunked loading for large files.
        chunk_size (Optional[int]): Size of chunks when using chunked loading.

    Returns:
        pd.DataFrame: The loaded dataset, potentially sampled, with reset indices if sampled.

    Note:
        - When using chunked loading, `chunk_size` must be provided.
        - In the non-chunked branch, if `sample_size` is provided, a random sample of `sample_size`
          rows is taken from the entire file.
    """
    if use_chunks:
        if chunk_size is None:
            raise ValueError("chunk_size must be provided when use_chunks is True")

        chunks = []
        num_samples_collected = 0
        # Iterate through the file in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # If a sample size is defined, check if we have already collected enough rows.
            if sample_size is not None and num_samples_collected >= sample_size:
                break

            if sample_size is not None:
                # Determine how many rows to sample from this chunk
                n_to_sample = min(sample_size - num_samples_collected, len(chunk))
                sampled_chunk = chunk.sample(n=n_to_sample, random_state=42)
            else:
                sampled_chunk = chunk

            chunks.append(sampled_chunk)
            num_samples_collected += len(sampled_chunk)

        sampled_data = pd.concat(chunks, axis=0, ignore_index=True)

    else:
        # Load the entire dataset at once
        full_data = pd.read_csv(file_path)
        if sample_size is not None:
            sampled_data = full_data.sample(n=sample_size, random_state=42).reset_index(
                drop=True
            )
        else:
            sampled_data = full_data

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

        if loss.item() < loss_threshold * initial_loss:
            logging.info("Sanity check passed.")
            return True

    logging.info("Sanity check failed.")
    return False


def create_dataset(X: pd.DataFrame, y: pd.Series) -> TensorDataset:
    if X.empty or y.empty:
        raise ValueError("Input features and labels cannot be empty.")
    if len(X) != len(y):
        raise ValueError(
            "Feature matrix X and target variable y must have the same number of samples."
        )
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


def create_dataloader(
    X: Union[pd.DataFrame, pd.Series],
    y: pd.Series,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    dataset = create_dataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def filter_dataset_columns(df, gene_mapping):
    """
    Filter the DataFrame columns so that only gene expression columns present
    in the gene_mapping are retained, along with the last three columns which
    are assumed to be 'viability', 'cell_mfc_name', and 'pert_dose'.

    Args:
        df (pd.DataFrame): The input DataFrame containing gene expression data and extra columns.
        gene_mapping (dict): A dictionary mapping gene symbols to IDs.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if df.shape[1] < 3:
        raise ValueError(
            "DataFrame does not have at least three extra columns to preserve."
        )
    gene_cols = df.columns[:-3]  # All except the last three columns
    extra_cols = df.columns[-3:]  # The last three columns
    filtered_gene_cols = [col for col in gene_cols if col in gene_mapping]
    new_columns = filtered_gene_cols + list(extra_cols)
    filtered_df = df[new_columns].copy()
    return filtered_df

def create_smiles_dict(smiles_df: pd.DataFrame) -> Dict[str, str]:
    """
    Creates and validates a dictionary mapping pert_id to canonical SMILES strings from a DataFrame.

    Args:
        smiles_df (pd.DataFrame): DataFrame containing 'pert_id' and 'canonical_smiles' columns.

    Returns:
        Dict[str, str]: Dictionary mapping pert_id to canonical SMILES strings.
    """
    # Check if needed columns exist otherwise raise error
    if (
        "pert_id" not in smiles_df.columns
        or "canonical_smiles" not in smiles_df.columns
    ):
        raise ValueError(
            "The DataFrame must contain 'pert_id' and 'canonical_smiles' columns."
        )

    # Ensure no leading/trailing spaces
    smiles_df["pert_id"] = smiles_df["pert_id"].str.strip()
    smiles_df["canonical_smiles"] = smiles_df["canonical_smiles"].str.strip()

    # Remove duplicates, keeping the first occurrence
    smiles_df = smiles_df.drop_duplicates(subset="pert_id", keep="first")

    # Check for missing values and handle them
    if smiles_df["canonical_smiles"].isnull().any():
        smiles_df.loc[smiles_df["canonical_smiles"].isnull(), "canonical_smiles"] = (
            "UNKNOWN"
        )

    # Create the mapping dictionary
    smiles_dict = dict(zip(smiles_df["pert_id"], smiles_df["canonical_smiles"]))

    # Validate the dictionary
    if not smiles_dict:
        raise ValueError(
            "The DataFrame must contain non-empty 'pert_id' and 'canonical_smiles' columns."
        )

    for pert_id, smiles in smiles_dict.items():
        if not isinstance(pert_id, str) or not isinstance(smiles, str):
            raise TypeError("The pert_id and canonical_smiles must be strings.")
        if pd.isna(smiles):
            raise ValueError("The canonical_smiles cannot be NaN.")

    return smiles_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def sampled_split(
    dataset, 
    test_size: float = 0.5, 
    random_state: int = 42, 
    stratify: bool = False, 
    stratify_col: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly splits the dataset indices into training and test sets using a specified test_size.
    
    If stratify is True and stratify_col is provided, the split is stratified on that metadata column.
    
    Args:
        dataset: An instance that implements get_row_metadata() returning a pandas DataFrame.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.
        stratify (bool): Whether to stratify the split.
        stratify_col (str, optional): Column name in row metadata to use for stratification.
        
    Returns:
        train_idx (np.ndarray): Array of training indices.
        test_idx (np.ndarray): Array of test indices.
    """
    indices = np.arange(len(dataset))
    row_metadata = dataset.get_row_metadata()
    
    if stratify:
        if stratify_col is None or stratify_col not in row_metadata.columns:
            raise ValueError("Stratification enabled but a valid stratify_col was not provided.")
        stratify_values = row_metadata[stratify_col].values
    else:
        stratify_values = None
        
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values
    )
    logging.info(f"Custom random split: {len(train_idx)} train samples, {len(test_idx)} test samples.")
    return train_idx, test_idx

def stratified_split(
    dataset, 
    stratify_col: str, 
    test_size: float = 0.5, 
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the dataset randomly into train and test sets while preserving the distribution
    of the values in stratify_col.
    
    Args:
        dataset: An object that implements get_row_metadata() returning a pandas DataFrame.
        stratify_col (str): Column name in row metadata to use for stratification (e.g. "cell_mfc_name").
        test_size (float): Fraction of the dataset to use as the test set.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        train_idx (np.ndarray): Array of training indices.
        test_idx (np.ndarray): Array of test indices.
    """
    row_metadata: pd.DataFrame = dataset.get_row_metadata()
    if stratify_col not in row_metadata.columns:
        raise ValueError(f"Stratify column '{stratify_col}' not found in row metadata.")
    
    stratify_values = row_metadata[stratify_col].values
    indices = np.arange(len(dataset))
    
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values
    )
    
    logging.info(f"Stratified split: {len(train_idx)} training samples and {len(test_idx)} test samples.")
    return train_idx, test_idx

def grouped_split(
    dataset, 
    group_col: str, 
    test_size: float = 0.5, 
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the dataset into train and test sets such that entire groups (defined by group_col)
    are assigned exclusively to either train or test.
    
    Args:
        dataset: An object that implements get_row_metadata() returning a pandas DataFrame.
        group_col (str): Column name in row metadata representing groups (e.g. "cell_mfc_name").
        test_size (float): Fraction of the unique groups to assign to the test set.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        train_idx (np.ndarray): Array of indices for training.
        test_idx (np.ndarray): Array of indices for testing.
    """
    row_metadata: pd.DataFrame = dataset.get_row_metadata()
    if group_col not in row_metadata.columns:
        raise ValueError(f"Group column '{group_col}' not found in row metadata.")
    
    # Identify unique groups
    unique_groups = np.unique(row_metadata[group_col].values)
    
    # Split groups into train and test groups
    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=random_state
    )
    
    # Get the indices corresponding to each set of groups.
    indices = np.arange(len(dataset))
    groups = row_metadata[group_col].values
    
    train_idx = indices[np.isin(groups, train_groups)]
    test_idx = indices[np.isin(groups, test_groups)]
    
    logging.info(f"Grouped split: {len(train_idx)} training samples and {len(test_idx)} test samples from {len(unique_groups)} unique groups.")
    return train_idx, test_idx