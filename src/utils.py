import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ShuffleSplit, StratifiedGroupKFold, StratifiedShuffleSplit, train_test_split
from data_sets import LINCSDataset
from metrics import get_regression_metrics
from results import CVResults
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

def run_multiple_iterations_models(X, y, models, n_iterations=20,
                                   train_size=0.6, val_size=0.2, test_size=0.2,
                                   random_state=42, metric_fn=get_regression_metrics):
    """
    Runs multiple random splits and trains each model.
    
    Parameters:
        X, y : array-like (features and targets)
        models : dict of {model_name: model_factory}
                 model_factory is a callable that returns a fresh model instance.
        n_iterations : number of random subsampling iterations.
        train_size, val_size, test_size : fractions that sum to 1.0.
        random_state : seed for reproducibility.
        metric_fn : function to compute regression metrics.
    
    Returns:
        A CVResults object aggregating the mean and std of each metric for each model.
    """
    # Initialize an empty dictionary to collect metric lists per model.
    metrics_collection = {model_name: {} for model_name in models.keys()}
    
    for i in range(n_iterations):
        seed = random_state + i if random_state is not None else None
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
            X, y, train_size, val_size, test_size, random_state=seed
        )
        # For simple ML models, combine train and validation
        X_train_combined = np.concatenate([X_train, X_val], axis=0)
        y_train_combined = np.concatenate([y_train, y_val], axis=0)
        
        for model_name, model_factory in models.items():
            # Instantiate a fresh model instance using the factory callable.
            current_model = model_factory()
            current_model.fit(X_train_combined, y_train_combined)
            y_pred = current_model.predict(X_test)
            metrics = metric_fn(y_test, y_pred)
            
            # Update metrics_collection dynamically based on the keys returned by metric_fn.
            for metric_name, value in metrics.items():
                if metric_name not in metrics_collection[model_name]:
                    metrics_collection[model_name][metric_name] = []
                metrics_collection[model_name][metric_name].append(value)
    
    # Compute mean and std for each metric per model.
    results_summary = {}
    for model_name, metric_lists in metrics_collection.items():
        mean_metrics = {m: np.mean(vals) for m, vals in metric_lists.items()}
        std_metrics = {m: np.std(vals) for m, vals in metric_lists.items()}
        results_summary[model_name] = {"mean": mean_metrics, "std": std_metrics}
    
    return CVResults(results_summary)

def train_deep_model_simple(X_train, y_train, X_val, y_val, X_test, 
                            model_factory, training_params, device="cpu"):
    """
    Trains a PyTorch deep learning model using a simple training loop.
    Converts input arrays into DataLoaders, trains for a fixed number of epochs,
    and returns predictions on X_test.
    
    Args:
        X_train, y_train, X_val, y_val, X_test: NumPy arrays.
        model_factory: Callable that returns a new PyTorch model instance.
        training_params: Dict with keys: epochs, batch_size, learning_rate.
        device: 'cpu' or 'cuda'.
    
    Returns:
        y_pred: Numpy array of predictions on X_test.
    """
    # Create TensorDatasets and DataLoaders for training and validation
    batch_size = training_params.get("batch_size", 32)
    epochs = training_params.get("epochs", 20)
    learning_rate = training_params.get("learning_rate", 1e-3)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(),
                                  torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(),
                                torch.from_numpy(y_val).float())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Instantiate the model
    model = model_factory().to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Simple training loop (no early stopping implemented for simplicity)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Optionally evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # After training, evaluate on test set
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch[0].to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
    y_pred = np.concatenate(predictions, axis=0).flatten()
    return y_pred

# -----------------------------
# Iterative Deep Model Evaluation Function
# -----------------------------
def run_multiple_iterations_deep(X, y, model_factory, n_iterations=20,
                                 train_size=0.6, val_size=0.2, test_size=0.2,
                                 random_state=42, metric_fn=get_regression_metrics,
                                 training_params=None, device="cpu"):
    """
    Runs multiple random splits and trains a PyTorch deep learning model.
    
    Parameters:
        X, y : array-like (features and targets)
        model_factory : Callable that returns a new PyTorch model instance.
        n_iterations : number of iterations.
        train_size, val_size, test_size : fractions summing to 1.0.
        random_state : seed for reproducibility.
        metric_fn : function to compute regression metrics.
        training_params : dict with training parameters (epochs, batch_size, learning_rate).
        device : 'cpu' or 'cuda'.
    
    Returns:
        A CVResults object aggregating the metrics.
    """
    if training_params is None:
        training_params = {"epochs": 20, "batch_size": 32, "learning_rate": 1e-3}
    
    metrics_collection = {"DeepModel": {}}
    
    for i in range(n_iterations):
        seed = random_state + i if random_state is not None else None
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
            X, y, train_size, val_size, test_size, random_state=seed
        )
        # For deep models, we use train and validation separately.
        y_pred = train_deep_model_simple(X_train, y_train, X_val, y_val, X_test,
                                         model_factory, training_params, device=device)
        metrics = metric_fn(y_test, y_pred)
        for metric_name, value in metrics.items():
            if metric_name not in metrics_collection["DeepModel"]:
                metrics_collection["DeepModel"][metric_name] = []
            metrics_collection["DeepModel"][metric_name].append(value)
    
    # Compute mean and std for each metric.
    results_summary = {}
    for model_name, metric_lists in metrics_collection.items():
        mean_metrics = {m: np.mean(vals) for m, vals in metric_lists.items()}
        std_metrics = {m: np.std(vals) for m, vals in metric_lists.items()}
        results_summary[model_name] = {"mean": mean_metrics, "std": std_metrics}
    
    return CVResults(results_summary)

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