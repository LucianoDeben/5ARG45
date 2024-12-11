import logging
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


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


def split_data(
    df,
    config,
    target_name="target_name",
    random_state=42,
):
    """
    Split the DataFrame into train, validation, and test sets, and then into features and labels.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        train_ratio (float): The ratio of the training set.
        val_ratio (float): The ratio of the validation set.
        test_ratio (float): The ratio of the test set.
        target_name (str): The name of the target column.
        random_state (int): The random seed for reproducibility.

    Returns:
        tuple: Splits of the dataset into features and labels for train, validation, and test sets.
    """
    # Split into train and temp (validation + test)
    train_df, temp_df = train_test_split(
        df, test_size=1 - config["preprocess"]["train_ratio"], random_state=random_state
    )

    # Split temp into validation and test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=config["preprocess"]["test_ratio"]
        / (config["preprocess"]["test_ratio"] + config["preprocess"]["val_ratio"]),
        random_state=random_state,
    )

    # Split the datasets into features and labels (X, y)
    X_train, y_train = train_df.drop(target_name, axis=1), train_df[target_name]
    X_val, y_val = val_df.drop(target_name, axis=1), val_df[target_name]
    X_test, y_test = test_df.drop(target_name, axis=1), test_df[target_name]

    return X_train, y_train, X_val, y_val, X_test, y_test
