import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, StratifiedShuffleSplit
from data_sets import LINCSDataset


def train_val_test_split(
    dataset: LINCSDataset,
    source: str = "gene",
    train_size: float = 0.4,
    val_size: float = 0.1,
    test_size: float = 0.5,
    random_state: Optional[int] = None,
    splitter: Optional[object] = None,
    group_column: Optional[str] = None,
    stratify_column: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into train, validation, and test sets using a two-stage splitting approach.

    Args:
        dataset (LINCSDataset): The dataset instance to split.
        source (str): Data source to use ("gene" for gene expression, "tf" for TF activity). Default: "gene".
        train_size (float): Proportion of data for training (0.0 to 1.0). Default: 0.6.
        val_size (float): Proportion of data for validation (0.0 to 1.0). Default: 0.2.
        test_size (float): Proportion of data for testing (0.0 to 1.0). Default: 0.2.
        random_state (int, optional): Seed for reproducibility.
        splitter (object, optional): Custom scikit-learn splitter (e.g., GroupShuffleSplit).
        group_column (str, optional): Column name in row metadata for group-based splitting (e.g., 'patient_id').
        stratify_column (str, optional): Column name in row metadata for stratified splitting.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (X_train, y_train, X_val, y_val, X_test, y_test) as NumPy arrays.

    Raises:
        ValueError: If split sizes don't sum to 1.0, source is invalid, or group_column/stratify_column is missing.
    """
    # Validate split proportions
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(f"train_size ({train_size}) + val_size ({val_size}) + test_size ({test_size}) must sum to 1.0")

    # Validate and set the data source
    if source not in ["gene", "tf"]:
        raise ValueError(f"Invalid source '{source}'. Must be 'gene' or 'tf'.")

    # Fetch data from the dataset
    X_df, y_series = dataset.get_pandas(source=source)
    X = X_df.to_numpy()
    y = y_series.to_numpy()

    # Fetch groups or stratification data from row metadata if specified
    groups = None
    stratify = None
    if group_column:
        try:
            groups = dataset.get_row_metadata(columns=group_column).to_numpy()
        except KeyError as e:
            raise ValueError(f"Group column '{group_column}' not found in row metadata: {str(e)}")
    if stratify_column:
        try:
            stratify = dataset.get_row_metadata(columns=stratify_column).to_numpy()
        except KeyError as e:
            raise ValueError(f"Stratify column '{stratify_column}' not found in row metadata: {str(e)}")

    # Determine the splitter if not provided
    if splitter is None:
        if group_column:
            splitter = GroupShuffleSplit(
                n_splits=1,
                train_size=train_size,
                test_size=val_size + test_size,
                random_state=random_state
            )
        elif stratify_column:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=train_size,
                test_size=val_size + test_size,
                random_state=random_state
            )
        else:
            splitter = ShuffleSplit(
                n_splits=1,
                train_size=train_size,
                test_size=val_size + test_size,
                random_state=random_state
            )

    # Stage 1: Split into train and temp (val + test)
    if isinstance(splitter, GroupShuffleSplit):
        if groups is None:
            raise ValueError("Groups must be provided for GroupShuffleSplit")
        for train_idx, temp_idx in splitter.split(X, y, groups):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_temp = X[temp_idx]
            y_temp = y[temp_idx]
            groups_temp = groups[temp_idx]
    elif isinstance(splitter, StratifiedShuffleSplit):
        if stratify is None:
            raise ValueError("Stratification data must be provided for StratifiedShuffleSplit")
        for train_idx, temp_idx in splitter.split(X, y, stratify):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_temp = X[temp_idx]
            y_temp = y[temp_idx]
            stratify_temp = stratify[temp_idx]
    else:
        for train_idx, temp_idx in splitter.split(X, y):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_temp = X[temp_idx]
            y_temp = y[temp_idx]
            groups_temp = None
            stratify_temp = None

    # Stage 2: Split temp into validation and test
    temp_size = val_size + test_size
    val_prop = val_size / temp_size
    test_prop = test_size / temp_size

    if isinstance(splitter, GroupShuffleSplit):
        temp_splitter = GroupShuffleSplit(
            n_splits=1,
            train_size=val_prop,
            test_size=test_prop,
            random_state=random_state
        )
        for val_idx, test_idx in temp_splitter.split(X_temp, y_temp, groups_temp):
            X_val = X_temp[val_idx]
            y_val = y_temp[val_idx]
            X_test = X_temp[test_idx]
            y_test = y_temp[test_idx]
    elif isinstance(splitter, StratifiedShuffleSplit):
        temp_splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=val_prop,
            test_size=test_prop,
            random_state=random_state
        )
        for val_idx, test_idx in temp_splitter.split(X_temp, y_temp, stratify_temp):
            X_val = X_temp[val_idx]
            y_val = y_temp[val_idx]
            X_test = X_temp[test_idx]
            y_test = y_temp[test_idx]
    else:
        temp_splitter = ShuffleSplit(
            n_splits=1,
            train_size=val_prop,
            test_size=test_prop,
            random_state=random_state
        )
        for val_idx, test_idx in temp_splitter.split(X_temp, y_temp):
            X_val = X_temp[val_idx]
            y_val = y_temp[val_idx]
            X_test = X_temp[test_idx]
            y_test = y_temp[test_idx]

    # Log the split sizes
    logging.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]} samples")

    return X_train, y_train, X_val, y_val, X_test, y_test