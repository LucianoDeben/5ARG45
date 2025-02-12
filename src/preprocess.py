import argparse
import logging
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import decoupler as dc
import pandas as pd
from evaluation import evaluate_model
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from training import train_model
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np

logger = logging.getLogger(__name__)
sys.path.append("src")

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler

from utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

import pandas as pd
from sklearn.decomposition import PCA


def perform_pca(X, explained_variance_threshold=0.99):
    """
    Perform PCA on the dataset and return transformed data, PCA object, and features.

    Args:
        X (pd.DataFrame): The dataset containing features.
        explained_variance_threshold (float): Threshold for cumulative explained variance (default: 0.99).

    Returns:
        X_transformed (pd.DataFrame): PCA-transformed dataset.
        pca (PCA): Fitted PCA object.
    """

    # Initialize and fit PCA
    pca = PCA(n_components=explained_variance_threshold)
    X_pca = pca.fit_transform(X)

    # Convert to DataFrame and add the target column
    X_transformed = pd.DataFrame(
        X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
    )

    return (X_transformed, pca)


def _select_genes(geneinfo: pd.DataFrame, feature_space: str) -> pd.DataFrame:
    """
    Filter the geneinfo DataFrame based on the requested feature space.
    """
    if feature_space == "landmark":
        logging.debug("Using landmark genes.")
        selected_genes = geneinfo[geneinfo.feature_space == "landmark"]
    elif feature_space == "best inferred":
        logging.debug("Using best inferred genes (including landmark genes).")
        selected_genes = geneinfo[
            geneinfo.feature_space.isin(["landmark", "best inferred"])
        ]
    elif feature_space == "all":
        logging.debug("Using all genes.")
        selected_genes = geneinfo
    else:
        raise ValueError("Invalid feature space selected.")

    logging.debug(
        f"Selected {selected_genes.shape[0]} genes for feature space '{feature_space}'."
    )
    return selected_genes


def _merge_data(
    X_df: pd.DataFrame, y_df: pd.DataFrame, merge_columns: list = None
) -> pd.DataFrame:
    """
    Merge the gene expression data with selected columns from the metadata.
    """
    if merge_columns is None:
        merge_columns = ["viability", "cell_mfc_name", "pert_dose"]
    try:
        merged_df = pd.concat([X_df, y_df[merge_columns]], axis=1)
        logging.debug("Merging of data complete.")
        return merged_df
    except KeyError as e:
        logging.error(f"Missing expected merge columns in y data: {e}")
        raise


def _write_df_in_chunks(df: pd.DataFrame, output_file: str, chunk_size: int):
    """
    Write a DataFrame to CSV in chunks to reduce memory pressure.
    """
    try:
        with open(output_file, mode="w", newline="") as f:
            for start in range(0, df.shape[0], chunk_size):
                chunk = df.iloc[start : start + chunk_size]
                write_header = start == 0  # Only write header for the first chunk
                chunk.to_csv(f, mode="a", header=write_header, index=False)
        logging.debug(f"Data written to {output_file} in chunks of {chunk_size} rows.")
    except Exception as e:
        logging.error(f"Error writing CSV in chunks: {e}")
        raise


def preprocess_gene_data(
    config: dict,
    standardize: bool = True,
    feature_space: str = "all",
    chunk_size: int = 2500,
):
    """
    Preprocess gene expression data in a memory-efficient manner.

    Steps:
    1. Load RNA data (using np.memmap), gene information, and metadata.
    2. Filter genes based on the selected feature space.
    3. Select corresponding columns from the RNA data.
    4. Optionally standardize the gene expression data.
    5. Merge the gene expression data with metadata.
    6. Write the final DataFrame to CSV in chunks.

    Args:
        config (dict): Configuration dictionary with keys:
            - data_paths: Contains paths for "rna_file", "geneinfo_file", "y_file",
              and output file paths for various feature spaces.
        standardize (bool): Whether to standardize the features.
        feature_space (str): Which genes to use ("all", "landmark", "best inferred").
        chunk_size (int): Number of rows per chunk when writing CSV.

    Returns:
        pd.DataFrame: The final preprocessed DataFrame.
    """
    try:
        logging.debug("Starting preprocessing of gene data.")

        # 1. Load datasets
        # Assuming shape (31567, number_of_genes). Adjust if needed.
        X_rna = np.memmap(config["data_paths"]["rna_file"], dtype=np.float32, mode="r")
        X_rna = X_rna.reshape(31567, -1)  # Assign the reshaped view back to X_rna
        logging.debug(f"Loaded RNA data with shape: {X_rna.shape}")

        geneinfo = pd.read_csv(config["data_paths"]["geneinfo_file"], sep="\t")
        logging.debug(f"Loaded gene info with shape: {geneinfo.shape}")

        y_df = pd.read_csv(config["data_paths"]["y_file"], delimiter="\t")
        logging.debug(f"Loaded y data with shape: {y_df.shape}")

        # 2. Select genes based on the requested feature space
        selected_genes = _select_genes(geneinfo, feature_space)
        selected_gene_indices = selected_genes.index

        if X_rna.shape[1] < len(selected_gene_indices):
            raise ValueError(
                "Mismatch: The number of columns in the RNA data is less than the number of selected genes."
            )

        # 3. Select corresponding gene expression data
        X_selected = X_rna[:, selected_gene_indices]
        X_selected_df = pd.DataFrame(
            X_selected, index=y_df.index, columns=selected_genes.gene_symbol
        )
        logging.debug(f"Gene expression DataFrame shape: {X_selected_df.shape}")

        # 4. Standardize data if required
        if standardize:
            logging.debug("Standardizing gene expression data.")
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(X_selected_df)
            X_selected_df = pd.DataFrame(
                scaled_values, columns=X_selected_df.columns, index=X_selected_df.index
            )

        # 5. Merge gene expression data with metadata
        final_df = _merge_data(X_selected_df, y_df)
        # Drop any rows with missing data
        final_df.dropna(inplace=True)
        logging.debug(
            f"Final DataFrame shape after merging and dropping NA: {final_df.shape}"
        )

        # 6. Determine output file based on feature space
        if feature_space == "landmark":
            output_file = config["data_paths"]["preprocessed_landmark_file"]
        elif feature_space == "best inferred":
            output_file = config["data_paths"]["preprocessed_best_inferred_file"]
        else:
            output_file = config["data_paths"]["preprocessed_gene_file"]

        # Write final DataFrame in chunks
        _write_df_in_chunks(final_df, output_file, chunk_size)

        logging.info("Preprocessing and chunked saving completed successfully.")
        return final_df

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise

def _validate_ratios(config: Dict) -> None:
    """Validate train/val/test split ratios."""
    ratios = config.get("preprocess", {})
    required_keys = {"train_ratio", "val_ratio", "test_ratio"}

    if not required_keys.issubset(ratios.keys()):
        missing = required_keys - set(ratios.keys())
        raise ValueError(f"Missing required keys in config: {missing}")

    train_ratio = ratios["train_ratio"]
    val_ratio = ratios["val_ratio"]
    test_ratio = ratios["test_ratio"]

    if any(r < 0 or r > 1 for r in [train_ratio, val_ratio, test_ratio]):
        raise ValueError("Train/val/test ratios must be between 0 and 1.")

    total = train_ratio + val_ratio + test_ratio
    if not (0.999 <= total <= 1.001):  # Allow for float precision tolerance
        raise ValueError(f"Ratios must sum to 1.0 (current sum: {total:.4f})")


def _get_excluded_columns(
    df: pd.DataFrame,
    target_name: str,
    stratify_by: Optional[str],
    keep_columns: List[str],
) -> List[str]:
    """Determine columns to exclude from features"""
    exclude = [target_name]

    # Handle stratification column
    if stratify_by:
        if stratify_by in df.columns and stratify_by not in keep_columns:
            exclude.append(stratify_by)
    else:
        # Always exclude cell_mfc_name if not explicitly kept
        if "cell_mfc_name" in df.columns and "cell_mfc_name" not in keep_columns:
            exclude.append("cell_mfc_name")

    return exclude


def _stratified_split(
    df: pd.DataFrame, stratify_col: str, config: Dict, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data using group stratification"""
    groups = df[stratify_col]
    if groups.nunique() < 2:
        raise ValueError(
            f"Need at least 2 groups in '{stratify_col}' for stratification"
        )

    # First split: train vs temp
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=1 - config["preprocess"]["train_ratio"],
        random_state=random_state,
    )
    train_idx, temp_idx = next(gss.split(df, groups=groups))
    train_df, temp_df = df.iloc[train_idx], df.iloc[temp_idx]

    # Second split: val vs test
    test_size = config["preprocess"]["test_ratio"] / (
        1 - config["preprocess"]["train_ratio"]
    )
    gss_temp = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    val_idx, test_idx = next(gss_temp.split(temp_df, groups=temp_df[stratify_col]))

    return train_df, temp_df.iloc[val_idx], temp_df.iloc[test_idx]


def _random_split(
    df: pd.DataFrame, config: Dict, target_name: str, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data randomly without stratification (regression-specific)."""

    # First split: train vs temp
    train_df, temp_df = train_test_split(
        df,
        train_size=config["preprocess"]["train_ratio"],
        random_state=random_state,
    )

    # Second split: val vs test
    val_test_ratio = config["preprocess"]["val_ratio"] / (
        1 - config["preprocess"]["train_ratio"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_test_ratio,
        random_state=random_state,
    )

    return train_df, val_df, test_df


def _prepare_features(df: pd.DataFrame, exclude_cols: List[str]) -> pd.DataFrame:
    """Prepare feature DataFrame by dropping excluded columns"""
    return df.drop(columns=exclude_cols, errors="ignore").copy()
    
def k_fold_cross_validation(
    model_class,           
    train_dataset,         
    k_folds: int = 5,
    batch_size: int = 32,
    epochs: int = 10,
    criterion_fn=None,     
    optimizer_fn=None,     
    scheduler_fn=None,     
    device: str = "cuda",
    use_mixed_precision: bool = True,
):
    """
    Perform K-Fold cross-validation on the training dataset.
    
    Returns:
        A dictionary with metrics for each fold.
    """
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics = {}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(train_dataset)))):
        print(f"--- Fold {fold+1}/{k_folds} ---")
        
        # Create subsets for the current fold.
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Create a new instance of your model for this fold.
        model = model_class()
        
        optimizer = optimizer_fn(model.parameters())
        scheduler = scheduler_fn(optimizer) if scheduler_fn is not None else None
        
        # Train the model on the current fold.
        train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )
        
        # Evaluate the model on the validation fold.
        metrics = evaluate_model(model, val_loader, criterion_fn, device=device, calculate_metrics=True)
        fold_metrics[fold] = metrics
        
        print(f"Fold {fold+1} Metrics: {metrics}\n")
        
    # Optionally, average the metrics over the folds.
    avg_metrics = {key: np.mean([fold_metrics[f][key] for f in fold_metrics]) for key in fold_metrics[0]}
    print("Average Metrics Across Folds:", avg_metrics)
    
    return fold_metrics

def filter_dataset_and_network(dataset: pd.DataFrame, network: pd.DataFrame) -> tuple:
    """
    Filters both the dataset and the regulatory network to include only genes
    (targets) that are present in both.

    Args:
        dataset (pd.DataFrame): Gene expression dataset (columns = genes, rows = samples).
        network (pd.DataFrame): Regulatory network with 'source', 'target', and 'weight' columns.

    Returns:
        tuple: Filtered dataset and filtered network as (filtered_dataset, filtered_network).
    """
    # Validate network columns
    required_columns = {"source", "target", "weight"}
    if not required_columns.issubset(network.columns):
        raise ValueError(
            f"The network DataFrame must contain the columns: {required_columns}"
        )

    # Identify intersecting genes
    dataset_genes = set(dataset.columns)
    network_genes = set(network["target"].unique())
    intersecting_genes = dataset_genes.intersection(network_genes)

    if not intersecting_genes:
        raise ValueError(
            "No overlapping genes between the dataset and the regulatory network."
        )

    # Filter dataset to include only intersecting genes
    filtered_dataset = dataset[list(intersecting_genes)]

    # Filter network to include only interactions involving intersecting genes
    filtered_network = network[network["target"].isin(intersecting_genes)]

    logging.info(
        f"Filtered dataset shape: {filtered_dataset.shape}, "
        f"Filtered network size: {len(filtered_network)} interactions for {len(intersecting_genes)} genes."
    )

    return filtered_dataset, filtered_network


def create_gene_tf_matrix(net: pd.DataFrame, genes: list) -> torch.Tensor:
    """
    Creates a PyTorch tensor representing the gene-TF regulatory matrix.

    Args:
        net (pd.DataFrame): Filtered regulatory network with 'source', 'target', and 'weight' columns.
        genes (list): List of genes to include as rows in the matrix.

    Returns:
        torch.Tensor: Gene-TF matrix of shape (num_genes, num_tfs), where:
            - `1` indicates an activating interaction.
            - `-1` indicates an inhibiting interaction.
            - `0` indicates no interaction.
    """
    # Validate input
    required_columns = {"source", "target", "weight"}
    if not required_columns.issubset(net.columns):
        raise ValueError(
            f"The network DataFrame must contain the columns: {required_columns}"
        )

    # Extract unique TFs and initialize the matrix
    unique_tfs = sorted(net["source"].unique())
    gene_to_tf_df = pd.DataFrame(0, index=genes, columns=unique_tfs, dtype=float)

    # Populate the matrix with interaction weights
    for _, row in net.iterrows():
        gene = row["target"]
        tf = row["source"]
        weight = row["weight"]
        if gene in genes and tf in unique_tfs:
            gene_to_tf_df.at[gene, tf] = weight

    # Convert the DataFrame to a PyTorch tensor
    gene_tf_matrix = torch.tensor(gene_to_tf_df.values, dtype=torch.float32)
    logging.info(
        f"Created gene-TF matrix with shape {gene_tf_matrix.shape} "
        f"(num_genes={len(genes)}, num_tfs={len(unique_tfs)})."
    )
    return gene_tf_matrix

def split_data(
    df: pd.DataFrame,
    config: Dict,
    target_name: str = "viability",
    stratify_by: Optional[str] = None,
    keep_columns: Optional[List[str]] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split DataFrame into stratified train/val/test sets with proper column handling.

    Args:
        df: Input DataFrame containing features and target
        config: Dictionary with 'train_ratio', 'val_ratio', 'test_ratio'
        target_name: Name of the target column
        stratify_by: Column name to stratify by (None for random splitting)
        keep_columns: Additional columns to retain in features
        random_state: Seed for reproducibility

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Validate configuration and inputs
    _validate_ratios(config)
    keep_columns = keep_columns or []

    # Handle column exclusions
    exclude_cols = _get_excluded_columns(df, target_name, stratify_by, keep_columns)

    # Split dataset
    if stratify_by:
        train_df, val_df, test_df = _stratified_split(
            df, stratify_by, config, random_state
        )
    else:
        train_df, val_df, test_df = _random_split(df, config, target_name, random_state)

    # Prepare features and targets
    X_train, X_val, X_test = [
        _prepare_features(df, exclude_cols) for df in [train_df, val_df, test_df]
    ]
    y_train, y_val, y_test = [df[target_name] for df in [train_df, val_df, test_df]]

    _log_split_details(X_train, X_val, X_test)
    return X_train, y_train, X_val, y_val, X_test, y_test

def split_data_flexible(
    df: pd.DataFrame,
    config: Dict,
    target_name: str = "viability",
    stratify_by: Optional[str] = None,
    keep_columns: Optional[List[str]] = None,
    random_state: int = 42,
    return_val: bool = True 
) -> Tuple:
    """
    Split the DataFrame either into train/val/test sets or into train/test sets,
    depending on the return_val flag. This function reuses the same helper functions
    for column exclusion and feature preparation.
    
    Args:
        df: Input DataFrame.
        config: Dictionary with keys "train_ratio", and if return_val is True, also "val_ratio" and "test_ratio".
        target_name: Name of the target column.
        stratify_by: Column to stratify by.
        keep_columns: Columns to retain in features.
        random_state: Seed for reproducibility.
        return_val: If True, return (X_train, y_train, X_val, y_val, X_test, y_test);
                    if False, return (X_train, y_train, X_test, y_test).
    
    Returns:
        Tuple with the appropriate splits.
    """
    keep_columns = keep_columns or []
    exclude_cols = _get_excluded_columns(df, target_name, stratify_by, keep_columns)
    
    if return_val:
        # Use your existing fixed-split logic.
        # (Assumes that _validate_ratios and _random_split/_stratified_split are defined.)
        _validate_ratios(config)
        if stratify_by:
            train_df, val_df, test_df = _stratified_split(df, stratify_by, config, random_state)
        else:
            train_df, val_df, test_df = _random_split(df, config, target_name, random_state)
        
        X_train = _prepare_features(train_df, exclude_cols)
        X_val = _prepare_features(val_df, exclude_cols)
        X_test = _prepare_features(test_df, exclude_cols)
        y_train = train_df[target_name]
        y_val = val_df[target_name]
        y_test = test_df[target_name]
        _log_split_details(X_train, X_val, X_test)
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        # Return only train/test splits.
        train_ratio = config["preprocess"]["train_ratio"]
        if stratify_by:
            gss = GroupShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=random_state)
            groups = df[stratify_by]
            train_idx, test_idx = next(gss.split(df, groups=groups))
            train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        else:
            train_df, test_df = train_test_split(df, train_size=train_ratio, random_state=random_state, shuffle=True)
        
        X_train = _prepare_features(train_df, exclude_cols)
        X_test = _prepare_features(test_df, exclude_cols)
        y_train = train_df[target_name]
        y_test = test_df[target_name]
        _log_split_details(X_train, X_test, names=["Train", "Test"])
        return X_train, y_train, X_test, y_test


def _log_split_details(*feature_dfs: pd.DataFrame, names: list = None) -> None:
    """
    Log split details for an arbitrary number of DataFrame splits.
    
    Args:
        *feature_dfs: One or more DataFrames representing different splits.
        names (list): Optional list of names to assign to each split.
    """
    if names is None:
        default_names = ["Train", "Validation", "Test"]
        names = default_names[:len(feature_dfs)]
    logging.info("Split sizes:")
    for name, df in zip(names, feature_dfs):
        logging.info(f"{name}: {len(df)} samples")
        if len(df) == 0:
            warnings.warn(f"{name} set is empty! Check your split ratios.")
    logging.debug(f"Feature columns: {feature_dfs[0].columns.tolist()}")


def run_tf_activity_inference(
    X: pd.DataFrame,
    net: pd.DataFrame,
    min_n: int = 1,
    algorithm: str = "ulm"  # Default algorithm, can be changed dynamically
) -> pd.DataFrame:
    """
    Run TF activity inference on the input data using one of several decoupler algorithms.

    Args:
        X (pd.DataFrame): Gene expression matrix, including metadata columns.
        net (pd.DataFrame): Regulatory network for TF activity inference.
        min_n (int): Minimum number of targets for each TF.
        algorithm (str): Which decoupler algorithm to use. Options include:
                         "ulm", "viper", "aucell",
                         "mlm".

    Returns:
        pd.DataFrame: TF activity matrix with metadata reattached.

    Raises:
        ValueError: If required metadata columns or shared genes are missing,
                    or if an unsupported algorithm is provided.
        KeyError: If the expected key for the chosen algorithm is not found in AnnData.obsm.
    """
    import scanpy as sc
    import decoupler as dc
    import logging

    # Define expected metadata columns
    metadata_cols = {"cell_mfc_name", "viability", "pert_dose"}
    missing_cols = metadata_cols - set(X.columns)
    if missing_cols:
        raise ValueError(f"Missing expected metadata columns: {missing_cols}")

    # Separate metadata from gene expression data
    metadata = X[list(metadata_cols)]
    gene_expression = X.drop(columns=metadata_cols)

    # Determine shared genes between the network and gene expression data
    shared_genes = set(net["target"]).intersection(gene_expression.columns)
    if not shared_genes:
        raise ValueError("No shared genes found between network and gene expression matrix!")
    logging.debug(f"Number of shared genes: {len(shared_genes)}")

    # Filter the network and gene expression data to include only shared genes
    net_filtered = net[net["target"].isin(shared_genes)]
    logging.debug(f"Filtered network has {len(net_filtered)} interactions.")
    gene_expression = gene_expression[list(shared_genes)]

    # Create an AnnData object for the gene expression data
    adata = sc.AnnData(
        X=gene_expression.values,
        obs=pd.DataFrame(index=gene_expression.index),
        var=pd.DataFrame(index=gene_expression.columns),
    )
    logging.info(f"AnnData object created with shape: {adata.shape}")

    # Define a mapping from algorithm names to the corresponding decoupler function,
    # its keyword arguments, and the expected output key in adata.obsm.
    algorithm = algorithm.lower()
    methods = {
        "ulm": {
            "func": dc.run_ulm,
            "kwargs": {
                "source": "source",
                "target": "target",
                "weight": "weight",
                "min_n": min_n,
                "use_raw": False,
            },
            "estimate_key": "ulm_estimate",
        },
        "viper": {
            "func": dc.run_viper,
            "kwargs": {
                "source": "source",
                "target": "target",
                "use_raw": False,
            },
            "estimate_key": "viper_estimate",
        },
        "aucell": {
            "func": dc.run_aucell,
            "kwargs": {
                "source": "source",
                "target": "target",
                "n_up": min_n,
                "use_raw": False,
            },
            "estimate_key": "aucell_estimate",
        },
        "mlm": {
            "func": dc.run_mlm,
            "kwargs": {
                "source": "source",
                "target": "target",
                "weight": "weight",
                "use_raw": False,
            },
            "estimate_key": "mlm_estimate",
        },
    }

    if algorithm not in methods:
        raise ValueError(f"Algorithm '{algorithm}' is not supported. Choose from: {list(methods.keys())}")

    # Retrieve the decoupler function and its parameters from the mapping
    method = methods[algorithm]
    func = method["func"]
    kwargs = method["kwargs"]

    # Run the selected decoupler method
    func(mat=adata, net=net_filtered, **kwargs)
    estimate_key = method["estimate_key"]

    # Check that the expected output key is present in adata.obsm
    if estimate_key not in adata.obsm:
        raise KeyError(f"Expected key '{estimate_key}' not found in AnnData.obsm. "
                       "Ensure that the chosen algorithm populates obsm with its output.")

    # Extract the TF activity estimates from the AnnData object
    tf_activity = pd.DataFrame(adata.obsm[estimate_key], index=adata.obs.index)
    tf_activity.index = tf_activity.index.astype(int)  # Convert index to integers if needed
    
    # Set Nan values to 0.0 (if any)
    tf_activity.fillna(0.0, inplace=True)

    # Reattach the metadata columns to the output
    tf_activity = tf_activity.join(metadata)

    return tf_activity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets.")
    parser.add_argument(
        "--config_file",
        default="config.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config_file)
    # preprocess_tf_data(config, standardize=True)
    # preprocess_gene_data(config, standardize=True, feature_space="all")
    preprocess_gene_data(config, standardize=True, feature_space="all", chunk_size=3000)
    # preprocess_gene_data(config, standardize=True, feature_space="landmark")