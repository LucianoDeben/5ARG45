import argparse
import logging
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import decoupler as dc
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split

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
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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


def preprocess_tf_data(config: dict, standardize: bool = True):
    """
    Preprocess TF data according to the configuration.

    Args:
        config (dict): Configuration dictionary.
        standardize (bool): Whether to standardize the features.

    Raises:
        Exception: If any step in the preprocessing fails.
    """
    try:
        # Load datasets
        logging.debug("Loading datasets.")
        x_df = pd.read_csv(config["data_paths"]["x_file"], delimiter="\t")
        y_df = pd.read_csv(config["data_paths"]["y_file"], delimiter="\t")

        # Remove the first column from x_df (perturbation ID)
        logging.debug("Removing the first column from x_df.")
        x_df = x_df.iloc[:, 1:]

        # Standardize x_df
        if standardize:
            logging.debug("Standardizing x_df.")
            scaler = StandardScaler()
            x_df = pd.DataFrame(scaler.fit_transform(x_df), columns=x_df.columns)

        # Select relevant columns from y_df
        logging.debug(
            "Selecting 'viability', 'cell_mfc_name', and 'pert_dose' columns from y_df."
        )
        y_df = y_df[["viability", "cell_mfc_name", "pert_dose"]]

        # Check if x_df and y_df have the same length
        if x_df.shape[0] != y_df.shape[0]:
            logging.error(
                "The number of rows in x_df and y_df do not match. Cannot merge."
            )
            raise ValueError("The number of rows in x_df and y_df do not match.")

        # Concatenate the processed x_df with y_df
        logging.debug("Concatenating x_df and y_df.")
        final_df = pd.concat([x_df, y_df], axis=1)

        # Handle missing data
        logging.debug(f"Handling missing data, initial shape: {final_df.shape}")
        final_df.dropna(inplace=True)
        logging.debug(f"Shape after dropping missing data: {final_df.shape}")

        # Save the final dataset
        logging.debug("Saving the final dataset.")
        final_df.to_csv(config["data_paths"]["preprocessed_tf_file"], index=False)
        logging.info("Preprocessing completed successfully.")

        return final_df

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise


def preprocess_gene_data(
    config: dict, standardize: bool = True, feature_space: str = "all", chunk_size=2500
):
    """
    Preprocess gene expression data with memory-efficient writing.

    Args:
        config (dict): Configuration dictionary.
        standardize (bool): Whether to standardize the features.
        feature_space (str): Feature space to use ("all", "landmark", "best inferred").
        chunk_size (int): Chunk size for writing large CSV files.

    Raises:
        Exception: If any step in the preprocessing fails.
    """
    try:
        # Load datasets
        logging.debug("Loading datasets.")
        X_rna = np.fromfile(config["data_paths"]["rna_file"], dtype=np.float32).reshape(
            31567, -1
        )
        geneinfo = pd.read_csv(config["data_paths"]["geneinfo_file"], sep="\t")
        y_df = pd.read_csv(config["data_paths"]["y_file"], delimiter="\t")
        print(geneinfo.shape), print(y_df.shape)

        # Select genes based on user's choice
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

        # Select corresponding gene expression data
        selected_gene_indices = selected_genes.index
        X_selected = X_rna[:, selected_gene_indices]
        X_selected_df = pd.DataFrame(
            X_selected, index=y_df.index, columns=selected_genes.gene_symbol
        )

        # Standardize data if required
        if standardize:
            logging.debug("Standardizing gene expression data.")
            scaler = StandardScaler()
            X_selected_df = pd.DataFrame(
                scaler.fit_transform(X_selected_df), columns=X_selected_df.columns
            )

        # Merge with additional columns
        logging.debug(
            "Merging gene expression data with 'viability', 'cell_mfc_name', and 'pert_dose'."
        )
        final_df = pd.concat(
            [X_selected_df, y_df[["viability", "cell_mfc_name", "pert_dose"]]], axis=1
        )

        # Handle missing data
        final_df.dropna(inplace=True)
        logging.debug(f"Final DataFrame shape: {final_df.shape}")

        # Write to CSV in chunks to avoid memory issues
        logging.debug("Saving the preprocessed DataFrame in chunks.")
        output_file = (
            config["data_paths"]["preprocessed_landmark_file"]
            if feature_space == "landmark"
            else (
                config["data_paths"]["preprocessed_best_inferred_file"]
                if feature_space == "best inferred"
                else config["data_paths"]["preprocessed_gene_file"]
            )
        )

        with open(output_file, mode="w", newline="") as f:
            for start in range(0, final_df.shape[0], chunk_size):
                chunk = final_df.iloc[start : start + chunk_size]
                write_header = start == 0  # Write the header only once
                chunk.to_csv(f, mode="a", header=write_header, index=False)

        logging.info("Preprocessing and chunked saving completed successfully.")
        return final_df

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise


def merge_chemical_and_y(y_df: pd.DataFrame, compound_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge chemical data with Y-labels.

    Args:
        y_df (pd.DataFrame): DataFrame containing Y-labels.
        compound_df (pd.DataFrame): DataFrame containing chemical data.

    Returns:
        pd.DataFrame: Merged DataFrame.

    Raises:
        ValueError: If the merged DataFrame is empty.
    """
    logging.info("Merging chemical data with Y-labels")
    merged_df = pd.merge(
        y_df, compound_df, left_on="pert_mfc_id", right_on="pert_id", how="inner"
    )

    if merged_df.empty:
        logging.error(
            "Merging chemical data and Y-labels resulted in an empty DataFrame"
        )
        raise ValueError(
            "Merged DataFrame is empty after merging Y-labels and chemical data"
        )
    merged_df.drop(columns=["pert_id"], inplace=True)
    return merged_df


def merge_with_transcriptomic(
    merged_df: pd.DataFrame, x_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge the merged chemical and Y-labels DataFrame with transcriptomic data.

    Args:
        merged_df (pd.DataFrame): DataFrame containing merged chemical and Y-labels data.
        x_df (pd.DataFrame): DataFrame containing transcriptomic data.

    Returns:
        pd.DataFrame: Final merged DataFrame.

    Raises:
        ValueError: If the final merged DataFrame is empty.
    """
    logging.info("Merging with transcriptomic data")
    final_df = pd.merge(
        merged_df, x_df, left_on="sig_id", right_on="cell_line", how="inner"
    )
    final_df.drop(columns=["cell_line"], inplace=True)
    if final_df.empty:
        logging.error("Merging with transcriptomic data resulted in an empty DataFrame")
        raise ValueError(
            "Merged DataFrame is empty after merging with transcriptomic data"
        )
    logging.info(f"Shape after merging with transcriptomic data: {final_df.shape}")
    return final_df


def preprocess_transcriptomic_features(
    final_df: pd.DataFrame, x_df: pd.DataFrame, scale_features: bool = True
) -> pd.DataFrame:
    """
    Preprocess transcriptomic features by scaling them if required.

    Args:
        final_df (pd.DataFrame): Final merged DataFrame.
        x_df (pd.DataFrame): DataFrame containing transcriptomic data.
        scale_features (bool): Whether to scale the transcriptomic features.

    Returns:
        pd.DataFrame: DataFrame with preprocessed transcriptomic features.

    Raises:
        ValueError: If there is an error during scaling.
    """
    if scale_features:
        logging.info("Scaling transcriptomic features")
        transcriptomic_features = [col for col in x_df.columns if col != "cell_line"]
        scaler = StandardScaler()
        try:
            final_df[transcriptomic_features] = scaler.fit_transform(
                final_df[transcriptomic_features]
            )
        except ValueError as e:
            logging.error(f"Error scaling transcriptomic features: {e}")
            raise ValueError(f"Error scaling transcriptomic features: {e}")
    return final_df


def partition_data(
    final_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition the merged dataset into chemical compounds, target viability scores, and gene expression data.

    Args:
        final_df (pd.DataFrame): The merged DataFrame containing all data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - DataFrame containing chemical compounds (SMILES strings).
            - DataFrame containing target viability scores.
            - DataFrame containing gene expression data.
    """
    # Extract chemical compounds (SMILES strings)
    chemical_compounds_df = final_df[["canonical_smiles"]].copy()

    # Extract target viability scores
    viability_df = final_df[["viability"]].copy()

    # Extract gene expression data (assuming columns are labeled from 1 to 682 and are the last columns)
    gene_expression_df = final_df.iloc[:, -682:].copy()

    return chemical_compounds_df, viability_df, gene_expression_df


def preprocess_data(config: dict):
    """
    Preprocess data according to the configuration.

    Args:
        config (dict): Configuration dictionary.

    Raises:
        Exception: If any step in the preprocessing fails.
    """
    try:
        # Load datasets
        compound_df = pd.read_csv(config["data_paths"]["compoundinfo_file"])
        x_df = pd.read_csv(config["data_paths"]["x_file"], delimiter="\t", header=None)
        y_df = pd.read_csv(config["data_paths"]["y_file"], delimiter="\t")

        # Set the name of the first column in x_df to 'cell_line'
        x_df.columns = ["cell_line"] + x_df.columns.tolist()[1:]
        logging.info(
            f"Columns in x_df after setting column names: {x_df.columns.tolist()}"
        )

        # Merge datasets
        merged_df = merge_chemical_and_y(y_df, compound_df)
        final_df = merge_with_transcriptomic(merged_df, x_df)

        # Handle missing data
        logging.info(f"Handling missing data, initial shape: {final_df.shape}")
        final_df.dropna(inplace=True)
        logging.info(f"Shape after dropping missing data: {final_df.shape}")

        # Preprocess transcriptomic features
        final_df = preprocess_transcriptomic_features(
            final_df, x_df, config["preprocess_params"]["scale_features"]
        )

        # Save the final dataset
        final_df.to_csv(final_df, config["data_paths"]["output_file"])
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise


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


def _log_split_details(*feature_dfs: pd.DataFrame) -> None:
    """Log split details for debugging"""
    logger.info("Split sizes:")
    for name, df in zip(["Train", "Validation", "Test"], feature_dfs):
        logger.info(f"{name}: {len(df)} samples")
        if len(df) == 0:
            warnings.warn(f"{name} set is empty! Check your split ratios.")
    logger.debug(f"Feature columns: {feature_dfs[0].columns.tolist()}")


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


def run_tf_activity_inference(
    X: pd.DataFrame, net: pd.DataFrame, min_n: int = 1
) -> pd.DataFrame:
    """
    Run TF activity inference on the input data.

    Args:
        X (pd.DataFrame): Gene expression matrix, including metadata columns.
        net (pd.DataFrame): Regulatory network for TF activity inference.
        min_n (int): Minimum number of targets for each TF.

    Returns:
        pd.DataFrame: TF activity matrix with metadata reattached.

    Raises:
        ValueError: If no shared genes are found between network and gene expression matrix.
    """
    # Define metadata columns
    metadata_cols = {"cell_mfc_name", "viability", "pert_dose"}
    missing_cols = metadata_cols - set(X.columns)

    if missing_cols:
        raise ValueError(f"Missing expected metadata columns: {missing_cols}")

    metadata = X[list(metadata_cols)]
    gene_expression = X.drop(columns=metadata_cols)

    # Filter the network for shared genes
    shared_genes = set(net["target"]).intersection(gene_expression.columns)
    if not shared_genes:
        raise ValueError(
            "No shared genes found between network and gene expression matrix!"
        )

    logging.debug(f"Number of shared genes: {len(shared_genes)}")

    # Filter network and gene expression data
    net_filtered = net[net["target"].isin(shared_genes)]
    logging.debug(f"Filtered network has {len(net_filtered)} interactions.")
    gene_expression = gene_expression[list(shared_genes)]

    # Create AnnData object
    adata = sc.AnnData(
        X=gene_expression.values,
        obs=pd.DataFrame(index=gene_expression.index),
        var=pd.DataFrame(index=gene_expression.columns),
    )
    logging.info(f"AnnData object created with shape: {adata.shape}")

    # Run ULM for TF activity inference
    dc.run_ulm(
        mat=adata,
        net=net_filtered,
        source="source",
        target="target",
        weight="weight",
        min_n=min_n,
        use_raw=False,
    )

    # Extract TF activity estimates
    tf_activity = pd.DataFrame(adata.obsm["ulm_estimate"], index=adata.obs.index)
    tf_activity.index = tf_activity.index.astype(int)  # Convert index to integers

    # Reattach metadata columns
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
    preprocess_gene_data(
        config, standardize=True, feature_space="best inferred", chunk_size=5000
    )
    # preprocess_gene_data(config, standardize=True, feature_space="landmark")
