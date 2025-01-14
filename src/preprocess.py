import argparse
import logging
import sys

import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.append("src")

from typing import Tuple

import numpy as np
import pandas as pd
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
    df,
    config,
    target_name="target_name",
    stratify_by=None,
    keep_columns=None,
    random_state=42,
):
    """
    Split the DataFrame into train, validation, and test sets, optionally stratified by a column.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        config (dict): Configuration dictionary for train/val/test ratios.
        target_name (str): The name of the target column.
        stratify_by (str): Column name to stratify by (e.g., 'cell_mfc_name').
        keep_columns (list): List of additional columns to retain (e.g., 'pert_dose').
        random_state (int): The random seed for reproducibility.

    Returns:
        tuple: Splits of the dataset into features and labels for train, validation, and test sets.
    """
    # Ensure keep_columns is a list
    keep_columns = keep_columns or []

    if stratify_by:
        groups = df[stratify_by]

        if groups.nunique() < 3:
            raise ValueError(
                f"Not enough unique groups in '{stratify_by}' for stratified splitting."
            )

        # Grouped split: train and temp (val + test)
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=1 - config["preprocess"]["train_ratio"],
            random_state=random_state,
        )
        train_idx, temp_idx = next(gss.split(df, groups=groups))

        train_df = df.iloc[train_idx]
        temp_df = df.iloc[temp_idx]

        # Further split temp into validation and test
        gss_temp = GroupShuffleSplit(
            n_splits=1,
            test_size=config["preprocess"]["test_ratio"]
            / (config["preprocess"]["test_ratio"] + config["preprocess"]["val_ratio"]),
            random_state=random_state,
        )
        val_idx, test_idx = next(gss_temp.split(temp_df, groups=temp_df[stratify_by]))

        val_df = temp_df.iloc[val_idx]
        test_df = temp_df.iloc[test_idx]
    else:
        # Default randomized splitting if no stratification is specified
        train_df, temp_df = train_test_split(
            df,
            test_size=1 - config["preprocess"]["train_ratio"],
            random_state=random_state,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=config["preprocess"]["test_ratio"]
            / (config["preprocess"]["test_ratio"] + config["preprocess"]["val_ratio"]),
            random_state=random_state,
        )

    # Drop target and additional columns from X
    exclude_columns = [target_name, stratify_by, "pert_dose"] + keep_columns
    exclude_columns = [col for col in exclude_columns if col in df.columns]

    X_train = train_df.drop(columns=exclude_columns, errors="ignore")
    y_train = train_df[target_name]

    X_val = val_df.drop(columns=exclude_columns, errors="ignore")
    y_val = val_df[target_name]

    X_test = test_df.drop(columns=exclude_columns, errors="ignore")
    y_test = test_df[target_name]

    # Validate group distribution in splits
    if stratify_by:
        logging.info(f"Train Groups: {train_df[stratify_by].nunique()} unique values.")
        logging.info(
            f"Validation Groups: {val_df[stratify_by].nunique()} unique values."
        )
        logging.info(f"Test Groups: {test_df[stratify_by].nunique()} unique values.")

    logging.debug(
        f"Train Shape: {X_train.shape}, Validation Shape: {X_val.shape}, Test Shape: {X_test.shape}"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


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
    preprocess_gene_data(config, standardize=True, feature_space="best inferred")
    # preprocess_gene_data(config, standardize=True, feature_space="landmark")
