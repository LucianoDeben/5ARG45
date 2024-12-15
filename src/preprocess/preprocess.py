import argparse
import logging
import sys

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
    Preprocess data according to the configuration.

    Args:
        config (dict): Configuration dictionary.

    Raises:
        Exception: If any step in the preprocessing fails.
    """
    try:
        # Load datasets
        logging.debug("Loading datasets.")
        x_df = pd.read_csv(config["data_paths"]["x_file"], delimiter="\t")
        y_df = pd.read_csv(config["data_paths"]["y_file"], delimiter="\t")

        # Remove the first column from x_df
        logging.debug("Removing the first column from x_df.")
        x_df = x_df.iloc[:, 1:]

        # Standardize x_df
        if standardize:
            logging.debug("Standardizing x_df.")
            scaler = StandardScaler()
            x_df = pd.DataFrame(scaler.fit_transform(x_df), columns=x_df.columns)

        # Only select the 'viability' column from y_df
        logging.debug("Selecting the 'viability' column from y_df.")
        y_df = y_df[["viability"]]

        # Check if x_df and y_df have the same length, then concatenate/merge them
        if x_df.shape[0] != y_df.shape[0]:
            logging.error(
                "The number of rows in x_df and y_df do not match. Cannot merge."
            )
            raise ValueError("The number of rows in x_df and y_df do not match.")

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

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logging.error(f"Empty data error: {e}")
        raise
    except pd.errors.ParserError as e:
        logging.error(f"Parsing error: {e}")
        raise
    except ValueError as e:
        logging.error(f"Value error: {e}")
        raise
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise


def preprocess_gene_data(
    config: dict, standardize: bool = True, use_landmarks: bool = True
):
    """
    Preprocess data according to the configuration.

    Args:
        config (dict): Configuration dictionary.

    Raises:
        Exception: If any step in the preprocessing fails.
    """
    try:
        # Load datasets
        logging.debug("Loading datasets.")

        # Get gene expressions data from binary file and reshape it
        X_rna = np.fromfile(config["data_paths"]["rna_file"], dtype=np.float32)
        X_rna = X_rna.reshape(31567, -1)

        # Get the gene information
        geneinfo = pd.read_csv(config["data_paths"]["geneinfo_file"], sep="\t")

        # Get the cell line information and viability data
        y_df = pd.read_csv(config["data_paths"]["y_file"], delimiter="\t")

        # Select genes based on the user's choice
        if use_landmarks:
            logging.debug("Using landmark genes.")
            selected_genes = geneinfo[geneinfo.feature_space == "landmark"]
        else:
            logging.debug("Using all genes.")
            selected_genes = geneinfo

        # Select the corresponding gene expression data
        selected_gene_indices = selected_genes.index
        X_selected = X_rna[:, selected_gene_indices]
        X_selected_df = pd.DataFrame(
            X_selected, index=y_df.index, columns=selected_genes.gene_symbol
        )

        # Remove the first column from X_selected_df
        logging.debug("Removing the first column from X_selected_df.")
        X_selected_df = X_selected_df.iloc[:, 1:]
        logging.debug(
            f"Shape of x_df after removing the first column: {X_selected_df.shape}"
        )

        # Standardize the gene expression data if required
        if standardize:
            logging.debug("Standardizing gene expression data.")
            scaler = StandardScaler()
            X_selected_df = pd.DataFrame(
                scaler.fit_transform(X_selected_df), columns=X_selected_df.columns
            )

        # Merge the gene expression data with the viability scores
        logging.debug("Merging gene expression data with viability scores.")
        final_df = pd.concat([X_selected_df, y_df[["viability"]]], axis=1)

        # Remove rows with any NaN values
        logging.debug(f"Handling missing data, initial shape: {final_df.shape}")
        final_df.dropna(inplace=True)
        logging.debug(f"Shape after dropping missing data: {final_df.shape}")

        # Save the final dataset
        if use_landmarks:
            logging.debug("Saving the final dataset with landmark genes.")
            final_df.to_csv(
                config["data_paths"]["preprocessed_landmark_file"], index=False
            )
        else:
            logging.debug("Saving the final dataset with all genes.")
            final_df.to_csv(config["data_paths"]["preprocessed_gene_file"], index=False)
        logging.info("Preprocessing completed successfully.")
        return final_df

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logging.error(f"Empty data error: {e}")
        raise
    except pd.errors.ParserError as e:
        logging.error(f"Parsing error: {e}")
        raise
    except ValueError as e:
        logging.error(f"Value error: {e}")
        raise
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
    random_state=42,
):
    """
    Split the DataFrame into train, validation, and test sets, optionally stratified by a column.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        config (dict): Configuration dictionary for train/val/test ratios.
        target_name (str): The name of the target column.
        stratify_by (str): Column name to stratify by (e.g., 'cell_mfc_name').
        random_state (int): The random seed for reproducibility.

    Returns:
        tuple: Splits of the dataset into features and labels for train, validation, and test sets.
    """
    if stratify_by:
        groups = df[stratify_by]

        # Grouped split: train and temp (val + test)
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=1 - config["preprocess"]["train_ratio"],
            random_state=random_state,
        )
        train_idx, temp_idx = next(gss.split(df, groups=groups, groups=groups))

        train_df = df.iloc[train_idx]
        temp_df = df.iloc[temp_idx]

        # Further split temp into validation and test
        gss_temp = GroupShuffleSplit(
            n_splits=1,
            test_size=config["preprocess"]["test_ratio"]
            / (config["preprocess"]["test_ratio"] + config["preprocess"]["val_ratio"]),
            random_state=random_state,
        )
        val_idx, test_idx = next(
            gss_temp.split(
                temp_df, groups=temp_df[stratify_by], groups=temp_df[stratify_by]
            )
        )

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

    # Split datasets into features (X) and target labels (y)
    X_train, y_train = train_df.drop(target_name, axis=1), train_df[target_name]
    X_val, y_val = val_df.drop(target_name, axis=1), val_df[target_name]
    X_test, y_test = test_df.drop(target_name, axis=1), test_df[target_name]

    print(
        f"Train Shape: {X_train.shape}, Validation Shape: {X_val.shape}, Test Shape: {X_test.shape}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets.")
    parser.add_argument(
        "--config_file",
        default="config.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config_file)
    preprocess_tf_data(config, standardize=True)
    preprocess_gene_data(config, standardize=True, use_landmarks=True)
    preprocess_gene_data(config, standardize=True, use_landmarks=False)
