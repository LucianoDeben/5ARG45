import argparse
import logging
from typing import Tuple

import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_file: str) -> dict:
    """
    Load the configuration from a YAML file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_dataset(file_path: str, delimiter: str = ",", header: int = 0) -> pd.DataFrame:
    """
    Load a dataset from a specified file path.

    Args:
        file_path (str): The path to the dataset file.
        delimiter (str, optional): Delimiter used in the file. Defaults to ','.
        header (int, optional): Row number to use as column names. Defaults to 0.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.ParserError: If there is an error parsing the file.
    """
    logging.info(f"Loading dataset from {file_path}")
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, header=header)
        logging.info(f"Successfully loaded dataset with shape {df.shape}")
        # Check if the number of columns matches the expected number
        expected_columns = len(df.columns)
        if delimiter == "," and expected_columns == 1:
            raise pd.errors.ParserError(
                f"Expected multiple columns but got {expected_columns} column with delimiter '{delimiter}'"
            )
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing the file {file_path}: {e}")
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
    final_df: pd.DataFrame, x_df: pd.DataFrame, scale_features: bool
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


def save_dataset(final_df: pd.DataFrame, file_path: str):
    """
    Save the final DataFrame to a CSV file.

    Args:
        final_df (pd.DataFrame): Final DataFrame to save.
        file_path (str): Path to the output file.

    Raises:
        IOError: If there is an error saving the file.
    """
    logging.info(f"Saving final dataset to {file_path}")
    try:
        final_df.to_csv(file_path, index=False)
        logging.info("Final dataset saved successfully.")
    except IOError as e:
        logging.error(f"Error saving the final dataset: {e}")
        raise


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
        compound_df = load_dataset(config["data_paths"]["compoundinfo_file"])
        x_df = load_dataset(config["data_paths"]["x_file"], delimiter="\t", header=None)
        y_df = load_dataset(config["data_paths"]["y_file"], delimiter="\t")

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
        save_dataset(final_df, config["data_paths"]["output_file"])
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets.")
    parser.add_argument(
        "--config_file",
        default="config.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config_file)
    preprocess_data(config)
