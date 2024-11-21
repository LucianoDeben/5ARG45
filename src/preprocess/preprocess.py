import argparse
import logging
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_dataset(file_path: str, delimiter: str = ",") -> pd.DataFrame:
    """
    Load a dataset from a specified file path.

    Args:
        file_path (str): The path to the dataset file.
        delimiter (str, optional): Delimiter used in the file. Defaults to ','.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.ParserError: If there is an error parsing the file.
    """
    logging.info(f"Loading dataset from {file_path}")
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
        logging.info(f"Successfully loaded dataset with shape {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing the file {file_path}: {e}")
        raise


def merge_chemical_and_y(y_df: pd.DataFrame, compound_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the Y-label dataset with the compound dataset based on compound IDs.

    Args:
        y_df (pd.DataFrame): The Y-label dataset containing target variables.
        compound_df (pd.DataFrame): The compound dataset containing chemical information.

    Returns:
        pd.DataFrame: The merged dataset.

    Raises:
        ValueError: If the merge results in an empty DataFrame.
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
    logging.info(f"Shape after merging chemical and Y datasets: {merged_df.shape}")
    return merged_df


def merge_with_transcriptomic(
    merged_df: pd.DataFrame, x_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge the intermediate dataset with the transcriptomic dataset.

    Args:
        merged_df (pd.DataFrame): The dataset obtained after merging Y-labels and chemical data.
        x_df (pd.DataFrame): The transcriptomic dataset containing gene expression data.

    Returns:
        pd.DataFrame: The merged dataset.

    Raises:
        ValueError: If the merge results in an empty DataFrame.
    """
    logging.info("Merging with transcriptomic data")
    final_df = pd.merge(
        merged_df, x_df, left_on="cell_mfc_name", right_on="cell_line", how="inner"
    )
    if final_df.empty:
        logging.error("Merging with transcriptomic data resulted in an empty DataFrame")
        raise ValueError(
            "Merged DataFrame is empty after merging with transcriptomic data"
        )
    logging.info(f"Shape after merging with transcriptomic data: {final_df.shape}")
    return final_df


def preprocess_transcriptomic_features(
    final_df: pd.DataFrame, x_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Scale transcriptomic features using StandardScaler.

    Args:
        final_df (pd.DataFrame): The merged dataset containing transcriptomic features.
        x_df (pd.DataFrame): The original transcriptomic dataset to identify feature columns.

    Returns:
        pd.DataFrame: The dataset with scaled transcriptomic features.
    """
    logging.info("Scaling transcriptomic features")
    transcriptomic_features = [col for col in x_df.columns if col != "cell_line"]
    scaler = StandardScaler()
    try:
        final_df[transcriptomic_features] = scaler.fit_transform(
            final_df[transcriptomic_features]
        )
    except Exception as e:
        logging.error(f"Error scaling transcriptomic features: {e}")
        raise
    return final_df


def save_dataset(final_df: pd.DataFrame, file_path: str):
    """
    Save the preprocessed dataset to a specified file path.

    Args:
        final_df (pd.DataFrame): The preprocessed dataset to save.
        file_path (str): The destination file path.

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


def preprocess_data(compoundinfo_file: str, x_file: str, y_file: str, output_file: str):
    """
    Main function to orchestrate the preprocessing pipeline.

    Args:
        compoundinfo_file (str): Path to the compound information CSV file.
        x_file (str): Path to the transcriptomic features TSV file.
        y_file (str): Path to the Y-labels TSV file.
        output_file (str): Path to save the final preprocessed dataset.

    Raises:
        Exception: General exception if any step in the preprocessing fails.
    """
    try:
        # Load datasets
        compound_df = load_dataset(compoundinfo_file)
        x_df = load_dataset(x_file, delimiter="\t")
        y_df = load_dataset(y_file, delimiter="\t")

        # Merge datasets
        merged_df = merge_chemical_and_y(y_df, compound_df)
        final_df = merge_with_transcriptomic(merged_df, x_df)

        # Handle missing data
        logging.info(f"Handling missing data, initial shape: {final_df.shape}")
        final_df.dropna(inplace=True)
        logging.info(f"Shape after dropping missing data: {final_df.shape}")

        # Preprocess transcriptomic features
        final_df = preprocess_transcriptomic_features(final_df, x_df)

        # Save the final dataset
        save_dataset(final_df, output_file)
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets.")
    parser.add_argument(
        "--compoundinfo_file",
        default="data/raw/compoundinfo.csv",
        help="Path to compound info file.",
    )
    parser.add_argument(
        "--x_file", default="data/raw/X.tsv", help="Path to X feature file."
    )
    parser.add_argument(
        "--y_file", default="data/raw/Y.tsv", help="Path to Y labels file."
    )
    parser.add_argument(
        "--output_file",
        default="data/processed/final_dataset.csv",
        help="Path to save the final dataset.",
    )
    args = parser.parse_args()

    preprocess_data(
        compoundinfo_file=args.compoundinfo_file,
        x_file=args.x_file,
        y_file=args.y_file,
        output_file=args.output_file,
    )
