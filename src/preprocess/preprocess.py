import argparse
import logging
import os

import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_file: str) -> dict:
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_dataset(file_path: str, delimiter: str = ",") -> pd.DataFrame:
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
    final_df: pd.DataFrame, x_df: pd.DataFrame, scale_features: bool
) -> pd.DataFrame:
    if scale_features:
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
    logging.info(f"Saving final dataset to {file_path}")
    try:
        final_df.to_csv(file_path, index=False)
        logging.info("Final dataset saved successfully.")
    except IOError as e:
        logging.error(f"Error saving the final dataset: {e}")
        raise


def preprocess_data(config: dict):
    try:
        # Load datasets
        compound_df = load_dataset(config["data_paths"]["compoundinfo_file"])
        x_df = load_dataset(config["data_paths"]["x_file"], delimiter="\t")
        y_df = load_dataset(config["data_paths"]["y_file"], delimiter="\t")

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
