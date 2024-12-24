import os
from typing import Dict

import pandas as pd
import yaml


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


if __name__ == "__main__":
    # Load the data
    smiles_df = pd.read_csv("../data/raw/compoundinfo_beta.txt", sep="\t")

    smiles_dict = create_smiles_dict(smiles_df)
    print(len(smiles_dict.values()))

    # Check the number of UNKNOWN values
    unknown_count = list(smiles_dict.values()).count("UNKNOWN")
    print(f"Number of UNKNOWN values: {unknown_count}")
    # Calculate the percentage of UNKNOWN values
    unknown_percentage = (unknown_count / len(smiles_dict)) * 100
    print(f"Percentage of UNKNOWN values: {unknown_percentage:.2f}%")
