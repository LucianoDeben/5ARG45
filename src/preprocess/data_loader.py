from typing import List

import pandas as pd
import torch
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader

from .molecule_graph import collect_continuous_features, mol_to_graph


def prepare_chemical_data(
    smiles_list: List[str], targets: List[float], batch_size: int = 32
) -> GeometricDataLoader:  # type: ignore
    """
    Prepare chemical data for training.

    Args:
        smiles_list: List of SMILES strings
        targets: List of target values
        batch_size: Batch size for DataLoader

    Returns:
        GeometricDataLoader with processed molecular graphs
    """
    # Validate and convert SMILES
    smiles_list = [str(smiles) for smiles in smiles_list]
    valid_smiles = validate_smiles_list(smiles_list)

    if not valid_smiles:
        raise ValueError("No valid SMILES strings found in input")

    # Fit scaler on continuous features
    continuous_features = collect_continuous_features(valid_smiles)
    scaler = StandardScaler()
    scaler.fit(continuous_features)

    # Convert SMILES to graphs
    data_list = []
    for smiles, target in zip(valid_smiles, targets):
        try:
            data = mol_to_graph(smiles, scaler)
            if data is not None:
                data.y = torch.tensor([target], dtype=torch.float)
                data_list.append(data)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {str(e)}")

    if not data_list:
        raise ValueError("No valid graphs were created")

    print(f"Processed {len(data_list)} valid graphs out of {len(smiles_list)} SMILES")
    return GeometricDataLoader(data_list, batch_size=batch_size, shuffle=True)


def prepare_transcriptomics_data(
    transcriptomics_df: pd.DataFrame, targets: List[float], batch_size: int = 32
) -> DataLoader:
    """
    Prepare transcriptomics data for training.

    Args:
        transcriptomics_df: DataFrame with transcriptomics features
        targets: List of target values
        batch_size: Batch size for DataLoader

    Returns:
        DataLoader with transcriptomics data
    """
    # Validate inputs
    if transcriptomics_df.empty:
        raise ValueError("Empty transcriptomics DataFrame")
    if len(targets) != len(transcriptomics_df):
        raise ValueError("Number of targets doesn't match number of samples")

    # Convert to tensors
    transcriptomics_tensor = torch.tensor(transcriptomics_df.values, dtype=torch.float)
    targets_tensor = torch.tensor(targets, dtype=torch.float).unsqueeze(1)

    # Create dataset and loader
    dataset = TensorDataset(transcriptomics_tensor, targets_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def validate_smiles_list(smiles_list: List[str]) -> List[str]:
    """
    Validate and clean SMILES strings.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of valid SMILES strings
    """
    valid_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
        else:
            print(f"Invalid SMILES: {smiles}")
    return valid_smiles
