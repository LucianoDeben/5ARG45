from concurrent.futures import ProcessPoolExecutor
from typing import List

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import DataLoader
from torch_geometric.data import DataLoader as GeometricDataLoader

from .molecule_graph import (
    collect_continuous_atom_features,
    mol_to_graph,
    process_smiles,
)


def prepare_chemical_data(
    smiles_list: List[str], targets: List[float], batch_size: int = 32
) -> GeometricDataLoader:  # type: ignore
    """
    Prepare chemical data for training.

    Args:
        smiles_list (List[str]): List of SMILES strings.
        targets (List[float]): List of target values.
        batch_size (int): Batch size for DataLoader.

    Returns:
        GeometricDataLoader: DataLoader for chemical data.
    """
    # Ensure smiles_list contains only strings
    smiles_list = [str(smiles) for smiles in smiles_list]

    # Collect continuous atom features and fit scaler
    continuous_atom_features = collect_continuous_atom_features(smiles_list)
    scaler = StandardScaler()
    scaler.fit(continuous_atom_features)

    # Convert SMILES to graph data using parallel processing
    data_list = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_smiles, smiles, scaler) for smiles in smiles_list
        ]
        for future, target in zip(futures, targets):
            data = future.result()
            if data is not None:
                data.y = torch.tensor([target], dtype=torch.float)
                data_list.append(data)

    return GeometricDataLoader(data_list, batch_size=batch_size, shuffle=True)


def prepare_transcriptomics_data(transcriptomics_df, targets, batch_size=32):
    """
    Prepare transcriptomics data for training.

    Args:
        transcriptomics_df (pd.DataFrame): DataFrame containing transcriptomics features.
        targets (List[float]): List of target values.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: PyTorch DataLoader for the transcriptomics data.
    """
    # Convert DataFrame to tensor
    transcriptomics_tensor = torch.tensor(transcriptomics_df.values, dtype=torch.float)
    targets_tensor = torch.tensor(targets, dtype=torch.float).unsqueeze(1)

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(transcriptomics_tensor, targets_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
