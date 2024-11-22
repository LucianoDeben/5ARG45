import pandas as pd
import torch
from molecule_graph import collect_continuous_atom_features, mol_to_graph
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import DataLoader


def prepare_chemical_data(smiles_list, targets, batch_size=32):
    """
    Prepare chemical data for training.

    Args:
        smiles_list (List[str]): List of SMILES strings.
        targets (List[float]): List of target values.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: PyTorch Geometric DataLoader for the chemical data.
    """
    # Collect continuous features and fit scaler
    continuous_atom_features = collect_continuous_atom_features(smiles_list)
    scaler = StandardScaler()
    scaler.fit(continuous_atom_features)

    # Convert SMILES to graphs with scaled features and add targets
    graph_list = []
    for smiles, target in zip(smiles_list, targets):
        graph = mol_to_graph(smiles, scaler)
        if graph is not None:
            graph.y = torch.tensor(
                [target], dtype=torch.float
            )  # Add target to data object
            graph_list.append(graph)

    # Create DataLoader
    data_loader = DataLoader(graph_list, batch_size=batch_size, shuffle=True)
    return data_loader


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
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
