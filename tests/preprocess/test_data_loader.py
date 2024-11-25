import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader as GeometricDataLoader

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.preprocess.data_loader import (
    prepare_chemical_data,
    prepare_transcriptomics_data,
    validate_smiles_list,
)
from src.preprocess.molecule_graph import process_smiles


@pytest.fixture
def smiles_list():
    """Provide a list of SMILES strings."""
    return ["CCO", "CCC", "CCN"]  # Valid SMILES strings


@pytest.fixture
def invalid_smiles_list():
    """Provide a list containing invalid SMILES strings."""
    return ["CCO", "INVALID_SMILES", "CCC"]


@pytest.fixture
def targets():
    """Provide a list of target values."""
    return [0.5, 0.8, 0.2]


@pytest.fixture
def transcriptomics_df():
    """Provide a sample transcriptomics DataFrame."""
    return pd.DataFrame(
        {
            "gene_1": [0.1, 0.2, 0.3],
            "gene_2": [0.4, 0.5, 0.6],
            "gene_3": [0.7, 0.8, 0.9],
        }
    )


@pytest.fixture
def scaler():
    """Provide a fitted StandardScaler."""
    scaler = StandardScaler()
    scaler.fit([[0.0, 1, 0, 1], [0.1, 2, -1, 0]])  # Example feature ranges
    return scaler


def test_prepare_chemical_data(smiles_list, targets):
    """Test chemical data preparation with valid inputs."""
    loader = prepare_chemical_data(smiles_list, targets, batch_size=2)

    assert isinstance(loader, GeometricDataLoader)
    for batch in loader:
        assert hasattr(batch, "x")  # Atom features
        assert hasattr(batch, "edge_index")  # Edge connections
        assert hasattr(batch, "y")  # Targets
        assert batch.y.shape[1] == 1  # Single target per molecule


def test_prepare_chemical_data_with_invalid_smiles(invalid_smiles_list, targets):
    """Test chemical data preparation with invalid SMILES included."""
    loader = prepare_chemical_data(invalid_smiles_list, targets, batch_size=2)

    assert isinstance(loader, GeometricDataLoader)
    for batch in loader:
        assert hasattr(batch, "x")  # Atom features
        assert hasattr(batch, "edge_index")  # Edge connections
        assert hasattr(batch, "y")  # Targets


def test_prepare_transcriptomics_data(transcriptomics_df, targets):
    """Test transcriptomics data preparation."""
    loader = prepare_transcriptomics_data(transcriptomics_df, targets, batch_size=1)

    assert isinstance(loader, torch.utils.data.DataLoader)
    for batch in loader:
        inputs, labels = batch
        assert inputs.shape[1] == transcriptomics_df.shape[1]  # Features
        assert labels.shape[1] == 1  # Targets


def test_validate_smiles_list(smiles_list, invalid_smiles_list):
    """Test SMILES validation."""
    valid_smiles = validate_smiles_list(smiles_list)
    assert len(valid_smiles) == len(smiles_list)  # All SMILES are valid

    valid_smiles = validate_smiles_list(invalid_smiles_list)
    assert len(valid_smiles) == len(smiles_list) - 1  # One invalid SMILES removed


def test_process_smiles_valid(smiles_list, scaler):
    """Test individual SMILES processing."""
    for smiles in smiles_list:
        graph = process_smiles(smiles, scaler)
        assert graph is not None
        assert hasattr(graph, "x")  # Atom features
        assert hasattr(graph, "edge_index")  # Edge connections


def test_process_smiles_invalid(invalid_smiles_list, scaler):
    """Test processing invalid SMILES strings."""
    for smiles in invalid_smiles_list:
        if Chem.MolFromSmiles(smiles) is None:
            assert process_smiles(smiles, scaler) is None
