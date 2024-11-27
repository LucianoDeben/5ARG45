import sys
from pathlib import Path

import pytest
from torch_geometric.loader.dataloader import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.preprocess.data_loader import prepare_chemical_data


def test_prepare_chemical_data_valid():
    smiles_list = ["CCO", "CCN", "CCC"]
    targets = [1.0, 2.0, 3.0]
    batch_size = 2

    data_loader = prepare_chemical_data(smiles_list, targets, batch_size)
    assert isinstance(data_loader, DataLoader)
    assert len(data_loader.dataset) == len(smiles_list)
    for data in data_loader.dataset:
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "edge_attr")
        assert hasattr(data, "y")


def test_prepare_chemical_data_invalid_smiles():
    smiles_list = ["CCO", "INVALID", "CCC"]
    targets = [1.0, 2.0, 3.0]
    batch_size = 2

    with pytest.raises(ValueError, match="No valid SMILES strings found in input"):
        prepare_chemical_data(["INVALID"], targets, batch_size)


def test_prepare_chemical_data_no_valid_graphs():
    smiles_list = ["INVALID1", "INVALID2"]
    targets = [1.0, 2.0]
    batch_size = 2

    with pytest.raises(
        ValueError, match="Number of valid SMILES doesn't match number of targets"
    ):
        prepare_chemical_data(smiles_list, targets, batch_size)


def test_prepare_chemical_data_mismatched_targets():
    smiles_list = ["CCO", "CCN", "CCC"]
    targets = [1.0, 2.0]  # Mismatched length

    with pytest.raises(
        ValueError, match="Number of valid SMILES doesn't match number of targets"
    ):
        prepare_chemical_data(smiles_list, targets, batch_size=2)
