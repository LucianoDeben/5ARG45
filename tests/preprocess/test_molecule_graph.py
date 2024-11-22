import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.preprocess.molecule_graph import (
    _feature_cache,
    clear_cache,
    collect_continuous_atom_features_parallel,
    get_atom_features,
    get_bond_features,
    mol_to_graph,
    process_smiles,
)


@pytest.fixture
def sample_smiles():
    """Provide a valid SMILES string."""
    return "CCO"  # Ethanol


@pytest.fixture
def invalid_smiles():
    """Provide an invalid SMILES string."""
    return "INVALID_SMILES"


@pytest.fixture
def scaler():
    """Provide a fitted StandardScaler."""
    scaler = StandardScaler()
    scaler.fit([[0.0, 1, 0, 1], [0.1, 2, -1, 0]])  # Example feature ranges
    return scaler


def test_mol_to_graph_valid(sample_smiles, scaler):
    """Test graph creation for a valid SMILES string."""
    graph = mol_to_graph(sample_smiles, scaler)
    assert graph is not None
    assert isinstance(graph, Data)
    assert graph.x.shape[1] > 0  # Atom features
    assert graph.edge_index.shape[0] == 2  # Edge index
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]  # Matching edges


def test_mol_to_graph_invalid(invalid_smiles, scaler):
    """Test graph creation for an invalid SMILES string."""
    graph = mol_to_graph(invalid_smiles, scaler)
    assert graph is None


def test_get_atom_features(sample_smiles, scaler):
    """Test atom feature extraction."""
    mol = Chem.MolFromSmiles(sample_smiles)
    Chem.AddHs(mol)
    atom = mol.GetAtomWithIdx(0)
    features = get_atom_features(atom, scaler)
    assert len(features) > 0
    assert isinstance(features, list)


def test_get_bond_features(sample_smiles):
    """Test bond feature extraction."""
    mol = Chem.MolFromSmiles(sample_smiles)
    bond = mol.GetBondWithIdx(0)
    features = get_bond_features(bond)
    assert len(features) > 0
    assert isinstance(features, list)


def test_process_smiles_valid(sample_smiles, scaler):
    """Test SMILES processing for a valid string."""
    graph = process_smiles(sample_smiles, scaler)
    assert graph is not None
    assert isinstance(graph, Data)


def test_process_smiles_invalid(invalid_smiles, scaler):
    """Test SMILES processing for an invalid string."""
    graph = process_smiles(invalid_smiles, scaler)
    assert graph is None


def test_collect_continuous_atom_features_parallel(sample_smiles):
    """Test parallel feature collection."""
    smiles_list = [sample_smiles, sample_smiles, "INVALID_SMILES"]
    features = collect_continuous_atom_features_parallel(smiles_list, n_jobs=2)
    assert isinstance(features, np.ndarray)
    assert features.shape[1] > 0  # Continuous features


def test_clear_cache():
    """Test clearing the feature cache."""
    clear_cache()
    assert len(_feature_cache) == 0
