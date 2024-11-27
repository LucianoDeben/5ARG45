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
    get_atom_features,
    get_bond_features,
    mol_to_graph,
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


def test_clear_cache():
    """Test clearing the feature cache."""
    clear_cache()
    assert len(_feature_cache) == 0


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


def test_mol_to_graph_cache(sample_smiles, scaler):
    """Test graph creation with caching."""
    clear_cache()
    graph1 = mol_to_graph(sample_smiles, scaler)
    graph2 = mol_to_graph(sample_smiles, scaler)
    assert graph1 is not None
    assert graph2 is not None
    assert graph1 is graph2  # Should be the same object from cache


def test_mol_to_graph_no_hydrogens(sample_smiles, scaler):
    """Test graph creation without adding hydrogens."""
    mol = Chem.MolFromSmiles(sample_smiles)
    graph = mol_to_graph(Chem.MolToSmiles(mol), scaler)
    assert graph is not None
    assert isinstance(graph, Data)
    assert graph.x.shape[1] > 0  # Atom features
    assert graph.edge_index.shape[0] == 2  # Edge index
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]  # Matching edges


def test_mol_to_graph_no_3d_coordinates(sample_smiles, scaler):
    """Test graph creation without generating 3D coordinates."""
    mol = Chem.MolFromSmiles(sample_smiles)
    mol = Chem.AddHs(mol)
    graph = mol_to_graph(Chem.MolToSmiles(mol), scaler)
    assert graph is not None
    assert isinstance(graph, Data)
    assert graph.x.shape[1] > 0  # Atom features
    assert graph.edge_index.shape[0] == 2  # Edge index
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]  # Matching edges
