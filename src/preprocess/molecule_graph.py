from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType, HybridizationType
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# Cache for computed features
_feature_cache: Dict[str, Tuple[np.ndarray, List[List[int]], List[List[float]]]] = {}


def collect_continuous_atom_features_parallel(
    smiles_list: List[str], n_jobs: int = -1
) -> np.ndarray:
    """
    Collect continuous atom features from all molecules in parallel.

    Args:
        smiles_list (List[str]): List of SMILES strings.
        n_jobs (int): Number of parallel jobs. -1 means using all processors.

    Returns:
        np.ndarray: Array of continuous features from all atoms.
    """
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        features_lists = list(executor.map(_collect_single_mol_features, smiles_list))

    continuous_features = [f for f in features_lists if f is not None]
    return np.vstack(continuous_features) if continuous_features else np.array([])


def _collect_single_mol_features(smiles: str) -> Optional[np.ndarray]:
    """Helper function to collect features from a single molecule."""
    try:
        if smiles in _feature_cache:
            return _feature_cache[smiles][0]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            return None

        mol = Chem.AddHs(mol)
        success = AllChem.EmbedMolecule(mol, randomSeed=42)
        if success != 0:
            print(f"Failed to embed molecule: {smiles}")
            return None

        AllChem.ComputeGasteigerCharges(mol)
        features = []

        for atom in mol.GetAtoms():
            partial_charge = float(
                atom.GetProp("_GasteigerCharge")
                if atom.HasProp("_GasteigerCharge")
                else "0.0"
            )
            degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge()
            num_h = atom.GetTotalNumHs()
            features.append([partial_charge, degree, formal_charge, num_h])

        return np.array(features)
    except Exception as e:
        print(f"Error processing {smiles}: {str(e)}")
        return None


def one_hot_encode(value: str, allowable_values: List[str]) -> List[int]:
    """One-hot encode a value among allowable values."""
    return [int(value == v) for v in allowable_values]


def get_atom_features(atom: Chem.Atom, scaler: StandardScaler) -> List[float]:
    """
    Extract features from an RDKit Atom object.

    Args:
        atom: RDKit Atom object
        scaler: Fitted StandardScaler for continuous features

    Returns:
        List of numerical features
    """
    partial_charge = (
        float(atom.GetProp("_GasteigerCharge"))
        if atom.HasProp("_GasteigerCharge")
        else 0.0
    )
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    num_h = atom.GetTotalNumHs()

    continuous_features = [partial_charge, degree, formal_charge, num_h]
    scaled_features = scaler.transform([continuous_features])[0].tolist()

    hybridization_types = [
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
    ]
    hybridization = [int(atom.GetHybridization() == ht) for ht in hybridization_types]

    aromatic = [int(atom.GetIsAromatic())]

    chirality = [
        int(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
        int(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW),
        int(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW),
        int(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_OTHER),
    ]

    in_ring = [int(atom.IsInRing())]

    features = scaled_features + hybridization + aromatic + chirality + in_ring

    return features


def get_bond_features(bond: Chem.Bond) -> List[float]:
    """Extract bond features."""
    features = []

    bond_types = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
    bond_type = one_hot_encode(str(bond.GetBondType()), [str(bt) for bt in bond_types])
    features.extend(bond_type)

    stereo = one_hot_encode(
        str(bond.GetStereo()),
        [
            "STEREONONE",
            "STEREOANY",
            "STEREOZ",
            "STEREOE",
            "STEREOCIS",
            "STEREOTRANS",
            "STEREOOTHER",
        ],
    )
    features.extend(stereo)

    features.append(int(bond.GetIsConjugated()))
    features.append(int(bond.IsInRing()))

    return features


def mol_to_graph(smiles: str, scaler: StandardScaler) -> Optional[Data]:
    """Convert SMILES to PyTorch Geometric Data object."""
    try:
        if smiles in _feature_cache:
            cached_features = _feature_cache[smiles]
            return _create_graph_from_features(*cached_features)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            return None

        mol = Chem.AddHs(mol)
        success = AllChem.EmbedMolecule(mol, randomSeed=42)
        if success != 0:
            print(f"Failed to embed molecule: {smiles}")
            return None

        AllChem.ComputeGasteigerCharges(mol)

        atom_features = [get_atom_features(atom, scaler) for atom in mol.GetAtoms()]

        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_features = get_bond_features(bond)

            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([bond_features, bond_features])

        if len(edge_index) != len(edge_attr):
            raise ValueError(f"Inconsistent edges in molecule: {smiles}")

        _feature_cache[smiles] = (np.array(atom_features), edge_index, edge_attr)

        return _create_graph_from_features(
            np.array(atom_features), edge_index, edge_attr
        )

    except Exception as e:
        print(f"Error creating graph for {smiles}: {str(e)}")
        return None


def _create_graph_from_features(
    atom_features: np.ndarray, edge_index: List[List[int]], edge_attr: List[List[float]]
) -> Data:
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    if edge_index.shape[1] != edge_attr.shape[0]:
        raise ValueError(
            f"Edge index and edge attribute shapes do not match: {edge_index.shape[1]} != {edge_attr.shape[0]}"
        )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def process_smiles(smiles: str, scaler: StandardScaler) -> Optional[Data]:
    """Process a single SMILES string to a graph."""
    return mol_to_graph(smiles, scaler)


def clear_cache():
    """Clear the feature cache."""
    _feature_cache.clear()
