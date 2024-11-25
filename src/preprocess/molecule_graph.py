from typing import Dict, List, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType, HybridizationType
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# Cache for computed features
_feature_cache: Dict[str, Data] = {}


def one_hot_encode(value: str, allowable_values: List[str]) -> List[int]:
    """One-hot encode a value among allowable values."""
    return [int(value == v) for v in allowable_values]


def get_atom_features(atom: Chem.Atom, scaler: StandardScaler) -> List[float]:
    """Extract features from an RDKit Atom object."""
    # Continuous features
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

    # Categorical features
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

    return scaled_features + hybridization + aromatic + chirality + in_ring


def get_bond_features(bond: Chem.Bond) -> List[float]:
    """Extract bond features."""
    features = []

    # Bond type
    bond_types = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
    bond_type = one_hot_encode(str(bond.GetBondType()), [str(bt) for bt in bond_types])
    features.extend(bond_type)

    # Stereo configuration
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

    # Additional features
    features.append(int(bond.GetIsConjugated()))
    features.append(int(bond.IsInRing()))

    return features


def mol_to_graph(smiles: str, scaler: StandardScaler) -> Optional[Data]:
    """Convert SMILES to PyTorch Geometric Data object."""
    if smiles in _feature_cache:
        return _feature_cache[smiles]

    try:
        # Create molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            return None

        # Compute charges
        AllChem.ComputeGasteigerCharges(mol)

        # Get atom features
        atom_features = [get_atom_features(atom, scaler) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)

        # Get bond features
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_features = get_bond_features(bond)

            # Add edges in both directions
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([bond_features, bond_features])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Create graph
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        _feature_cache[smiles] = data
        return data

    except Exception as e:
        print(f"Error creating graph for {smiles}: {str(e)}")
        return None


def collect_continuous_features(smiles_list: List[str]) -> np.ndarray:
    """Collect continuous features for fitting the scaler."""
    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        mol = Chem.AddHs(mol)
        AllChem.ComputeGasteigerCharges(mol)

        for atom in mol.GetAtoms():
            partial_charge = (
                float(atom.GetProp("_GasteigerCharge"))
                if atom.HasProp("_GasteigerCharge")
                else 0.0
            )
            degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge()
            num_h = atom.GetTotalNumHs()
            features.append([partial_charge, degree, formal_charge, num_h])

    return np.array(features)


def clear_cache():
    """Clear the feature cache."""
    _feature_cache.clear()
