# src/data/feature_transforms.py
"""
Feature extraction transformations for converting raw data into model-ready representations.
"""
import logging
from typing import Callable, Dict, List, Optional, Union

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class BasicSmilesDescriptorTransform:
    """Generate simple descriptors from SMILES strings."""

    def __call__(self, mol_input: Union[Dict, List[str]]) -> np.ndarray:
        """Convert SMILES to basic numerical descriptors."""
        smiles_list = mol_input["smiles"] if isinstance(mol_input, dict) else mol_input
        dosage = (
            np.array(mol_input["dosage"]).reshape(-1, 1)
            if isinstance(mol_input, dict)
            else np.zeros((len(smiles_list), 1))
        )

        features = np.zeros((len(smiles_list), 10), dtype=np.float32)
        for i, smiles in enumerate(smiles_list):
            if not isinstance(smiles, str):
                logger.warning(f"Skipping invalid SMILES at index {i}: {smiles}")
                continue
            features[i, 0] = smiles.count("C")  # Carbon count
            features[i, 1] = smiles.count("N")  # Nitrogen count
            features[i, 2] = smiles.count("O")  # Oxygen count
            features[i, 3] = sum(
                smiles.count(x) for x in ["F", "Cl", "Br", "I"]
            )  # Halogens
            features[i, 4] = smiles.count("=")  # Double bonds
            features[i, 5] = smiles.count("#")  # Triple bonds
            features[i, 6] = smiles.count("(") + smiles.count(")")  # Branches
            features[i, 7] = smiles.count("[") + smiles.count("]")  # Special atoms
            features[i, 8] = len(smiles)  # Length
            features[i, 9] = sum(smiles.count(x) for x in ["c", "n", "o"])  # Aromatic

        return np.hstack([features, dosage])

class MorganFingerprintTransform:
    """Generate Morgan/ECFP fingerprints from SMILES using recommended RDKit API."""

    def __init__(self, radius: int = 2, size: int = 1024):
        """
        Initialize the Morgan fingerprint transform.
        
        Args:
            radius: The radius of the Morgan fingerprint (radius 2 = ECFP4)
            size: The number of bits in the fingerprint
        """
        self.radius = radius
        self.size = size
        # Create the fingerprint generator once at initialization
        self.fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=size)
        logger.info(f"Initialized Morgan fingerprint generator with radius={radius}, size={size}")

    def __call__(self, mol_input: Union[Dict, List[str]]) -> np.ndarray:
        """
        Generate Morgan fingerprints for a batch of SMILES strings.
        
        Args:
            mol_input: Either a dictionary with 'smiles' key, or a list of SMILES strings.
                       If a dictionary with 'dosage' key is provided, dosage is appended to features.
        
        Returns:
            NumPy array of fingerprints with shape (n_valid_smiles, self.size + n_dosage_features)
        """
        smiles_list = mol_input["smiles"] if isinstance(mol_input, dict) else mol_input
        dosage = (
            np.array(mol_input["dosage"]).reshape(-1, 1)
            if isinstance(mol_input, dict) and "dosage" in mol_input
            else np.zeros((len(smiles_list), 1))
        )

        fingerprints = np.zeros((len(smiles_list), self.size), dtype=np.float32)
        valid_indices = []

        for i, smiles in enumerate(smiles_list):
            # Skip invalid SMILES strings
            if not isinstance(smiles, str) or not smiles.strip():
                logger.warning(f"Invalid SMILES at index {i}: {smiles}")
                continue

            try:
                # Try to parse the SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Failed to parse SMILES at index {i}: {smiles}")
                    continue

                # Generate fingerprint using the recommended approach
                fp = self.fpgen.GetFingerprint(mol)
                
                # Convert the ExplicitBitVect to numpy array
                fingerprints[i] = np.array(fp)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(
                    f"Error processing SMILES at index {i} ({smiles}): {str(e)}"
                )
                continue
            
        if not valid_indices:
            logger.error("No valid SMILES found in batch, returning zeros")
            return np.zeros((1, self.size + 1), dtype=np.float32) 

        valid_indices = np.array(valid_indices)
        return np.hstack([fingerprints[valid_indices], dosage[valid_indices]])

class MolecularGraphTransform:
    """Generate tensor-ready graph representations from SMILES for GNNs."""

    def __init__(
        self,
        node_features: List[str] = ["atomic_num"],
        edge_features: List[str] = ["order"],
    ):
        """Initialize with graph feature specifications."""
        self.node_features = node_features
        self.edge_features = edge_features

    def __call__(self, mol_input: Union[Dict, List[str]]) -> List[Data]:
        """Convert SMILES to PyTorch Geometric Data objects."""
        smiles_list = mol_input["smiles"] if isinstance(mol_input, dict) else mol_input
        graphs = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    logger.warning(f"Could not parse SMILES: {smiles}")
                    graphs.append(None)
                    continue

                # Node features
                node_feats = []
                for atom in mol.GetAtoms():
                    feat = []
                    if "atomic_num" in self.node_features:
                        feat.append(atom.GetAtomicNum())
                    if "element" in self.node_features:
                        feat.append(hash(atom.GetSymbol()))  # Simplified encoding
                    node_feats.append(feat or [0])
                x = torch.tensor(node_feats, dtype=torch.float)

                # Edge indices and features
                edge_index = []
                edge_attr = []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    edge_index.append([start, end])
                    edge_index.append([end, start])  # Undirected
                    feat = []
                    if "order" in self.edge_features:
                        feat.append(bond.GetBondTypeAsDouble())
                    edge_attr.extend([feat or [0]] * 2)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)

                graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
            except Exception as e:
                logger.warning(f"Error processing SMILES {smiles}: {e}")
                graphs.append(None)
        return graphs


def create_feature_transform(
    transform_type: str,
    fingerprint_size: int = 2048,
    fingerprint_radius: int = 2,
    **kwargs,
) -> Callable:
    """
    Factory function to create feature extraction transformations.

    Args:
        transform_type: Type of transformation ('fingerprint', 'descriptors', 'graph')
        fingerprint_size: Number of bits for Morgan fingerprints
        fingerprint_radius: Radius for Morgan fingerprints
        **kwargs: Additional arguments (e.g., graph_args for MolecularGraphTransform)

    Returns:
        A feature transformation object
    """
    valid_types = {"fingerprint", "descriptors", "graph"}
    if transform_type not in valid_types:
        raise ValueError(
            f"Invalid feature transform type: {transform_type}. Allowed: {valid_types}"
        )

    transform_map = {
        "fingerprint": MorganFingerprintTransform(
            radius=fingerprint_radius, size=fingerprint_size
        ),
        "descriptors": BasicSmilesDescriptorTransform(),
        "graph": MolecularGraphTransform(**kwargs.get("graph_args", {})),
    }
    return transform_map[transform_type]
