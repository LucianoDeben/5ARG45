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
from rdkit.DataStructs import FoldFingerprint
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


class RobustSmilesDescriptorTransform:
    """
    Generate simple descriptors from SMILES strings without requiring RDKit parsing.
    This is a fallback for when RDKit fingerprinting fails.
    """

    def __init__(self, output_dim: int = 64):
        """
        Initialize the robust SMILES descriptor transform.

        Args:
            output_dim: Dimension of the output representation
        """
        self.output_dim = output_dim
        self.logger = logging.getLogger(__name__)

    def validate_smiles(self, smiles: str) -> bool:
        """
        Perform basic validation on SMILES strings without using RDKit.

        Args:
            smiles: SMILES string to validate

        Returns:
            True if the SMILES string passes basic validation
        """
        if not isinstance(smiles, str):
            return False

        if len(smiles) < 2:  # Too short to be valid
            return False

        # Check for basic balancing of brackets
        if smiles.count("(") != smiles.count(")"):
            return False

        if smiles.count("[") != smiles.count("]"):
            return False

        # Check for common problematic patterns
        if "1PS" in smiles:  # From your error message
            return False

        return True

    def __call__(self, mol_input: Union[Dict, List[str]]) -> np.ndarray:
        """
        Convert SMILES to simple numerical descriptors without RDKit parsing.

        Args:
            mol_input: Dictionary with 'smiles' key or list of SMILES strings

        Returns:
            Array of numerical descriptors
        """
        smiles_list = mol_input["smiles"] if isinstance(mol_input, dict) else mol_input
        dosage = (
            np.array(mol_input["dosage"]).reshape(-1, 1)
            if isinstance(mol_input, dict) and "dosage" in mol_input
            else np.zeros((len(smiles_list), 1))
        )

        # Initialize output features
        features = np.zeros((len(smiles_list), self.output_dim - 1), dtype=np.float32)
        valid_mask = np.zeros(len(smiles_list), dtype=bool)

        for i, smiles in enumerate(smiles_list):
            # Basic validation
            if not self.validate_smiles(smiles):
                self.logger.warning(f"Invalid SMILES at index {i}: {smiles}")
                continue

            # Extract simple character-based features
            valid_mask[i] = True

            # Only compute features for valid SMILES
            try:
                # These are basic statistical features that don't require parsing
                features[i, 0] = len(smiles)  # Length
                features[i, 1] = smiles.count("C")  # Carbon count
                features[i, 2] = smiles.count("N")  # Nitrogen count
                features[i, 3] = smiles.count("O")  # Oxygen count
                features[i, 4] = (
                    smiles.count("F")
                    + smiles.count("Cl")
                    + smiles.count("Br")
                    + smiles.count("I")
                )  # Halogens
                features[i, 5] = smiles.count("S") + smiles.count("P")  # S and P count
                features[i, 6] = smiles.count("[")  # Special atoms
                features[i, 7] = smiles.count("(")  # Branching
                features[i, 8] = smiles.count("=")  # Double bonds
                features[i, 9] = smiles.count("#")  # Triple bonds
                features[i, 10] = smiles.count("@")  # Chirality

                # Ring counts
                for j in range(1, 10):
                    idx = 10 + j
                    if idx < self.output_dim - 1:
                        features[i, idx] = smiles.count(str(j))

                # Aromatic characters
                aromatic_count = sum(smiles.count(c) for c in ["c", "n", "o", "s", "p"])
                if 10 + 10 < self.output_dim - 1:
                    features[i, 10 + 10] = aromatic_count

                # Fill the rest with additional features or zeros
                for j in range(10 + 11, self.output_dim - 1):
                    if j < self.output_dim - 1:
                        features[i, j] = 0.0

            except Exception as e:
                self.logger.warning(
                    f"Error extracting features for SMILES {smiles}: {e}"
                )
                valid_mask[i] = False

        # Return only valid data
        if np.any(valid_mask):
            result = np.hstack([features[valid_mask], dosage[valid_mask]])
            self.logger.info(
                f"Processed {np.sum(valid_mask)}/{len(smiles_list)} valid SMILES strings"
            )
            return result
        else:
            self.logger.error("No valid SMILES found in batch")
            # Return at least one row of zeros to prevent shape errors
            return np.zeros((1, self.output_dim), dtype=np.float32)


# class MorganFingerprintTransform:
#     fpgen = None

#     def __init__(self, fingerprint_size=1024, fingerprint_radius=2):
#         self.fingerprint_size = fingerprint_size
#         self.fingerprint_radius = fingerprint_radius
#         # Default Morgan fingerprint size is 2048
#         self.default_size = 2048
#         # Calculate fold factor (must be a power of 2 reduction)
#         if (
#             self.default_size % self.fingerprint_size != 0
#             or self.fingerprint_size > self.default_size
#         ):
#             raise ValueError(
#                 f"fingerprint_size ({self.fingerprint_size}) must be a divisor of 2048 (e.g., 2048, 1024, 512)"
#             )
#         self.fold_factor = self.default_size // self.fingerprint_size

#     def __call__(self, mol_input):
#         if MorganFingerprintTransform.fpgen is None:
#             MorganFingerprintTransform.fpgen = AllChem.GetMorganGenerator(
#                 radius=self.fingerprint_radius
#             )
#         smiles_list = mol_input["smiles"]
#         mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
#         fps = []
#         for mol in mols:
#             if mol is None:
#                 fp = [0] * self.fingerprint_size
#             else:
#                 full_fp = MorganFingerprintTransform.fpgen.GetFingerprint(
#                     mol
#                 )  # 2048 bits
#                 folded_fp = FoldFingerprint(full_fp, foldFactor=self.fold_factor)
#                 fp_list = list(folded_fp)
#                 assert len(fp_list) == self.fingerprint_size
#             fps.append(fp_list)
#         return np.array(fps)


class MorganFingerprintTransform:
    """Generate Morgan fingerprints from SMILES."""

    def __init__(self, radius: int = 2, size: int = 1024):
        self.radius = radius
        self.size = size

    def __call__(self, mol_input: Union[Dict, List[str]]) -> np.ndarray:
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

                # Generate fingerprint
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.radius, nBits=self.size
                )
                fingerprints[i] = np.array(fp)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(
                    f"Error processing SMILES at index {i} ({smiles}): {str(e)}"
                )
                continue

        if not valid_indices:
            logger.error("No valid SMILES found in batch, returning empty arrays")
            return np.zeros((0, self.size + 1), dtype=np.float32)

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
