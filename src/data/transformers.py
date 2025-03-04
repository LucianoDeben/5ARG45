# data/transformers.py
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class Normalizer:
    """Normalizes data using L2 norm."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import Normalizer as SklearnNormalizer

        normalizer = SklearnNormalizer()
        return normalizer.fit_transform(x)


class StandardScaler:
    """Standardizes data to zero mean and unit variance."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler as SklearnScaler

        scaler = SklearnScaler()
        return scaler.fit_transform(x)


class BasicSmilesDescriptorTransform:
    """Generate simple descriptors from SMILES strings."""

    def __call__(self, mol_input: Dict) -> np.ndarray:
        smiles_list = mol_input["smiles"]
        dosage = np.array(mol_input["dosage"]).reshape(-1, 1)

        features = np.zeros((len(smiles_list), 10))
        for i, smiles in enumerate(smiles_list):
            if not isinstance(smiles, str):
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
    """Generate Morgan fingerprints from SMILES."""

    def __init__(self, radius: int = 2, size: int = 1024):
        self.radius = radius
        self.size = size

    def __call__(self, mol_input: Union[Dict, np.ndarray]) -> np.ndarray:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        if isinstance(mol_input, dict):
            smiles_list = mol_input["smiles"]
            dosage = np.array(mol_input["dosage"]).reshape(-1, 1)

            fingerprints = np.zeros((len(smiles_list), self.size))
            for i, smiles in enumerate(smiles_list):
                try:
                    if not isinstance(smiles, str):
                        logger.warning(f"Invalid SMILES: {type(smiles)}")
                        continue
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        # Generate Morgan fingerprint (ECFP)
                        fp = AllChem.GetMorganFingerprintAsBitVect(
                            mol, self.radius, nBits=self.size
                        )
                        # Convert to numpy array
                        fingerprints[i] = np.array(fp)
                    else:
                        logger.warning(f"Failed to parse SMILES: {smiles}")
                except Exception as e:
                    logger.warning(f"Error with SMILES {smiles}: {e}")

            return np.hstack([fingerprints, dosage]).astype(np.float32)
        return mol_input


class MolecularGraphTransform:
    """
    Generate NetworkX graph representations from SMILES strings.

    Supports conversion of SMILES to graph representation with
    configurable node and edge features.
    """

    def __init__(
        self,
        node_features: List[str] = ["atomic_num", "element"],
        edge_features: List[str] = ["order"],
    ):
        """
        Initialize MolecularGraphTransform.

        Args:
            node_features: List of node features to extract
            edge_features: List of edge features to extract
        """
        self.node_features = node_features
        self.edge_features = edge_features

    def __call__(self, mol_input: Union[Dict, List[str]]) -> List[nx.Graph]:
        """
        Convert SMILES to NetworkX graphs.

        Args:
            mol_input: Dictionary with 'smiles' key or list of SMILES strings

        Returns:
            List of NetworkX graph representations
        """
        import networkx as nx
        from rdkit import Chem

        # Handle different input types
        if isinstance(mol_input, dict):
            smiles_list = mol_input.get("smiles", [])
        elif isinstance(mol_input, list):
            smiles_list = mol_input
        else:
            raise ValueError(
                "Input must be a dictionary with 'smiles' key or a list of SMILES"
            )

        graphs = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Could not parse SMILES: {smiles}")
                    graphs.append(None)
                    continue

                # Create graph
                graph = nx.Graph()

                # Add nodes with features
                for atom in mol.GetAtoms():
                    node_data = {}
                    if "atomic_num" in self.node_features:
                        node_data["atomic_num"] = atom.GetAtomicNum()
                    if "element" in self.node_features:
                        node_data["element"] = atom.GetSymbol()

                    graph.add_node(atom.GetIdx(), **node_data)

                # Add edges with features
                for bond in mol.GetBonds():
                    start = bond.GetBeginAtomIdx()
                    end = bond.GetEndAtomIdx()

                    edge_data = {}
                    if "order" in self.edge_features:
                        edge_data["order"] = bond.GetBondTypeAsDouble()

                    graph.add_edge(start, end, **edge_data)

                graphs.append(graph)

            except Exception as e:
                logger.warning(f"Error processing SMILES {smiles}: {e}")
                graphs.append(None)

        return graphs


def create_transformations(
    transcriptomics_transform_type: Optional[str] = None,
    molecular_transform_type: Optional[str] = None,
    fingerprint_size: int = 2048,
    fingerprint_radius: int = 2,  # ECFP4
    **kwargs,
) -> Tuple[Optional[Callable], Optional[Callable]]:
    """
    Create data transformations.

    Args:
        transcriptomics_transform_type: Type of transformation for gene expression data
            ('normalize', 'scale', or None)
        molecular_transform_type: Type of transformation for molecular data
            ('fingerprint', 'descriptors', 'graph', or None)
        fingerprint_size: Number of bits for Morgan fingerprints
        fingerprint_radius: Radius for Morgan fingerprints

    Returns:
        Tuple of (transcriptomics_transform, molecular_transform)
    """
    # Create transcriptomics transformation
    transcriptomics_transform = None
    if transcriptomics_transform_type == "normalize":
        transcriptomics_transform = Normalizer()
    elif transcriptomics_transform_type == "scale":
        transcriptomics_transform = StandardScaler()

    # Create molecular transformation
    molecular_transform = None
    if molecular_transform_type == "fingerprint":
        try:
            molecular_transform = MorganFingerprintTransform(
                radius=fingerprint_radius, size=fingerprint_size
            )
        except ImportError:
            logger.warning("RDKit not available, falling back to basic descriptors")
            molecular_transform = BasicSmilesDescriptorTransform()
    elif molecular_transform_type == "descriptors":
        molecular_transform = BasicSmilesDescriptorTransform()
    elif molecular_transform_type == "graph":
        molecular_transform = MolecularGraphTransform()

    return transcriptomics_transform, molecular_transform
