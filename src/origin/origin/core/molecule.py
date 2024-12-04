import logging
from typing import Any, Dict, Optional

import torch
from deepchem.feat import Featurizer
from origin.core.entity import Entity
from rdkit import Chem
from torch_geometric.data import Data as GeometricData

logger = logging.getLogger(__name__)


class Molecule(Entity):
    def __init__(self, smiles: str):
        """
        Represents a molecule with multiple potential representations.

        Args:
            smiles (str): SMILES representation of the molecule.
        """
        self.smiles = smiles
        self._mol = None  # RDKit molecule object
        self._features: Dict[str, Any] = {}  # Cache for molecule representations

    @property
    def mol(self) -> Chem.Mol:
        """
        Lazily computes and returns the RDKit molecule object.
        """
        if self._mol is None:
            self._mol = Chem.MolFromSmiles(self.smiles)
            if self._mol is None:
                logger.warning(f"Failed to parse SMILES string: {self.smiles}")
        return self._mol

    def featurize(self, featurizer: Featurizer, representation: str) -> None:
        """
        Featurizes the molecule using the specified featurizer and stores the result.

        Args:
            featurizer (Featurizer): Featurizer object to compute features.
            representation (str): Name of the representation (e.g., "graph", "fingerprint").
        """
        if representation in self._features:
            logger.info(
                f"Representation '{representation}' already computed. Skipping."
            )
            return

        try:
            features = featurizer.featurize([self.mol])
            if features is not None and len(features) > 0:
                self._features[representation] = features[0]
                logger.info(f"Successfully computed '{representation}' representation.")
            else:
                logger.warning(
                    f"Failed to compute '{representation}' representation for molecule."
                )
        except Exception as e:
            logger.error(
                f"Error featurizing molecule for representation '{representation}': {e}"
            )

    def get_features(self, representation: str) -> Optional[Any]:
        """
        Retrieves the specified representation from the cache.

        Args:
            representation (str): Name of the representation to retrieve.

        Returns:
            The representation object, or None if not available.
        """
        return self._features.get(representation, None)

    @property
    def graph(self) -> Optional[GeometricData]:
        """
        Retrieves the molecular graph representation if already computed.

        Raises:
            ValueError: If the representation is not computed.

        Returns:
            Optional[GeometricData]: The graph representation.
        """
        if "graph" not in self._features:
            raise ValueError(
                "Graph representation not computed yet. "
                "Use `featurize()` with an appropriate featurizer to compute it."
            )
        return self._features.get("graph")

    @property
    def fingerprint(self) -> Optional[torch.Tensor]:
        """
        Retrieves the molecular fingerprint representation if already computed.

        Raises:
            ValueError: If the representation is not computed.

        Returns:
            Optional[torch.Tensor]: The fingerprint representation.
        """
        if "fingerprint" not in self._features:
            raise ValueError(
                "Fingerprint representation not computed yet. "
                "Use `featurize()` with an appropriate featurizer to compute it."
            )
        return self._features.get("fingerprint")

    @property
    def embedding(self) -> Optional[torch.Tensor]:
        """
        Retrieves the molecular embedding representation if already computed.

        Raises:
            ValueError: If the representation is not computed.

        Returns:
            Optional[torch.Tensor]: The embedding representation.
        """
        if "embedding" not in self._features:
            raise ValueError(
                "Embedding representation not computed yet. "
                "Use `featurize()` with an appropriate featurizer to compute it."
            )
        return self._features.get("embedding")

    def clear_cache(self) -> None:
        """
        Clears all cached molecular representations.
        """
        self._features.clear()

    def __repr__(self) -> str:
        return f"Molecule(smiles={self.smiles})"
