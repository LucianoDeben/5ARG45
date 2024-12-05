import logging
import os
from typing import Callable, List, Optional

import pandas as pd
import torch
from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data as GeometricData
from torch_geometric.data import Dataset as GeometricDataset

from origin.core.molecule import Molecule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoleculeDataset(Dataset):
    def __init__(
        self,
        smiles_list: Optional[List[str]] = None,
        molecule_list: Optional[List[Molecule]] = None,
    ):
        """
        A flexible dataset for managing molecules and their representations.

        Args:
            smiles_list (Optional[List[str]]): List of SMILES strings.
            molecule_list (Optional[List[Molecule]]): List of precomputed Molecule objects.
        """
        if molecule_list:
            self.molecules = molecule_list
        elif smiles_list:
            self.smiles_list = smiles_list
            self.molecules = [Molecule(smiles) for smiles in smiles_list]
        else:
            raise ValueError(
                "Provide at least one of smiles_list, mol_list, or molecule_list."
            )

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> Molecule:
        return self.molecules[idx]

    def add_graphs(self, featurizer: MolGraphConvFeaturizer) -> None:
        """
        Add a new graph representation for all molecules in the dataset.

        Args:
            featurizer (Callable): Function to compute the graph representation.
        """
        self.graphs_list = featurizer.featurize(self.smiles_list)

    # CRUD operations
    def add_representation(self, representation: str, featurizer: Callable) -> None:
        """
        Add a new representation for all molecules in the dataset.

        Args:
            representation (str): Name of the representation.
            featurizer (Callable): Function to compute the representation.
        """
        for molecule in self.molecules:
            molecule.featurize(featurizer, representation)

    def get_representation(self, representation: str) -> List[Optional[torch.Tensor]]:
        """
        Retrieve a specific representation for all molecules.

        Args:
            representation (str): Name of the representation to retrieve.

        Returns:
            List[Optional[torch.Tensor]]: Representation for each molecule.
        """
        return [molecule.get_features(representation) for molecule in self.molecules]

    def update_representation(self, representation: str, featurizer: Callable) -> None:
        """
        Update an existing representation for all molecules.

        Args:
            representation (str): Name of the representation.
            featurizer (Callable): Function to compute the updated representation.
        """
        for molecule in self.molecules:
            molecule.featurize(featurizer, representation)

    def delete_representation(self, representation: str) -> None:
        """
        Delete a specific representation for all molecules.

        Args:
            representation (str): Name of the representation to delete.
        """
        for molecule in self.molecules:
            if representation in molecule._features:
                del molecule._features[representation]

    def list_representations(self) -> List[str]:
        """
        List all available representations across all molecules.

        Returns:
            List[str]: Names of available representations.
        """
        return list(
            set().union(*(molecule._features.keys() for molecule in self.molecules))
        )


class TranscriptomicsDataset(Dataset):
    def __init__(
        self,
        features_df: pd.DataFrame,
        targets: List[float],
    ):
        """
        Dataset for transcriptomics data.

        Args:
            features_df (pd.DataFrame): DataFrame of features.
            targets (List[float]): Target values.
            scaler (StandardScaler, optional): Scaler for feature normalization.
            fit_scaler (bool): Whether to fit the scaler on the data.
        """
        self.features_df = features_df
        self.targets = targets
        self.features = torch.tensor(self.features_df.values, dtype=torch.float)
        self.targets_tensor = torch.tensor(targets.values, dtype=torch.float).unsqueeze(
            1
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets_tensor[idx]


class MultimodalDataset(Dataset):
    def __init__(self, chem_dataset: Dataset, trans_dataset: Dataset):
        assert len(chem_dataset) == len(trans_dataset)
        self.chem_dataset = chem_dataset
        self.trans_dataset = trans_dataset

    def __len__(self):
        return len(self.chem_dataset)

    def __getitem__(self, idx):
        chem_data = self.chem_dataset[idx]
        trans_data, target = self.trans_dataset[idx]
        return chem_data, trans_data, target


if __name__ == "__main__":
    # Test the MoleculeDataset class
    smiles_list = ["CCO", "CCN", "CC(=O)OC1=CC=CC=C1C(=O)O"]
    dataset = MoleculeDataset(smiles_list=smiles_list)
    dataset.add_graphs(MolGraphConvFeaturizer())

    # Print the items in the dataset
    print(dataset.graphs_list)
    print(type(dataset.graphs_list[0]))
    print(type(dataset.graphs_list[0].x))
