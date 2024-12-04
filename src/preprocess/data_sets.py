import logging
import os
from typing import List, Optional

import pandas as pd
import torch
from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data as GeometricData
from torch_geometric.data import Dataset as GeometricDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChemicalDataset(GeometricDataset):

    def __init__(
        self,
        smiles_list: List[str],
        root: str = "./",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        representation: str = "graph",  # Optional: 'graph', 'fingerprint', etc.
    ):
        """
        Custom Dataset for chemical data using PyTorch Geometric's Data class.

        Args:
            root (str): Root directory where the dataset is stored.
            smiles_list (List[str]): List of SMILES strings.
            transform (callable, optional): Function to transform Data objects.
            pre_transform (callable, optional): Function to transform Data objects before saving.
            pre_filter (callable, optional): Function to filter Data objects.
            representation (str): Molecular representation type.
        """
        self.smiles_list = smiles_list
        self.representation = representation
        super().__init__(root, transform, pre_transform, pre_filter)
        self.processed_file_paths = [
            os.path.join(self.processed_dir, f"data_{i}.pt")
            for i in range(len(self.smiles_list))
        ]

    @property
    def raw_file_names(self):
        # No raw files since data is provided directly
        return []

    @property
    def processed_file_names(self):
        # List of processed files
        return [f"data_{i}.pt" for i in range(len(self.smiles_list))]

    def download(self):
        # No download needed as data is provided directly
        pass

    def process(self):
        # Process SMILES strings into Data objects and save them
        for idx, smiles in enumerate(self.smiles_list):
            try:
                data = self.smiles_to_graphs(smiles)
                if data is None:
                    logger.warning(f"Data is None for SMILES: {smiles}")
                    continue
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                torch.save(data, self.processed_paths[idx])
            except Exception as e:
                logger.error(f"Error processing SMILES {smiles}: {e}")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # Load the Data object from disk
        data = torch.load(self.processed_paths[idx], weights_only=False)
        if self.transform:
            data = self.transform(data)
        return data

    def smiles_to_graphs(self, smiles: str) -> Optional[GeometricData]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES string: {smiles}")
            return None

        featurizer = MolGraphConvFeaturizer()
        try:
            graph = featurizer.featurize([mol])
        except Exception as e:
            logger.error(f"Error featurizing molecule {smiles}: {e}")
            return None

        return graph


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
        self.features = torch.tensor(self.features_df, dtype=torch.float)
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
    # Test the ChemicalDataset class
    smiles_list = ["CCO", "CCN", "CC(=O)OC1=CC=CC=C1C(=O)O"]
    dataset = ChemicalDataset(smiles_list, root="data/processed/chemical_data/")

    print(dataset.smiles_list)

    # Print the items in the dataset
    for i in range(len(dataset)):
        print(dataset[i])
