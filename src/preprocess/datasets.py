import logging
import os
from typing import List, Optional, Tuple

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeometricDataset

from preprocess.molecule_graph import mol_to_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChemicalDataset(GeometricDataset):
    def __init__(
        self,
        root: str,
        smiles_list: List[str],
        targets: List[float],
        scale: bool = True,
        transform=None,
        pre_transform=None,
        save_data_list: bool = False,
    ):
        """
        Custom PyTorch Geometric Dataset for chemical graph data.

        Args:
            root (str): Root directory where the dataset should be saved.
            smiles_list (List[str]): List of SMILES strings.
            targets (List[float]): List of target values.
            scaler (Optional[StandardScaler]): Scaler for normalizing continuous features.
            transform (callable, optional): Function to transform each Data object after loading.
            pre_transform (callable, optional): Function to transform each Data object before saving.
        """
        self.smiles_list = smiles_list
        self.targets = targets
        self.scale = scale
        self.data_list = None
        self.save_data_list = save_data_list
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # Placeholder: Normally would list the raw files to check for dataset existence
        return ["raw_data.csv"]

    @property
    def processed_file_names(self):
        # Placeholder: Normally would list the processed files to check for dataset existence
        return ["data.pt"]

    def download(self):
        # Download raw data if necessary
        pass

    def process(self):
        """Processes the raw data into graph objects."""
        self.data_list = []
        for smiles, target in zip(self.smiles_list, self.targets):
            try:
                graph = mol_to_graph(smiles, self.scale)
                if graph is not None:
                    graph.y = torch.tensor([target], dtype=torch.float)
                    self.data_list.append(graph)
            except Exception as e:
                logger.error(f"Error processing SMILES {smiles}: {e}")

        if self.pre_transform:
            self.data_list = [self.pre_transform(data) for data in self.data_list]

        if self.save_data_list:
            torch.save(
                self.data_list,
                os.path.join(self.processed_dir, self.processed_file_names[0]),
            )

    def len(self):
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        """
        Gets the graph data object at the specified index.
        """
        if self.data_list is None:
            self.data_list = torch.load(
                os.path.join(self.processed_dir, self.processed_file_names[0])
            )
        return self.data_list[idx]


class TranscriptomicsDataset(Dataset):
    def __init__(
        self,
        transcriptomics_df: pd.DataFrame,
        targets: List[float],
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True,
    ):
        """
        Custom Dataset for transcriptomics data with optional standardization.

        Args:
            transcriptomics_df (pd.DataFrame): DataFrame with transcriptomics features.
            targets (List[float]): List of target values.
            scaler (StandardScaler, optional): Pre-fitted StandardScaler for standardization.
            fit_scaler (bool): Whether to fit the scaler on the provided data if no scaler is passed.
        """
        self.original_df = transcriptomics_df
        self.targets = targets
        self.scaler = scaler or StandardScaler()

        # Fit the scaler if necessary
        if fit_scaler and not scaler:
            self.scaler.fit(transcriptomics_df)

        # Standardize the features
        self.transcriptomics_tensor = torch.tensor(
            self.scaler.transform(transcriptomics_df), dtype=torch.float
        )
        self.targets_tensor = torch.tensor(targets, dtype=torch.float).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.transcriptomics_tensor)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a sample and its target value.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Standardized features and target value.
        """
        return self.transcriptomics_tensor[idx], self.targets_tensor[idx]

    def get_scaler(self) -> StandardScaler:
        """
        Retrieve the scaler used for standardization.

        Returns:
            StandardScaler: The scaler instance.
        """
        return self.scaler
