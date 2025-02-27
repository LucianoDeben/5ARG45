import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """
    A PyTorch Dataset for multimodal drug response prediction, integrating transcriptomic data
    and chemical descriptors from LINCS/CTRP data.

    Attributes:
        transcriptomics (pd.DataFrame): Gene expression data.
        chemicals (pd.DataFrame): Chemical descriptors derived from SMILES.
        targets (np.ndarray): Target values (e.g., cell viability).
        metadata (pd.DataFrame, optional): Additional metadata (e.g., cell lines).
    """

    def __init__(
        self,
        transcriptomics: pd.DataFrame,
        row_metadata: pd.DataFrame,
        smiles_column: str = "canonical_smiles",
        target_column: str = "viability",
        metadata_columns: Optional[List[str]] = None,
        transforms: Optional[Dict[str, Callable]] = None,
        chemical_descriptor_fn: Optional[Callable] = None,
    ):
        """
        Initialize the MultimodalDataset with preloaded data.

        Args:
            transcriptomics: Preloaded gene expression DataFrame.
            row_metadata: Preloaded row metadata DataFrame.
            smiles_column: Column name in row_metadata for SMILES strings.
            target_column: Column name in row_metadata for target values.
            metadata_columns: Optional list of additional metadata columns.
            transforms: Optional dictionary of callable transforms for data preprocessing.
            chemical_descriptor_fn: Optional function to compute chemical descriptors.
        """
        self.transcriptomics = transcriptomics
        self.row_metadata = row_metadata
        self.smiles_column = smiles_column
        self.target_column = target_column
        self.metadata_columns = metadata_columns or []
        self.transforms = transforms or {}
        self.chemical_descriptor_fn = (
            chemical_descriptor_fn or self._default_chemical_descriptor
        )

        # Validate and load data
        self._load_data()

    def _default_chemical_descriptor(self, smiles: str) -> np.ndarray:
        """Compute Morgan fingerprints (1024 bits, radius=2) from SMILES strings."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        return np.array(fp, dtype=np.float32)

    def _load_data(self):
        """Validate and process preloaded data."""
        for col in [self.smiles_column, self.target_column] + self.metadata_columns:
            if col not in self.row_metadata.columns:
                raise KeyError(f"Column '{col}' not found in row metadata.")

        self.smiles = self.row_metadata[self.smiles_column].values
        self.targets = self.row_metadata[self.target_column].values.astype(np.float32)
        self.metadata = (
            self.row_metadata[self.metadata_columns] if self.metadata_columns else None
        )
        self.chemicals = self._compute_chemical_descriptors()
        logger.debug(
            f"Loaded dataset with {len(self.transcriptomics)} samples, "
            f"{len(self.transcriptomics.columns)} genes, {len(self.chemicals.columns)} chemical features"
        )

    def _compute_chemical_descriptors(self) -> pd.DataFrame:
        """Compute chemical descriptors for all SMILES strings."""
        descriptors = [self.chemical_descriptor_fn(smiles) for smiles in self.smiles]
        return pd.DataFrame(descriptors, index=self.transcriptomics.index)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.transcriptomics)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary with 'transcriptomics', 'chemicals', 'target', and optionally 'metadata'.
        """
        transcriptomic_data = self.transcriptomics.iloc[idx].values
        chemical_data = self.chemicals.iloc[idx].values
        target = self.targets[idx]

        if "transcriptomics" in self.transforms:
            transcriptomic_data = self.transforms["transcriptomics"](
                transcriptomic_data
            )
        if "chemicals" in self.transforms:
            chemical_data = self.transforms["chemicals"](chemical_data)

        sample = {
            "transcriptomics": torch.tensor(transcriptomic_data, dtype=torch.float32),
            "chemicals": torch.tensor(chemical_data, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
        }

        if self.metadata is not None:
            sample["metadata"] = self.metadata.iloc[idx].to_dict()

        return sample

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return features and targets as NumPy arrays for scikit-learn."""
        X = np.hstack((self.transcriptomics.values, self.chemicals.values))
        y = self.targets
        return X, y

    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return features and targets as Pandas objects."""
        features = pd.concat([self.transcriptomics, self.chemicals], axis=1)
        targets = pd.Series(self.targets, index=self.transcriptomics.index)
        return features, targets
