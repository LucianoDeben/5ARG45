# src/data/datasets.py
"""Dataset implementations for drug response prediction."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MultimodalDrugDataset(Dataset):
    """Dataset for multimodal deep learning models."""

    def __init__(
        self,
        transcriptomics_data: np.ndarray,
        metadata: pd.DataFrame,
        transform_transcriptomics: Optional[Callable] = None,
        transform_molecular: Optional[Callable] = None,
    ):
        """Initialize multimodal dataset for deep learning."""
        self.transcriptomics_data = transcriptomics_data
        self.metadata = metadata
        self.transform_transcriptomics = transform_transcriptomics
        self.transform_molecular = transform_molecular

        # Validate metadata
        required_columns = ["canonical_smiles", "pert_dose", "viability"]
        missing_columns = [
            col for col in required_columns if col not in metadata.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required metadata columns: {missing_columns}")

        # Validate dimensions
        if len(transcriptomics_data) != len(metadata):
            raise ValueError(
                f"Length mismatch: transcriptomics_data has {len(transcriptomics_data)} "
                f"rows, metadata has {len(metadata)} rows"
            )

        logger.info(f"Initialized MultimodalDrugDataset with {len(self)} samples")

    def __len__(self) -> int:
        return len(self.transcriptomics_data)

    def __getitem__(self, idx: Union[int, slice, List[int]]) -> Dict[str, torch.Tensor]:
        """Get sample(s) as PyTorch tensors."""
        # Handle different index types
        if isinstance(idx, int):
            indices = [idx]
        elif isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
        elif isinstance(idx, list):
            indices = idx
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

        # Get data
        transcriptomics = self.transcriptomics_data[indices]
        smiles = self.metadata.iloc[indices]["canonical_smiles"].values
        dosage = self.metadata.iloc[indices]["pert_dose"].values.reshape(-1, 1)
        viability = self.metadata.iloc[indices]["viability"].values

        # Apply transformations
        if self.transform_transcriptomics is not None:
            transcriptomics = self.transform_transcriptomics(transcriptomics)

        # Process molecular data
        if self.transform_molecular is not None:
            molecular = self.transform_molecular({"smiles": smiles, "dosage": dosage})
        else:
            molecular = np.hstack([self._default_smiles_encoding(smiles), dosage])

        # Convert to tensors
        return {
            "transcriptomics": torch.tensor(transcriptomics, dtype=torch.float32),
            "molecular": torch.tensor(molecular, dtype=torch.float32),
            "viability": torch.tensor(viability, dtype=torch.float32),
        }

    def _default_smiles_encoding(self, smiles_list: List[str]) -> np.ndarray:
        """Simple SMILES encoding when no transformation is provided."""
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
        return features


class TranscriptomicsDataset:
    """Simple dataset for traditional ML models using only transcriptomics data."""

    def __init__(
        self,
        transcriptomics_data: np.ndarray,
        viability: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        """Initialize transcriptomics dataset."""
        self.transcriptomics = transcriptomics_data
        self.viability = viability
        self.transform = transform

        if len(transcriptomics_data) != len(viability):
            raise ValueError("Length mismatch between features and labels")

        logger.info(f"Initialized TranscriptomicsDataset with {len(self)} samples")

    def __len__(self) -> int:
        return len(self.transcriptomics)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get features and labels as numpy arrays."""
        X = self.transcriptomics.copy()
        if self.transform is not None:
            X = self.transform(X)
        return X, self.viability

    @classmethod
    def from_multimodal(
        cls, dataset: MultimodalDrugDataset, transform: Optional[Callable] = None
    ) -> "TranscriptomicsDataset":
        """Create transcriptomics dataset from multimodal dataset."""
        return cls(
            transcriptomics_data=dataset.transcriptomics_data,
            viability=dataset.metadata["viability"].values,
            transform=transform,
        )


class DatasetFactory:
    """Factory for creating datasets from GCTX data."""

    @staticmethod
    def create_multimodal_dataset(
        gctx_loader,
        row_slice: Union[slice, list, None] = None,
        feature_space: Optional[Union[str, List[str]]] = "landmark",
        transform_transcriptomics: Optional[Callable] = None,
        transform_molecular: Optional[Callable] = None,
    ) -> MultimodalDrugDataset:
        """Create multimodal dataset for deep learning."""
        # Get data
        expr_data, metadata, _ = gctx_loader.get_data_with_metadata(
            row_slice=row_slice,
            feature_space=feature_space,
            row_columns=None,
        )

        # Convert metadata if needed
        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.DataFrame(metadata)

        return MultimodalDrugDataset(
            transcriptomics_data=expr_data,
            metadata=metadata,
            transform_transcriptomics=transform_transcriptomics,
            transform_molecular=transform_molecular,
        )

    @staticmethod
    def create_transcriptomics_dataset(
        gctx_loader,
        row_slice: Union[slice, list, None] = None,
        feature_space: Optional[Union[str, List[str]]] = "landmark",
        transform: Optional[Callable] = None,
    ) -> TranscriptomicsDataset:
        """Create transcriptomics dataset for traditional ML."""
        # Get data
        expr_data, metadata, _ = gctx_loader.get_data_with_metadata(
            row_slice=row_slice,
            feature_space=feature_space,
            row_columns=None,
        )

        # Extract viability values
        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.DataFrame(metadata)
        viability = metadata["viability"].values

        return TranscriptomicsDataset(
            transcriptomics_data=expr_data, viability=viability, transform=transform
        )
