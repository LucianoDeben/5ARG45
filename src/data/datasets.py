# data/datasets.py
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
        """
        Initialize multimodal dataset for deep learning.

        Args:
            transcriptomics_data: Gene expression data
            metadata: DataFrame with SMILES, dosage, and viability data
            transform_transcriptomics: Transformation function for transcriptomics data
            transform_molecular: Transformation function for molecular data
        """
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
            # Use default basic encoding if no transformer provided
            from data.transformers import BasicSmilesDescriptorTransform

            default_transform = BasicSmilesDescriptorTransform()
            molecular = default_transform({"smiles": smiles, "dosage": dosage})

        # Convert to tensors
        result = {
            "transcriptomics": torch.tensor(transcriptomics, dtype=torch.float32),
            "molecular": torch.tensor(molecular, dtype=torch.float32),
            "viability": torch.tensor(viability, dtype=torch.float32),
        }

        # Ensure batch dimension for single items
        if isinstance(idx, int):
            for key in result:
                if not result[key].dim() or result[key].dim() == 1:
                    result[key] = result[key].unsqueeze(0)

        return result


class TranscriptomicsDataset:
    """Simple dataset for traditional ML models using only transcriptomics data."""

    def __init__(
        self,
        transcriptomics_data: np.ndarray,
        viability: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize transcriptomics dataset.

        Args:
            transcriptomics_data: Gene expression data
            viability: Target cell viability values
            transform: Transformation function for gene expression data
        """
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


class ChemicalDataset:
    """Dataset for traditional ML models using only chemical/molecular data."""

    def __init__(
        self,
        smiles: List[str],
        dosage: np.ndarray,
        viability: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize chemical dataset.

        Args:
            smiles: List of SMILES strings representing compounds
            dosage: Dosage values for each compound
            viability: Target cell viability values
            transform: Transformation function for molecular data
        """
        self.smiles = np.array(smiles)
        self.dosage = dosage.reshape(-1, 1) if len(dosage.shape) == 1 else dosage
        self.viability = viability
        self.transform = transform

        # Validate dimensions
        if len(smiles) != len(viability) or len(dosage) != len(viability):
            raise ValueError("Length mismatch between features and labels")

        logger.info(f"Initialized ChemicalDataset with {len(self)} samples")

    def __len__(self) -> int:
        return len(self.smiles)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get features and labels as numpy arrays."""
        if self.transform is not None:
            features = self.transform({"smiles": self.smiles, "dosage": self.dosage})
        else:
            # Use default fingerprint transform if no transformer provided
            from data.transformers import MorganFingerprintTransform

            default_transform = MorganFingerprintTransform(radius=2, size=1024)
            features = default_transform({"smiles": self.smiles, "dosage": self.dosage})

        return features, self.viability

    @classmethod
    def from_multimodal(
        cls, dataset: MultimodalDrugDataset, transform: Optional[Callable] = None
    ) -> "ChemicalDataset":
        """Create chemical dataset from multimodal dataset."""
        return cls(
            smiles=dataset.metadata["canonical_smiles"].values,
            dosage=dataset.metadata["pert_dose"].values,
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

    @staticmethod
    def create_chemical_dataset(
        gctx_loader,
        row_slice: Union[slice, list, None] = None,
        transform: Optional[Callable] = None,
    ) -> ChemicalDataset:
        """Create chemical dataset for traditional ML."""
        # Get data
        _, metadata, _ = gctx_loader.get_data_with_metadata(
            row_slice=row_slice,
            row_columns=None,
        )

        # Convert metadata if needed
        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.DataFrame(metadata)

        return ChemicalDataset(
            smiles=metadata["canonical_smiles"].values,
            dosage=metadata["pert_dose"].values,
            viability=metadata["viability"].values,
            transform=transform,
        )
