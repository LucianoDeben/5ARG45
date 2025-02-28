# src/data/datasets.py
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """
    Dataset for multimodal drug response prediction that supports both PyTorch and scikit-learn.

    This dataset holds transcriptomic data, drug information (SMILES and dosage),
    and cell viability labels. It supports various transformations for both
    transcriptomic and molecular data.

    It can be used in three modes:
    1. PyTorch mode: Returns dictionaries with tensors for deep learning models
    2. Scikit-learn mode: Returns numpy arrays for traditional ML models
    3. Unimodal mode: Returns only specific modalities for unimodal models

    Attributes:
        transcriptomics_data: Gene expression data matrix
        metadata: DataFrame with metadata (SMILES, dosage, viability, etc.)
        transform_transcriptomics: Optional transformation for transcriptomic data
        transform_molecular: Optional transformation for molecular data
        mode: Operating mode ('pytorch', 'sklearn', or 'unimodal')
        unimodal_type: Which modality to use in unimodal mode
        sklearn_return_X_y: Whether to return (X, y) or just X in sklearn mode
        additional_features: List of additional metadata columns to include
    """

    def __init__(
        self,
        transcriptomics_data: np.ndarray,
        metadata: pd.DataFrame,
        transform_transcriptomics: Optional[Callable] = None,
        transform_molecular: Optional[Callable] = None,
        mode: str = "pytorch",
        unimodal_type: Optional[str] = None,
        sklearn_return_X_y: bool = True,
        additional_features: Optional[List[str]] = None,
    ):
        """
        Initialize multimodal dataset.

        Args:
            transcriptomics_data: Gene expression data matrix
            metadata: DataFrame with metadata (must contain at least canonical_smiles, pert_dose, viability)
            transform_transcriptomics: Optional transformation for transcriptomic data
            transform_molecular: Optional transformation for molecular data
            mode: Operating mode ('pytorch', 'sklearn', or 'unimodal')
            unimodal_type: Which modality to use in unimodal mode ('transcriptomics' or 'molecular')
            sklearn_return_X_y: Whether to return (X, y) or just X in sklearn mode
            additional_features: List of additional metadata columns to include as features
        """
        self.transcriptomics_data = transcriptomics_data
        self.metadata = metadata
        self.transform_transcriptomics = transform_transcriptomics
        self.transform_molecular = transform_molecular
        self.mode = mode.lower()
        self.unimodal_type = unimodal_type.lower() if unimodal_type else None
        self.sklearn_return_X_y = sklearn_return_X_y
        self.additional_features = additional_features or []

        # Validate mode
        valid_modes = ["pytorch", "sklearn", "unimodal"]
        if self.mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got {self.mode}")

        # Validate unimodal_type
        if self.mode == "unimodal" and self.unimodal_type not in [
            "transcriptomics",
            "molecular",
        ]:
            raise ValueError(
                f"unimodal_type must be 'transcriptomics' or 'molecular', got {self.unimodal_type}"
            )

        # Check required metadata columns
        required_columns = ["canonical_smiles", "pert_dose", "viability"]
        missing_columns = [
            col for col in required_columns if col not in metadata.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required metadata columns: {missing_columns}")

        # Check additional features
        invalid_features = [
            col for col in self.additional_features if col not in metadata.columns
        ]
        if invalid_features:
            raise ValueError(
                f"Additional features not found in metadata: {invalid_features}"
            )

        # Check data dimensions
        if len(transcriptomics_data) != len(metadata):
            raise ValueError(
                f"Length mismatch: transcriptomics_data has {len(transcriptomics_data)} "
                f"rows, metadata has {len(metadata)} rows"
            )

        logger.info(
            f"Initialized MultimodalDataset with {len(self)} samples, "
            f"mode={self.mode}, unimodal_type={self.unimodal_type}"
        )

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.transcriptomics_data)

    def __getitem__(
        self, idx: Union[int, slice, List[int]]
    ) -> Union[Dict[str, torch.Tensor], Tuple[np.ndarray, np.ndarray]]:
        """
        Get item(s) from dataset.

        Args:
            idx: Index, slice, or list of indices

        Returns:
            - In PyTorch mode: Dictionary with tensors for each modality
            - In scikit-learn mode: (X, y) tuple of numpy arrays or just X
            - In unimodal mode: Dictionary with tensors for the specified modality
        """
        # Handle different index types
        if isinstance(idx, int):
            indices = [idx]
        elif isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
        elif isinstance(idx, list):
            indices = idx
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

        # Get data for the specified indices
        transcriptomics = self.transcriptomics_data[indices]
        smiles = self.metadata.iloc[indices]["canonical_smiles"].values
        dosage = self.metadata.iloc[indices]["pert_dose"].values
        viability = self.metadata.iloc[indices]["viability"].values

        # Get additional features if specified
        additional_data = {}
        if self.additional_features:
            for feature in self.additional_features:
                additional_data[feature] = self.metadata.iloc[indices][feature].values

        # Apply transformations
        if self.transform_transcriptomics is not None:
            transcriptomics = self.transform_transcriptomics(transcriptomics)

        # Create molecular input (SMILES + dosage)
        molecular_input = {"smiles": smiles, "dosage": dosage}
        if self.transform_molecular is not None:
            molecular_input = self.transform_molecular(molecular_input)

        # Return data based on mode
        if self.mode == "pytorch":
            # Convert to PyTorch tensors
            result = {
                "transcriptomics": torch.tensor(transcriptomics, dtype=torch.float32),
                "molecular": self._prepare_molecular_for_pytorch(molecular_input),
                "viability": torch.tensor(viability, dtype=torch.float32),
            }

            # Add additional features
            for feature, data in additional_data.items():
                result[feature] = torch.tensor(data)

            return result

        elif self.mode == "sklearn":
            # Combine features for scikit-learn
            if isinstance(molecular_input, dict):
                # If molecular input is still a dictionary, it wasn't transformed
                # We'll use a simple concatenation of SMILES embedding and dosage
                molecular_features = np.column_stack(
                    [
                        self._default_smiles_encoding(molecular_input["smiles"]),
                        molecular_input["dosage"].reshape(-1, 1),
                    ]
                )
            else:
                molecular_features = molecular_input

            # Combine features
            X = np.hstack([transcriptomics, molecular_features])

            # Add additional features
            for feature, data in additional_data.items():
                data_reshaped = data.reshape(-1, 1) if data.ndim == 1 else data
                X = np.hstack([X, data_reshaped])

            if self.sklearn_return_X_y:
                return X, viability
            else:
                return X

        elif self.mode == "unimodal":
            # Return only the specified modality
            if self.unimodal_type == "transcriptomics":
                if self.sklearn_return_X_y:
                    return transcriptomics, viability
                else:
                    return transcriptomics
            elif self.unimodal_type == "molecular":
                if isinstance(molecular_input, dict):
                    molecular_features = np.column_stack(
                        [
                            self._default_smiles_encoding(molecular_input["smiles"]),
                            molecular_input["dosage"].reshape(-1, 1),
                        ]
                    )
                else:
                    molecular_features = molecular_input

                if self.sklearn_return_X_y:
                    return molecular_features, viability
                else:
                    return molecular_features

    def _prepare_molecular_for_pytorch(
        self, molecular_input: Union[Dict, np.ndarray]
    ) -> torch.Tensor:
        """
        Prepare molecular input for PyTorch.

        Args:
            molecular_input: Dictionary with SMILES and dosage or transformed array

        Returns:
            PyTorch tensor with molecular features
        """
        if isinstance(molecular_input, dict):
            # If not transformed, use default encoding + dosage
            smiles_encoding = self._default_smiles_encoding(molecular_input["smiles"])
            dosage = molecular_input["dosage"].reshape(-1, 1)
            molecular_features = np.hstack([smiles_encoding, dosage])
            return torch.tensor(molecular_features, dtype=torch.float32)
        else:
            # If already transformed, just convert to tensor
            return torch.tensor(molecular_input, dtype=torch.float32)

    def _default_smiles_encoding(self, smiles_list: List[str]) -> np.ndarray:
        """
        Default encoding for SMILES strings when no transformation is provided.
        This is a placeholder that returns a simple one-hot encoding of characters.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Numpy array with simple encoding
        """
        # For demonstration, just count some common atoms/bonds as features
        # In a real implementation, you'd use a proper fingerprint or embedding
        features = np.zeros((len(smiles_list), 10))
        for i, smiles in enumerate(smiles_list):
            if not isinstance(smiles, str):
                continue
            features[i, 0] = smiles.count("C")  # Carbon count
            features[i, 1] = smiles.count("N")  # Nitrogen count
            features[i, 2] = smiles.count("O")  # Oxygen count
            features[i, 3] = (
                smiles.count("F")
                + smiles.count("Cl")
                + smiles.count("Br")
                + smiles.count("I")
            )  # Halogens
            features[i, 4] = smiles.count("=")  # Double bonds
            features[i, 5] = smiles.count("#")  # Triple bonds
            features[i, 6] = smiles.count("(") + smiles.count(")")  # Branches
            features[i, 7] = smiles.count("[") + smiles.count("]")  # Special atoms
            features[i, 8] = len(smiles)  # SMILES length
            features[i, 9] = (
                smiles.count("c") + smiles.count("n") + smiles.count("o")
            )  # Aromatic atoms
        return features

    def to_pytorch(self) -> "MultimodalDataset":
        """Convert dataset to PyTorch mode."""
        if self.mode == "pytorch":
            return self
        return MultimodalDataset(
            transcriptomics_data=self.transcriptomics_data,
            metadata=self.metadata,
            transform_transcriptomics=self.transform_transcriptomics,
            transform_molecular=self.transform_molecular,
            mode="pytorch",
            additional_features=self.additional_features,
        )

    def to_sklearn(self, return_X_y: bool = True) -> "MultimodalDataset":
        """Convert dataset to scikit-learn mode."""
        if self.mode == "sklearn" and self.sklearn_return_X_y == return_X_y:
            return self
        return MultimodalDataset(
            transcriptomics_data=self.transcriptomics_data,
            metadata=self.metadata,
            transform_transcriptomics=self.transform_transcriptomics,
            transform_molecular=self.transform_molecular,
            mode="sklearn",
            sklearn_return_X_y=return_X_y,
            additional_features=self.additional_features,
        )

    def to_unimodal(self, modality_type: str) -> "MultimodalDataset":
        """
        Convert dataset to unimodal mode.

        Args:
            modality_type: Which modality to use ('transcriptomics' or 'molecular')
        """
        if self.mode == "unimodal" and self.unimodal_type == modality_type:
            return self
        return MultimodalDataset(
            transcriptomics_data=self.transcriptomics_data,
            metadata=self.metadata,
            transform_transcriptomics=self.transform_transcriptomics,
            transform_molecular=self.transform_molecular,
            mode="unimodal",
            unimodal_type=modality_type,
            sklearn_return_X_y=self.sklearn_return_X_y,
            additional_features=self.additional_features,
        )

    def with_transforms(
        self,
        transcriptomics_transform: Optional[Callable] = None,
        molecular_transform: Optional[Callable] = None,
    ) -> "MultimodalDataset":
        """
        Create a new dataset with specified transforms.

        Args:
            transcriptomics_transform: Transformation for transcriptomic data
            molecular_transform: Transformation for molecular data

        Returns:
            New dataset with the specified transforms
        """
        return MultimodalDataset(
            transcriptomics_data=self.transcriptomics_data,
            metadata=self.metadata,
            transform_transcriptomics=transcriptomics_transform
            or self.transform_transcriptomics,
            transform_molecular=molecular_transform or self.transform_molecular,
            mode=self.mode,
            unimodal_type=self.unimodal_type,
            sklearn_return_X_y=self.sklearn_return_X_y,
            additional_features=self.additional_features,
        )

    def with_additional_features(self, features: List[str]) -> "MultimodalDataset":
        """
        Create a new dataset with additional metadata features.

        Args:
            features: List of additional metadata columns to include

        Returns:
            New dataset with additional features
        """
        # Validate features
        invalid_features = [col for col in features if col not in self.metadata.columns]
        if invalid_features:
            raise ValueError(f"Features not found in metadata: {invalid_features}")

        return MultimodalDataset(
            transcriptomics_data=self.transcriptomics_data,
            metadata=self.metadata,
            transform_transcriptomics=self.transform_transcriptomics,
            transform_molecular=self.transform_molecular,
            mode=self.mode,
            unimodal_type=self.unimodal_type,
            sklearn_return_X_y=self.sklearn_return_X_y,
            additional_features=features,
        )

    # For scikit-learn compatibility
    def fit(self, X=None, y=None):
        """Dummy fit method for scikit-learn compatibility."""
        return self

    def transform(self, X=None):
        """Transform method for scikit-learn compatibility."""
        if self.mode != "sklearn":
            sklearn_dataset = self.to_sklearn(return_X_y=False)
            return sklearn_dataset[:]
        return self[:]

    def fit_transform(self, X=None, y=None):
        """Fit and transform method for scikit-learn compatibility."""
        return self.transform(X)


class DatasetFactory:
    """Factory for creating multimodal datasets from GCTX data."""

    @staticmethod
    def from_gctx(
        gctx_loader,
        row_slice: Union[slice, list, None] = None,
        feature_space: Optional[Union[str, List[str]]] = "landmark",
        transform_transcriptomics: Optional[Callable] = None,
        transform_molecular: Optional[Callable] = None,
        mode: str = "pytorch",
        additional_features: Optional[List[str]] = None,
    ) -> MultimodalDataset:
        """
        Create a multimodal dataset from a GCTX data loader.

        Args:
            gctx_loader: GCTXDataLoader instance
            row_slice: Rows to use (slice or list of indices)
            feature_space: Gene feature space to use
            transform_transcriptomics: Optional transformation for transcriptomic data
            transform_molecular: Optional transformation for molecular data
            mode: Operating mode ('pytorch', 'sklearn', or 'unimodal')
            additional_features: List of additional metadata columns to include

        Returns:
            MultimodalDataset instance
        """
        # Get expression data and metadata
        expr_data, metadata, _ = gctx_loader.get_data_with_metadata(
            row_slice=row_slice,
            feature_space=feature_space,
            row_columns=None,  # Get all row metadata
        )

        # Convert metadata to DataFrame if it's not already
        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.DataFrame(metadata)

        # Create and return dataset
        return MultimodalDataset(
            transcriptomics_data=expr_data,
            metadata=metadata,
            transform_transcriptomics=transform_transcriptomics,
            transform_molecular=transform_molecular,
            mode=mode,
            additional_features=additional_features,
        )
