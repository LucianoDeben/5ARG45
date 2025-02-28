# src/data/preprocessing.py
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader

from data.datasets import MultimodalDataset
from data.loaders import GCTXDataLoader

logger = logging.getLogger(__name__)


class LINCSCTRPDataProcessor:
    """
    Processor for LINCS/CTRP data that handles loading, preprocessing, and splitting.

    This class manages the workflow from raw GCTX data to ready-to-use datasets and dataloaders,
    supporting both random and group-based splitting strategies to simulate different
    patient cohorts.
    """

    def __init__(
        self,
        gctx_file: str,
        feature_space: Union[str, List[str]] = "landmark",
        nrows: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        group_by: Optional[str] = None,
        batch_size: int = 32,
        transform_transcriptomics: Optional[Callable] = None,
        transform_molecular: Optional[Callable] = None,
        additional_features: Optional[List[str]] = None,
    ):
        """
        Initialize the LINCSCTRPDataProcessor for loading and splitting LINCS/CTRP data.

        Args:
            gctx_file: Path to the GCTX file.
            feature_space: Genes to load (e.g., 'landmark', ['landmark', 'best inferred']).
            nrows: Number of rows to load (None for all rows, useful for debugging).
            test_size: Proportion of data for the test set (0 to 1).
            val_size: Proportion of data for the validation set (0 to 1).
            random_state: Seed for reproducible splitting.
            group_by: Metadata column for grouped splitting (e.g., 'cell_mfc_name').
            batch_size: Batch size for DataLoader objects.
            transform_transcriptomics: Optional transformation for transcriptomic data.
            transform_molecular: Optional transformation for molecular data.
            additional_features: List of additional metadata columns to include.
        """
        self.gctx_file = gctx_file
        self.feature_space = feature_space
        self.nrows = nrows
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.group_by = group_by
        self.batch_size = batch_size
        self.transform_transcriptomics = transform_transcriptomics
        self.transform_molecular = transform_molecular
        self.additional_features = additional_features or []

        # Add group_by to additional_features if it's not already there
        if self.group_by and self.group_by not in self.additional_features:
            self.additional_features.append(self.group_by)

        # Set random seed for reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    def process(self) -> Tuple[MultimodalDataset, MultimodalDataset, MultimodalDataset]:
        """
        Process the GCTX data and create train, validation, and test datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info(
            f"Processing LINCS/CTRP data with feature_space={self.feature_space}, "
            f"nrows={self.nrows}, group_by={self.group_by}"
        )

        try:
            # Load data from GCTX file
            with GCTXDataLoader(self.gctx_file) as loader:
                row_slice = slice(0, self.nrows) if self.nrows is not None else None

                # Get expression data, row metadata, and column metadata
                logger.info(f"Loading data with feature_space={self.feature_space}")
                transcriptomics, row_metadata, _ = loader.get_data_with_metadata(
                    row_slice=row_slice,
                    feature_space=self.feature_space,
                )

                # Validate required columns
                required_cols = ["canonical_smiles", "pert_dose", "viability"]
                if self.group_by:
                    required_cols.append(self.group_by)

                for col in required_cols:
                    if col not in row_metadata.columns:
                        raise ValueError(
                            f"Missing required column in row metadata: {col}"
                        )

                # Split data into train, val, and test sets
                logger.info(
                    f"Splitting data with {'group' if self.group_by else 'random'} strategy"
                )
                if self.group_by:
                    train_data, val_data, test_data = self._group_split(
                        transcriptomics, row_metadata
                    )
                else:
                    train_data, val_data, test_data = self._random_split(
                        transcriptomics, row_metadata
                    )

                # Validate splits
                for name, data in [
                    ("train", train_data),
                    ("val", val_data),
                    ("test", test_data),
                ]:
                    if len(data[0]) == 0 or len(data[1]) == 0:
                        raise ValueError(
                            f"{name.capitalize()} split is empty; check nrows or split proportions"
                        )

                # Create datasets
                train_dataset = self._create_dataset(*train_data)
                val_dataset = self._create_dataset(*val_data)
                test_dataset = self._create_dataset(*test_data)

                logger.info(f"Train set size: {len(train_dataset)}")
                logger.info(f"Validation set size: {len(val_dataset)}")
                logger.info(f"Test set size: {len(test_dataset)}")

                return train_dataset, val_dataset, test_dataset

        except Exception as e:
            logger.error(f"Error processing GCTX data: {str(e)}")
            raise

    def _random_split(
        self, transcriptomics: np.ndarray, row_metadata: pd.DataFrame
    ) -> Tuple[
        Tuple[np.ndarray, pd.DataFrame],
        Tuple[np.ndarray, pd.DataFrame],
        Tuple[np.ndarray, pd.DataFrame],
    ]:
        """
        Split data randomly into train, validation, and test sets.

        Args:
            transcriptomics: Gene expression data
            row_metadata: Row metadata DataFrame

        Returns:
            Tuple of (train_data, val_data, test_data), where each is a tuple of (transcriptomics, metadata)
        """
        indices = np.arange(len(transcriptomics))

        # First split into train+val and test
        train_val_indices, test_indices = train_test_split(
            indices, test_size=self.test_size, random_state=self.random_state
        )

        # Then split train+val into train and val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size_adjusted,
            random_state=self.random_state,
        )

        # Create data subsets
        return (
            (
                transcriptomics[train_indices],
                row_metadata.iloc[train_indices].reset_index(drop=True),
            ),
            (
                transcriptomics[val_indices],
                row_metadata.iloc[val_indices].reset_index(drop=True),
            ),
            (
                transcriptomics[test_indices],
                row_metadata.iloc[test_indices].reset_index(drop=True),
            ),
        )

    def _group_split(
        self, transcriptomics: np.ndarray, row_metadata: pd.DataFrame
    ) -> Tuple[
        Tuple[np.ndarray, pd.DataFrame],
        Tuple[np.ndarray, pd.DataFrame],
        Tuple[np.ndarray, pd.DataFrame],
    ]:
        """
        Split data by groups into train, validation, and test sets.

        This simulates different patient cohorts by ensuring samples from the same group
        (e.g., cell line) are not split across different sets.

        Args:
            transcriptomics: Gene expression data
            row_metadata: Row metadata DataFrame

        Returns:
            Tuple of (train_data, val_data, test_data), where each is a tuple of (transcriptomics, metadata)
        """
        # Ensure minimum number of groups in each split
        unique_groups = row_metadata[self.group_by].unique()
        if len(unique_groups) < 3:
            logger.warning(
                f"Only {len(unique_groups)} unique groups, using random split instead"
            )
            return self._random_split(transcriptomics, row_metadata)

        indices = np.arange(len(transcriptomics))

        # Split into train+val and test, respecting groups
        gss = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )
        train_val_indices, test_indices = next(
            gss.split(indices, groups=row_metadata[self.group_by])
        )

        # Split train+val into train and val, respecting groups
        val_size_adjusted = self.val_size / (1 - self.test_size)
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_size_adjusted, random_state=self.random_state
        )

        groups_train_val = row_metadata.iloc[train_val_indices][self.group_by]
        train_val_subset_indices = np.arange(len(train_val_indices))

        train_subset_indices, val_subset_indices = next(
            gss_val.split(train_val_subset_indices, groups=groups_train_val)
        )

        # Get the actual indices
        train_indices = train_val_indices[train_subset_indices]
        val_indices = train_val_indices[val_subset_indices]

        # Create data subsets
        return (
            (
                transcriptomics[train_indices],
                row_metadata.iloc[train_indices].reset_index(drop=True),
            ),
            (
                transcriptomics[val_indices],
                row_metadata.iloc[val_indices].reset_index(drop=True),
            ),
            (
                transcriptomics[test_indices],
                row_metadata.iloc[test_indices].reset_index(drop=True),
            ),
        )

    def _create_dataset(
        self, transcriptomics: np.ndarray, row_metadata: pd.DataFrame
    ) -> MultimodalDataset:
        """
        Create a MultimodalDataset instance from preloaded data.

        Args:
            transcriptomics: Gene expression data
            row_metadata: Row metadata DataFrame

        Returns:
            MultimodalDataset instance
        """
        return MultimodalDataset(
            transcriptomics_data=transcriptomics,
            metadata=row_metadata,
            transform_transcriptomics=self.transform_transcriptomics,
            transform_molecular=self.transform_molecular,
            additional_features=self.additional_features,
        )

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Process data and create PyTorch DataLoader objects for train, validation, and test sets.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_dataset, val_dataset, test_dataset = self.process()

        train_loader = DataLoader(
            train_dataset.to_pytorch(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset.to_pytorch(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset.to_pytorch(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader


def create_transformations(
    transcriptomics_transform_type: Optional[str] = None,
    molecular_transform_type: Optional[str] = None,
    fingerprint_size: int = 2048,
    fingerprint_radius: int = 3,
    **kwargs,
) -> Tuple[Optional[Callable], Optional[Callable]]:
    """
    Create transformation functions for transcriptomic and molecular data.

    Args:
        transcriptomics_transform_type: Type of transformation for transcriptomic data
            (None, 'normalize', 'scale', etc.)
        molecular_transform_type: Type of transformation for molecular data
            (None, 'fingerprint', 'descriptors', etc.)
        fingerprint_size: Size of Morgan fingerprints if using fingerprint transformation
        fingerprint_radius: Radius for Morgan fingerprints if using fingerprint transformation
        **kwargs: Additional parameters for transformations

    Returns:
        Tuple of (transcriptomics_transform, molecular_transform) functions
    """
    transcriptomics_transform = None
    molecular_transform = None

    # Create transcriptomics transformation
    if transcriptomics_transform_type == "normalize":
        from sklearn.preprocessing import Normalizer

        normalizer = Normalizer()
        transcriptomics_transform = lambda x: normalizer.fit_transform(x)
    elif transcriptomics_transform_type == "scale":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        transcriptomics_transform = lambda x: scaler.fit_transform(x)
    # Add more transformations as needed

    # Create molecular transformation
    if molecular_transform_type == "fingerprint":
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            def fingerprint_transform(mol_input):
                if isinstance(mol_input, dict):
                    smiles_list = mol_input["smiles"]
                    dosage = mol_input["dosage"].reshape(-1, 1)

                    fingerprints = np.zeros((len(smiles_list), fingerprint_size))
                    for i, smiles in enumerate(smiles_list):
                        try:
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                fp = AllChem.GetMorganFingerprintAsBitVect(
                                    mol, fingerprint_radius, nBits=fingerprint_size
                                )
                                fingerprints[i] = np.array(fp)
                        except Exception as e:
                            logger.warning(
                                f"Error generating fingerprint for {smiles}: {e}"
                            )

                    # Combine fingerprints with dosage
                    return np.hstack([fingerprints, dosage])
                else:
                    return mol_input

            molecular_transform = fingerprint_transform
        except ImportError:
            logger.warning("RDKit not available; fingerprint transformation disabled")
    elif molecular_transform_type == "descriptors":
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            def descriptor_transform(mol_input):
                if isinstance(mol_input, dict):
                    smiles_list = mol_input["smiles"]
                    dosage = mol_input["dosage"].reshape(-1, 1)

                    # List of descriptor functions to calculate
                    descriptor_fns = [
                        Descriptors.MolWt,
                        Descriptors.MolLogP,
                        Descriptors.NumHDonors,
                        Descriptors.NumHAcceptors,
                        Descriptors.NumRotatableBonds,
                        Descriptors.TPSA,
                        Descriptors.NumAromaticRings,
                        Descriptors.NumHeteroatoms,
                        Descriptors.FractionCSP3,
                        Descriptors.NumAliphaticRings,
                    ]

                    descriptors = np.zeros((len(smiles_list), len(descriptor_fns)))
                    for i, smiles in enumerate(smiles_list):
                        try:
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                descriptors[i] = [fn(mol) for fn in descriptor_fns]
                        except Exception as e:
                            logger.warning(
                                f"Error generating descriptors for {smiles}: {e}"
                            )

                    # Combine descriptors with dosage
                    return np.hstack([descriptors, dosage])
                else:
                    return mol_input

            molecular_transform = descriptor_transform
        except ImportError:
            logger.warning("RDKit not available; descriptor transformation disabled")
    # Add more transformations as needed

    return transcriptomics_transform, molecular_transform
