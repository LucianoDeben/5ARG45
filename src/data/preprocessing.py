# data/preprocessing.py
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader

from data.datasets import ChemicalDataset, MultimodalDrugDataset, TranscriptomicsDataset
from data.loaders import GCTXDataLoader
from data.transformers import create_transformations

logger = logging.getLogger(__name__)


class LINCSCTRPDataProcessor:
    """Processor for LINCS/CTRP data with support for different model types."""

    def __init__(
        self,
        gctx_file: str,
        feature_space: Union[str, List[str]] = "landmark",
        nrows: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        group_by: Optional[str] = None,
        stratify_by: Optional[str] = None,
        batch_size: int = 32,
        transform_transcriptomics: Optional[Callable] = None,
        transform_molecular: Optional[Callable] = None,
    ):
        """
        Initialize data processor.

        Args:
            gctx_file: Path to the GCTX file
            feature_space: Gene feature space to use ('landmark', 'best inferred', etc.)
            nrows: Number of rows to load (None for all)
            test_size: Fraction of data to use for testing
            val_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            group_by: Column name to use for group-based splitting (e.g., 'cell_id')
            stratify_by: Column name to use for stratified splitting (e.g., 'viability_bin')
            batch_size: Batch size for dataloaders
            transform_transcriptomics: Transformation function for transcriptomics data
            transform_molecular: Transformation function for molecular data
        """
        self.gctx_file = gctx_file
        self.feature_space = feature_space
        self.nrows = nrows
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.group_by = group_by
        self.stratify_by = stratify_by
        self.batch_size = batch_size
        self.transform_transcriptomics = transform_transcriptomics
        self.transform_molecular = transform_molecular

        # Set random seed
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        # Validate that group_by and stratify_by are not both specified
        if self.group_by and self.stratify_by:
            raise ValueError(
                "Cannot specify both group_by and stratify_by simultaneously"
            )

    def _load_data(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load data from GCTX file."""
        with GCTXDataLoader(self.gctx_file) as loader:
            row_slice = slice(0, self.nrows) if self.nrows is not None else None
            transcriptomics, metadata, _ = loader.get_data_with_metadata(
                row_slice=row_slice,
                feature_space=self.feature_space,
            )

            # Validate metadata
            required_cols = ["canonical_smiles", "pert_dose", "viability"]
            if self.group_by:
                required_cols.append(self.group_by)
            if self.stratify_by:
                required_cols.append(self.stratify_by)

            missing = [col for col in required_cols if col not in metadata.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            return transcriptomics, metadata

    def _split_data(self, transcriptomics: np.ndarray, metadata: pd.DataFrame) -> Tuple[
        Tuple[np.ndarray, pd.DataFrame],
        Tuple[np.ndarray, pd.DataFrame],
        Tuple[np.ndarray, pd.DataFrame],
    ]:
        """Split data into train/val/test sets."""
        if self.group_by and len(metadata[self.group_by].unique()) >= 3:
            return self._group_split(transcriptomics, metadata)
        return self._random_split(transcriptomics, metadata)

    def _random_split(
        self, transcriptomics: np.ndarray, metadata: pd.DataFrame
    ) -> Tuple[
        Tuple[np.ndarray, pd.DataFrame],
        Tuple[np.ndarray, pd.DataFrame],
        Tuple[np.ndarray, pd.DataFrame],
    ]:
        """Random splitting strategy with optional stratification."""
        indices = np.arange(len(transcriptomics))

        # Use stratification if specified
        stratify = None
        if self.stratify_by and self.stratify_by in metadata.columns:
            stratify = metadata[self.stratify_by]
            logger.info(f"Using stratified splitting based on '{self.stratify_by}'")

        # First split into train+val and test
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        # Adjusted stratification for val split
        stratify_val = None
        if stratify is not None:
            stratify_val = stratify.iloc[train_val_idx]

        # Then split train+val into train and val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=stratify_val,
        )

        # Create splits
        splits = []
        for idx in [train_idx, val_idx, test_idx]:
            splits.append(
                (transcriptomics[idx], metadata.iloc[idx].reset_index(drop=True))
            )
        return tuple(splits)

    def _group_split(
        self, transcriptomics: np.ndarray, metadata: pd.DataFrame
    ) -> Tuple[
        Tuple[np.ndarray, pd.DataFrame],
        Tuple[np.ndarray, pd.DataFrame],
        Tuple[np.ndarray, pd.DataFrame],
    ]:
        """Group-based splitting strategy."""
        indices = np.arange(len(transcriptomics))

        logger.info(f"Using group-based splitting based on '{self.group_by}'")
        unique_groups = metadata[self.group_by].unique()
        logger.info(f"Found {len(unique_groups)} unique groups")

        # Split into train+val and test
        gss = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )
        train_val_idx, test_idx = next(
            gss.split(indices, groups=metadata[self.group_by])
        )

        # Split train+val into train and val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        groups_train_val = metadata.iloc[train_val_idx][self.group_by]
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_size_adjusted, random_state=self.random_state
        )
        train_idx, val_idx = next(
            gss_val.split(np.arange(len(train_val_idx)), groups=groups_train_val)
        )

        # Get final indices
        train_idx = train_val_idx[train_idx]
        val_idx = train_val_idx[val_idx]

        # Create splits
        splits = []
        for idx in [train_idx, val_idx, test_idx]:
            splits.append(
                (transcriptomics[idx], metadata.iloc[idx].reset_index(drop=True))
            )
        return tuple(splits)

    def get_multimodal_data(
        self,
    ) -> Tuple[MultimodalDrugDataset, MultimodalDrugDataset, MultimodalDrugDataset]:
        """Get datasets for deep learning models."""
        logger.info("Preparing multimodal datasets...")

        # Load and split data
        transcriptomics, metadata = self._load_data()
        train_data, val_data, test_data = self._split_data(transcriptomics, metadata)

        # Create datasets
        train_ds = MultimodalDrugDataset(
            *train_data,
            transform_transcriptomics=self.transform_transcriptomics,
            transform_molecular=self.transform_molecular,
        )
        val_ds = MultimodalDrugDataset(
            *val_data,
            transform_transcriptomics=self.transform_transcriptomics,
            transform_molecular=self.transform_molecular,
        )
        test_ds = MultimodalDrugDataset(
            *test_data,
            transform_transcriptomics=self.transform_transcriptomics,
            transform_molecular=self.transform_molecular,
        )

        logger.info(
            f"Created datasets - Train: {len(train_ds)}, "
            f"Val: {len(val_ds)}, Test: {len(test_ds)}"
        )
        return train_ds, val_ds, test_ds

    def get_transcriptomics_data(
        self,
    ) -> Tuple[TranscriptomicsDataset, TranscriptomicsDataset, TranscriptomicsDataset]:
        """Get datasets for traditional ML models using only transcriptomics data."""
        logger.info("Preparing transcriptomics datasets...")

        # Load and split data
        transcriptomics, metadata = self._load_data()
        train_data, val_data, test_data = self._split_data(transcriptomics, metadata)

        # Create datasets
        train_ds = TranscriptomicsDataset(
            train_data[0],
            train_data[1]["viability"].values,
            transform=self.transform_transcriptomics,
        )
        val_ds = TranscriptomicsDataset(
            val_data[0],
            val_data[1]["viability"].values,
            transform=self.transform_transcriptomics,
        )
        test_ds = TranscriptomicsDataset(
            test_data[0],
            test_data[1]["viability"].values,
            transform=self.transform_transcriptomics,
        )

        logger.info(
            f"Created datasets - Train: {len(train_ds)}, "
            f"Val: {len(val_ds)}, Test: {len(test_ds)}"
        )
        return train_ds, val_ds, test_ds

    def get_chemical_data(
        self,
    ) -> Tuple[ChemicalDataset, ChemicalDataset, ChemicalDataset]:
        """Get datasets for traditional ML models using only chemical/molecular data."""
        logger.info("Preparing chemical datasets...")

        # Load and split data
        transcriptomics, metadata = self._load_data()
        train_data, val_data, test_data = self._split_data(transcriptomics, metadata)

        # Create datasets
        train_ds = ChemicalDataset(
            smiles=train_data[1]["canonical_smiles"].values,
            dosage=train_data[1]["pert_dose"].values,
            viability=train_data[1]["viability"].values,
            transform=self.transform_molecular,
        )
        val_ds = ChemicalDataset(
            smiles=val_data[1]["canonical_smiles"].values,
            dosage=val_data[1]["pert_dose"].values,
            viability=val_data[1]["viability"].values,
            transform=self.transform_molecular,
        )
        test_ds = ChemicalDataset(
            smiles=test_data[1]["canonical_smiles"].values,
            dosage=test_data[1]["pert_dose"].values,
            viability=test_data[1]["viability"].values,
            transform=self.transform_molecular,
        )

        logger.info(
            f"Created datasets - Train: {len(train_ds)}, "
            f"Val: {len(val_ds)}, Test: {len(test_ds)}"
        )
        return train_ds, val_ds, test_ds

    def get_multimodal_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get PyTorch dataloaders for deep learning."""
        train_ds, val_ds, test_ds = self.get_multimodal_data()

        # Use single process data loading on Windows
        num_workers = 0 if os.name == "nt" else 4

        return (
            DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            ),
            DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
            DataLoader(
                test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        )
