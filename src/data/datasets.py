# src/data/datasets.py
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.loaders import GCTXDataLoader

logger = logging.getLogger(__name__)


class MultimodalDrugDataset(Dataset):
    def __init__(
        self,
        transcriptomics_data: np.ndarray,
        metadata: pd.DataFrame,
        transform_transcriptomics: Optional[Callable] = None,
        transform_molecular: Optional[Callable] = None,
    ):
        self.transcriptomics_data = transcriptomics_data
        self.metadata = metadata
        self.transform_transcriptomics = transform_transcriptomics
        self.transform_molecular = transform_molecular

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        transcriptomics = torch.tensor(
            self.transcriptomics_data[idx], dtype=torch.float32
        )
        if self.transform_transcriptomics:
            transcriptomics = self.transform_transcriptomics(transcriptomics)

        smiles = self.metadata.iloc[idx]["canonical_smiles"]
        dosage = torch.tensor(self.metadata.iloc[idx]["pert_dose"], dtype=torch.float32)

        molecular = None
        if self.transform_molecular:
            mol_input = {"smiles": [smiles], "dosage": [dosage]}
            molecular = self.transform_molecular(mol_input)[0]
            molecular = torch.from_numpy(molecular)

        viability = torch.tensor(
            self.metadata.iloc[idx]["viability"], dtype=torch.float32
        )

        sample = {
            "transcriptomics": transcriptomics,
            "molecular": molecular,
            "dosage": dosage,
            "viability": viability,
        }
        return sample


class TranscriptomicsDataset(Dataset):
    def __init__(
        self,
        transcriptomics_data: np.ndarray,
        viability: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        self.transcriptomics_data = transcriptomics_data
        self.viability = viability
        self.transform = transform

    def __len__(self) -> int:
        return len(self.transcriptomics_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.transcriptomics_data[idx], dtype=torch.float32)
        y = torch.tensor(self.viability[idx], dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return x, y


class ChemicalDataset(Dataset):
    def __init__(
        self,
        smiles: np.ndarray,
        dosage: np.ndarray,
        viability: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        self.smiles = smiles
        self.dosage = dosage
        self.viability = viability
        self.transform = transform

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        sample = {
            "smiles": self.smiles[idx],
            "dosage": torch.tensor(self.dosage[idx], dtype=torch.float32),
            "viability": torch.tensor(self.viability[idx], dtype=torch.float32),
        }
        if self.transform:
            sample["molecular"] = self.transform(sample["smiles"])
        return sample


class DatasetFactory:
    """Factory for creating and splitting datasets from GCTX data with chunking support."""

    @staticmethod
    def _split_data(
        metadata: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        group_by: Optional[str] = None,
        stratify_by: Optional[str] = None,
        chunk_size: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split indices based on metadata with support for large datasets.

        Args:
            metadata: DataFrame containing metadata
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            group_by: Column to use for group-based splitting
            stratify_by: Column to use for stratified splitting
            chunk_size: Size of chunks for processing large datasets

        Returns:
            Tuple of train, validation, and test indices
        """
        logger.info("Performing data splitting...")

        # Ensure test_size + val_size < 1
        if test_size + val_size >= 1.0:
            raise ValueError("Combined test and validation sizes must be less than 1.0")

        indices = np.arange(len(metadata))

        # Group-based splitting
        if group_by and len(metadata[group_by].unique()) >= 3:
            logger.info(f"Using group-based splitting based on '{group_by}'")
            try:
                gss = GroupShuffleSplit(
                    n_splits=1, test_size=test_size, random_state=random_state
                )
                train_val_idx, test_idx = next(
                    gss.split(indices, groups=metadata[group_by])
                )

                # Adjust validation size for the remaining train set
                val_size_adjusted = val_size / (1 - test_size)
                groups_train_val = metadata.iloc[train_val_idx][group_by]

                gss_val = GroupShuffleSplit(
                    n_splits=1, test_size=val_size_adjusted, random_state=random_state
                )
                train_idx, val_idx = next(
                    gss_val.split(
                        np.arange(len(train_val_idx)), groups=groups_train_val
                    )
                )
                train_idx = train_val_idx[train_idx]
                val_idx = train_val_idx[val_idx]

            except Exception as e:
                logger.warning(
                    f"Group-based splitting failed: {e}. Falling back to standard splitting."
                )
                group_by = None

        # Standard stratified or random splitting
        if not group_by:
            # Prepare stratification
            stratify = None
            if stratify_by:
                try:
                    # Handle continuous variables
                    if metadata[stratify_by].dtype in [float, int]:
                        stratify = pd.qcut(
                            metadata[stratify_by], q=10, labels=False, duplicates="drop"
                        )
                    else:
                        stratify = metadata[stratify_by]
                    logger.info(f"Using stratified splitting based on '{stratify_by}'")
                except Exception as e:
                    logger.warning(
                        f"Stratification failed: {e}. Using random splitting."
                    )
                    stratify = None

            # Split into train+val and test sets
            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )

            # Adjust validation size for train set
            val_size_adjusted = val_size / (1 - test_size)
            stratify_val = (
                stratify.iloc[train_val_idx] if stratify is not None else None
            )

            # Split train+val into train and validation sets
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=stratify_val,
            )

        # Ensure indices are numpy arrays
        train_idx = np.array(train_idx, dtype=int)
        val_idx = np.array(val_idx, dtype=int)
        test_idx = np.array(test_idx, dtype=int)

        logger.info(
            f"Splitting complete. Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}"
        )
        return train_idx, val_idx, test_idx

    @staticmethod
    def _process_large_dataset(
        gctx_loader: GCTXDataLoader,
        nrows: Optional[int] = None,
        feature_space: Union[str, List[str]] = "landmark",
        chunk_size: int = 10000,
    ) -> pd.DataFrame:
        """
        Process large datasets in chunks to manage memory.

        Args:
            gctx_loader: GCTX data loader
            nrows: Number of rows to load
            feature_space: Gene feature space
            chunk_size: Size of chunks to process

        Returns:
            Processed metadata DataFrame
        """
        logger.info("Processing large dataset with chunked loading...")

        with gctx_loader:
            total_rows = (
                gctx_loader._n_rows
                if nrows is None
                else min(nrows, gctx_loader._n_rows)
            )

            # Process in chunks if dataset is large
            if total_rows > chunk_size:
                metadata_chunks = []
                for start in tqdm(
                    range(0, total_rows, chunk_size), desc="Loading metadata"
                ):
                    end = min(start + chunk_size, total_rows)
                    chunk = gctx_loader.get_row_metadata(row_slice=slice(start, end))
                    metadata_chunks.append(chunk)

                metadata = pd.concat(metadata_chunks, ignore_index=True)
            else:
                # Load entire dataset if small enough
                metadata = gctx_loader.get_row_metadata(row_slice=slice(0, total_rows))

        # Ensure metadata is a DataFrame
        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.DataFrame(metadata)

        return metadata

    @staticmethod
    def create_and_split_multimodal(
        gctx_loader: GCTXDataLoader,
        feature_space: Union[str, List[str]] = "landmark",
        nrows: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        group_by: Optional[str] = None,
        stratify_by: Optional[str] = None,
        transform_transcriptomics: Optional[Callable] = None,
        transform_molecular: Optional[Callable] = None,
        chunk_size: int = 10000,
    ) -> Tuple[MultimodalDrugDataset, MultimodalDrugDataset, MultimodalDrugDataset]:
        """Create and split multimodal datasets with chunking for large datasets."""
        logger.info("Creating and splitting multimodal datasets with chunking...")

        # Process metadata
        metadata = DatasetFactory._process_large_dataset(
            gctx_loader, nrows, feature_space, chunk_size
        )

        # Validate required columns
        required_cols = ["canonical_smiles", "pert_dose", "viability"]
        if group_by:
            required_cols.append(group_by)
        if stratify_by:
            required_cols.append(stratify_by)

        missing = [col for col in required_cols if col not in metadata.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Split indices based on metadata
        train_idx, val_idx, test_idx = DatasetFactory._split_data(
            metadata,
            test_size,
            val_size,
            random_state,
            group_by,
            stratify_by,
            chunk_size,
        )

        train_idx = train_idx.tolist()  # or np.array(train_idx)
        val_idx = val_idx.tolist()  # or np.array(val_idx)
        test_idx = test_idx.tolist()  # or np.array(test_idx)

        # Create datasets using split indices
        train_ds = MultimodalDrugDataset(
            transcriptomics_data=gctx_loader.get_expression_data(
                row_slice=train_idx, feature_space=feature_space
            ),
            metadata=metadata.iloc[train_idx].reset_index(drop=True),
            transform_transcriptomics=transform_transcriptomics,
            transform_molecular=transform_molecular,
        )
        val_ds = MultimodalDrugDataset(
            transcriptomics_data=gctx_loader.get_expression_data(
                row_slice=val_idx, feature_space=feature_space
            ),
            metadata=metadata.iloc[val_idx].reset_index(drop=True),
            transform_transcriptomics=transform_transcriptomics,
            transform_molecular=transform_molecular,
        )
        test_ds = MultimodalDrugDataset(
            transcriptomics_data=gctx_loader.get_expression_data(
                row_slice=test_idx, feature_space=feature_space
            ),
            metadata=metadata.iloc[test_idx].reset_index(drop=True),
            transform_transcriptomics=transform_transcriptomics,
            transform_molecular=transform_molecular,
        )

        logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        return train_ds, val_ds, test_ds

    @staticmethod
    def create_and_split_transcriptomics(
        gctx_loader: GCTXDataLoader,
        feature_space: Union[str, List[str]] = "landmark",
        nrows: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        group_by: Optional[str] = None,
        stratify_by: Optional[str] = None,
        transform: Optional[Callable] = None,
        chunk_size: int = 10000,
    ) -> Tuple[TranscriptomicsDataset, TranscriptomicsDataset, TranscriptomicsDataset]:
        """Create and split transcriptomics datasets with chunking."""
        logger.info("Creating and splitting transcriptomics datasets with chunking...")

        # Process metadata
        metadata = DatasetFactory._process_large_dataset(
            gctx_loader, nrows, feature_space, chunk_size
        )

        # Validate required columns
        required_cols = ["viability"]
        if group_by:
            required_cols.append(group_by)
        if stratify_by:
            required_cols.append(stratify_by)

        missing = [col for col in required_cols if col not in metadata.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Split indices based on metadata
        train_idx, val_idx, test_idx = DatasetFactory._split_data(
            metadata,
            test_size,
            val_size,
            random_state,
            group_by,
            stratify_by,
            chunk_size,
        )

        train_idx = train_idx.tolist()  # or np.array(train_idx)
        val_idx = val_idx.tolist()  # or np.array(val_idx)
        test_idx = test_idx.tolist()  # or np.array(test_idx)

        # Create datasets using split indices
        train_ds = TranscriptomicsDataset(
            transcriptomics_data=gctx_loader.get_expression_data(
                row_slice=train_idx, feature_space=feature_space
            ),
            viability=metadata.iloc[train_idx]["viability"].values,
            transform=transform,
        )
        val_ds = TranscriptomicsDataset(
            transcriptomics_data=gctx_loader.get_expression_data(
                row_slice=val_idx, feature_space=feature_space
            ),
            viability=metadata.iloc[val_idx]["viability"].values,
            transform=transform,
        )
        test_ds = TranscriptomicsDataset(
            transcriptomics_data=gctx_loader.get_expression_data(
                row_slice=test_idx, feature_space=feature_space
            ),
            viability=metadata.iloc[test_idx]["viability"].values,
            transform=transform,
        )

        logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        return train_ds, val_ds, test_ds

    @staticmethod
    def create_and_split_chemical(
        gctx_loader: GCTXDataLoader,
        nrows: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        group_by: Optional[str] = None,
        stratify_by: Optional[str] = None,
        transform: Optional[Callable] = None,
        chunk_size: int = 10000,
    ) -> Tuple[ChemicalDataset, ChemicalDataset, ChemicalDataset]:
        """Create and split chemical datasets with chunking."""
        logger.info("Creating and splitting chemical datasets with chunking...")

        # Process metadata
        metadata = DatasetFactory._process_large_dataset(
            gctx_loader, nrows, "landmark", chunk_size
        )

        # Validate required columns
        required_cols = ["canonical_smiles", "pert_dose", "viability"]
        if group_by:
            required_cols.append(group_by)
        if stratify_by:
            required_cols.append(stratify_by)

        missing = [col for col in required_cols if col not in metadata.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Split indices based on metadata
        train_idx, val_idx, test_idx = DatasetFactory._split_data(
            metadata,
            test_size,
            val_size,
            random_state,
            group_by,
            stratify_by,
            chunk_size,
        )

        train_idx = train_idx.tolist()  # or np.array(train_idx)
        val_idx = val_idx.tolist()  # or np.array(val_idx)
        test_idx = test_idx.tolist()  # or np.array(test_idx)

        # Create datasets using split indices
        train_ds = ChemicalDataset(
            smiles=metadata.iloc[train_idx]["canonical_smiles"].values,
            dosage=metadata.iloc[train_idx]["pert_dose"].values,
            viability=metadata.iloc[train_idx]["viability"].values,
            transform=transform,
        )
        val_ds = ChemicalDataset(
            smiles=metadata.iloc[val_idx]["canonical_smiles"].values,
            dosage=metadata.iloc[val_idx]["pert_dose"].values,
            viability=metadata.iloc[val_idx]["viability"].values,
            transform=transform,
        )
        test_ds = ChemicalDataset(
            smiles=metadata.iloc[test_idx]["canonical_smiles"].values,
            dosage=metadata.iloc[test_idx]["pert_dose"].values,
            viability=metadata.iloc[test_idx]["viability"].values,
            transform=transform,
        )

        logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        return train_ds, val_ds, test_ds
