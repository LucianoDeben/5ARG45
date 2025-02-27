import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from torch.utils.data import DataLoader
from typing import Tuple, Optional
from data.loaders import GCTXDataLoader
from data.datasets import MultimodalDataset

logger = logging.getLogger(__name__)

class LINCSCTRPDataProcessor:
    def __init__(
        self,
        lincs_file: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        group_by: Optional[str] = None,
        batch_size: int = 32
    ):
        self.lincs_file = lincs_file
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.group_by = group_by
        self.batch_size = batch_size

    def process(self) -> Tuple[MultimodalDataset, MultimodalDataset, MultimodalDataset]:
        logger.info("Processing LINCS data with embedded CTRP metadata")
        loader = GCTXDataLoader(self.lincs_file)

        with loader:
            transcriptomics = pd.DataFrame(loader.get_expression_data())
            row_metadata = loader.get_row_metadata()

            required_cols = ["canonical_smiles", "viability"]
            if self.group_by:
                required_cols.append(self.group_by)
            for col in required_cols:
                if col not in row_metadata.columns:
                    raise ValueError(f"Missing required column in row metadata: {col}")

            transcriptomics.index = row_metadata.index

            if self.group_by:
                train_data, val_data, test_data = self._group_split(transcriptomics, row_metadata)
            else:
                train_data, val_data, test_data = self._random_split(transcriptomics, row_metadata)

            train_dataset = self._create_dataset(loader, train_data[1])
            val_dataset = self._create_dataset(loader, val_data[1])
            test_dataset = self._create_dataset(loader, test_data[1])

            logger.info(f"Train set size: {len(train_dataset)}")
            logger.info(f"Validation set size: {len(val_dataset)}")
            logger.info(f"Test set size: {len(test_dataset)}")

            return train_dataset, val_dataset, test_dataset

    def _random_split(self, transcriptomics: pd.DataFrame, row_metadata: pd.DataFrame) -> Tuple:
        train_val_idx, test_idx = train_test_split(
            row_metadata.index, test_size=self.test_size, random_state=self.random_state
        )
        val_size_adjusted = self.val_size / (1 - self.test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size_adjusted, random_state=self.random_state
        )
        return (
            (transcriptomics.loc[train_idx], row_metadata.loc[train_idx]),
            (transcriptomics.loc[val_idx], row_metadata.loc[val_idx]),
            (transcriptomics.loc[test_idx], row_metadata.loc[test_idx])
        )

    def _group_split(self, transcriptomics: pd.DataFrame, row_metadata: pd.DataFrame) -> Tuple:
        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        train_val_idx, test_idx = next(gss.split(row_metadata, groups=row_metadata[self.group_by]))
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=self.val_size / (1 - self.test_size), random_state=self.random_state
        )
        train_idx, val_idx = next(gss_val.split(
            row_metadata.iloc[train_val_idx], groups=row_metadata.iloc[train_val_idx][self.group_by]
        ))
        return (
            (transcriptomics.iloc[train_idx], row_metadata.iloc[train_idx]),
            (transcriptomics.iloc[val_idx], row_metadata.iloc[val_idx]),
            (transcriptomics.iloc[test_idx], row_metadata.iloc[test_idx])
        )

    def _create_dataset(self, loader: GCTXDataLoader, row_metadata: pd.DataFrame) -> MultimodalDataset:
        return MultimodalDataset(
            gctx_loader=loader,
            smiles_column="canonical_smiles",
            target_column="viability",
            metadata_columns=[self.group_by] if self.group_by else None,
            indices=row_metadata.index.tolist()
        )

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_dataset, val_dataset, test_dataset = self.process()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader