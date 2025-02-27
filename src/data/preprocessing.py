import logging
from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader

from data.datasets import MultimodalDataset
from data.loaders import GCTXDataLoader

logger = logging.getLogger(__name__)


class LINCSCTRPDataProcessor:
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
        """
        self.gctx_file = gctx_file
        self.feature_space = feature_space
        self.nrows = nrows
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.group_by = group_by
        self.batch_size = batch_size

    def process(self) -> Tuple[MultimodalDataset, MultimodalDataset, MultimodalDataset]:
        logger.info(
            f"Processing LINCS/CTRP data with feature_space={self.feature_space}, nrows={self.nrows}"
        )
        loader = GCTXDataLoader(self.gctx_file)

        with loader:
            row_slice = slice(0, self.nrows) if self.nrows is not None else None
            transcriptomics = pd.DataFrame(
                loader.get_expression_data(
                    row_slice=row_slice, feature_space=self.feature_space
                )
            )
            row_metadata = loader.get_row_metadata(row_slice=row_slice)

            required_cols = ["canonical_smiles", "viability"]
            if self.group_by:
                required_cols.append(self.group_by)
            for col in required_cols:
                if col not in row_metadata.columns:
                    raise ValueError(f"Missing required column in row metadata: {col}")

            transcriptomics.index = row_metadata.index

            if self.group_by:
                train_data, val_data, test_data = self._group_split(
                    transcriptomics, row_metadata
                )
            else:
                train_data, val_data, test_data = self._random_split(
                    transcriptomics, row_metadata
                )

            for name, data in [
                ("train", train_data),
                ("val", val_data),
                ("test", test_data),
            ]:
                if data[0].empty or data[1].empty:
                    raise ValueError(
                        f"{name.capitalize()} split is empty; check nrows or split proportions"
                    )

            train_dataset = self._create_dataset(train_data[0], train_data[1])
            val_dataset = self._create_dataset(val_data[0], val_data[1])
            test_dataset = self._create_dataset(test_data[0], test_data[1])

            logger.info(f"Train set size: {len(train_dataset)}")
            logger.info(f"Validation set size: {len(val_dataset)}")
            logger.info(f"Test set size: {len(test_dataset)}")

            return train_dataset, val_dataset, test_dataset

    def _random_split(
        self, transcriptomics: pd.DataFrame, row_metadata: pd.DataFrame
    ) -> Tuple:
        """Split data randomly into train, validation, and test sets."""
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
            (transcriptomics.loc[test_idx], row_metadata.loc[test_idx]),
        )

    def _group_split(
        self, transcriptomics: pd.DataFrame, row_metadata: pd.DataFrame
    ) -> Tuple:
        """Split data by groups into train, validation, and test sets."""
        gss = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )
        train_val_idx, test_idx = next(
            gss.split(row_metadata, groups=row_metadata[self.group_by])
        )
        gss_val = GroupShuffleSplit(
            n_splits=1,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.random_state,
        )
        train_idx, val_idx = next(
            gss_val.split(
                row_metadata.iloc[train_val_idx],
                groups=row_metadata.iloc[train_val_idx][self.group_by],
            )
        )
        return (
            (transcriptomics.iloc[train_idx], row_metadata.iloc[train_idx]),
            (transcriptomics.iloc[val_idx], row_metadata.iloc[val_idx]),
            (transcriptomics.iloc[test_idx], row_metadata.iloc[test_idx]),
        )

    def _create_dataset(
        self, transcriptomics: pd.DataFrame, row_metadata: pd.DataFrame
    ) -> MultimodalDataset:
        """Create a MultimodalDataset instance from preloaded data."""
        return MultimodalDataset(
            transcriptomics=transcriptomics,
            row_metadata=row_metadata,
            smiles_column="canonical_smiles",
            target_column="viability",
            metadata_columns=[self.group_by] if self.group_by else None,
            transforms=None,
        )

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Return PyTorch DataLoaders for train, validation, and test sets.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        train_dataset, val_dataset, test_dataset = self.process()
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        return train_loader, val_loader, test_loader
