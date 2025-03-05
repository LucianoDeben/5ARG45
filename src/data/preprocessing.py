# src/data/preprocessing.py
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from src.data.loaders import GCTXDataLoader

logger = logging.getLogger(__name__)


class LINCSCTRPDataProcessor:
    """Preprocessor for LINCS/CTRP data before dataset creation."""

    def __init__(
        self,
        gctx_file: str,
        feature_space: Union[str, List[str]] = "landmark",
        nrows: Optional[int] = None,
        transform_transcriptomics: Optional[Callable] = None,
        transform_molecular: Optional[Callable] = None,
        imputation_strategy: str = "mean",
        handle_outliers: bool = False,
        outlier_threshold: float = 3.0,
    ):
        self.gctx_file = gctx_file
        self.feature_space = feature_space
        self.nrows = nrows
        self.transform_transcriptomics = transform_transcriptomics
        self.transform_molecular = transform_molecular
        self.imputation_strategy = imputation_strategy
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold
        self._transcriptomics = None
        self._metadata = None

    def _load_data(self) -> Tuple[np.ndarray, pd.DataFrame]:
        with GCTXDataLoader(self.gctx_file) as loader:
            row_slice = slice(0, self.nrows) if self.nrows else None
            data, meta, _ = loader.get_data_with_metadata(
                row_slice=row_slice, feature_space=self.feature_space
            )
            return data, meta

    def _impute_missing_values(
        self, data: np.ndarray, metadata: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Impute missing values in data and metadata using the specified strategy."""
        # Impute transcriptomics data
        if np.isnan(data).any():
            logger.info(
                f"Imputing {np.isnan(data).sum()} missing values in transcriptomics data"
            )
            if self.imputation_strategy == "mean":
                col_means = np.nanmean(data, axis=0)
                inds = np.where(np.isnan(data))
                data[inds] = np.take(col_means, inds[1])
            elif self.imputation_strategy == "median":
                col_medians = np.nanmedian(data, axis=0)
                inds = np.where(np.isnan(data))
                data[inds] = np.take(col_medians, inds[1])
            elif self.imputation_strategy == "knn":
                imputer = KNNImputer(n_neighbors=5)
                data = imputer.fit_transform(data)
            else:
                raise ValueError(
                    f"Unsupported imputation strategy: {self.imputation_strategy}"
                )

        # Impute metadata
        if "viability" in metadata.columns and metadata["viability"].isna().any():
            if self.imputation_strategy == "mean":
                metadata["viability"] = metadata["viability"].fillna(
                    metadata["viability"].mean()
                )
            elif self.imputation_strategy == "median":
                metadata["viability"] = metadata["viability"].fillna(
                    metadata["viability"].median()
                )
            elif self.imputation_strategy == "knn":
                # For simplicity, use mean for metadata in case of KNN
                metadata["viability"] = metadata["viability"].fillna(
                    metadata["viability"].mean()
                )

        if "pert_dose" in metadata.columns and metadata["pert_dose"].isna().any():
            metadata["pert_dose"] = metadata["pert_dose"].fillna(
                metadata["pert_dose"].median()
            )

        return data, metadata

    def _handle_outliers(self, data: np.ndarray) -> np.ndarray:
        """Handle outliers in transcriptomics data using z-score clipping."""
        if not self.handle_outliers:
            return data

        logger.info("Handling outliers in transcriptomics data")
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        z_scores = np.abs((data - means) / (stds + 1e-8))

        # Create a mask for values beyond the threshold
        mask = z_scores > self.outlier_threshold
        outlier_count = np.sum(mask)

        if outlier_count > 0:
            logger.info(
                f"Clipping {outlier_count} outliers ({outlier_count/(data.size)*100:.2f}%)"
            )
            # Replace outliers with the threshold value (in the right direction)
            max_allowed = means + self.outlier_threshold * stds
            min_allowed = means - self.outlier_threshold * stds
            data = np.minimum(data, max_allowed)
            data = np.maximum(data, min_allowed)

        return data

    def _validate_smiles(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Validate SMILES strings and remove invalid entries."""
        if "canonical_smiles" not in metadata.columns:
            return metadata

        valid_smiles = metadata["canonical_smiles"].apply(
            lambda x: isinstance(x, str) and len(x) > 0
        )
        invalid_count = (~valid_smiles).sum()

        if invalid_count > 0:
            logger.warning(
                f"Removing {invalid_count} samples with invalid SMILES strings"
            )
            return metadata[valid_smiles]
        return metadata

    def preprocess(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Preprocess data with imputation, outlier handling, and transformations."""
        if self._transcriptomics is None or self._metadata is None:
            logger.info(f"Loading data from {self.gctx_file}")
            self._transcriptomics, self._metadata = self._load_data()

        logger.info(f"Initial data shape: {self._transcriptomics.shape}")

        # Impute missing values
        self._transcriptomics, self._metadata = self._impute_missing_values(
            self._transcriptomics, self._metadata
        )

        # Handle outliers
        self._transcriptomics = self._handle_outliers(self._transcriptomics)

        # Validate SMILES
        valid_metadata = self._validate_smiles(self._metadata)
        if len(valid_metadata) < len(self._metadata):
            valid_indices = valid_metadata.index
            self._transcriptomics = self._transcriptomics[valid_indices]
            self._metadata = valid_metadata

        # Apply transcriptomics transformation
        if self.transform_transcriptomics:
            logger.info("Applying transcriptomics transformation")
            if hasattr(self.transform_transcriptomics, "fit_transform"):
                self._transcriptomics = self.transform_transcriptomics.fit_transform(
                    self._transcriptomics
                )
            else:
                self._transcriptomics = self.transform_transcriptomics(
                    self._transcriptomics
                )

        logger.info(f"Preprocessed data: {len(self._transcriptomics)} samples")
        return self._transcriptomics, self._metadata
