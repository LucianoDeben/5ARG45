# src/data/preprocessing_transforms.py
"""
Preprocessing transformations for normalizing and scaling data.
"""
import logging
from typing import Callable, Union

import numpy as np
from sklearn.preprocessing import Normalizer as SklearnNormalizer
from sklearn.preprocessing import StandardScaler as SklearnScaler

logger = logging.getLogger(__name__)


class Normalizer:
    """Normalizes data using L2 norm with fit/transform separation."""

    def __init__(self):
        """Initialize with an L2 Normalizer from sklearn."""
        self.scaler = SklearnNormalizer()

    def fit(self, x: np.ndarray) -> "Normalizer":
        """Fit the normalizer to the data."""
        self.scaler.fit(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the data using the fitted normalizer."""
        return self.scaler.transform(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the transformation directly (assumes fitted)."""
        return self.transform(x)


class StandardScaler:
    """Standardizes data to zero mean and unit variance with fit/transform separation."""

    def __init__(self):
        """Initialize with a StandardScaler from sklearn."""
        self.scaler = SklearnScaler()

    def fit(self, x: np.ndarray) -> "StandardScaler":
        """Fit the scaler to the data."""
        self.scaler.fit(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the data using the fitted scaler."""
        return self.scaler.transform(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the transformation directly (assumes fitted)."""
        return self.transform(x)


def create_preprocessing_transform(transform_type: str, **kwargs) -> Callable:
    """
    Factory function to create preprocessing transformations.

    Args:
        transform_type: Type of transformation ('normalize', 'scale')
        **kwargs: Additional arguments for the transformation (currently unused)

    Returns:
        A preprocessing transformation object
    """
    valid_types = {"normalize", "scale"}
    if transform_type not in valid_types:
        raise ValueError(
            f"Invalid preprocessing transform type: {transform_type}. Allowed: {valid_types}"
        )

    transform_map = {
        "normalize": Normalizer(),
        "scale": StandardScaler(),
    }
    return transform_map[transform_type]
