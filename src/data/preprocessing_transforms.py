# src/data/preprocessing_transforms.py
"""
Preprocessing transformations for normalizing and scaling data.
"""
import logging
from typing import Callable, List, Union

import numpy as np
from sklearn.preprocessing import Normalizer as SklearnNormalizer
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
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

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit the normalizer to the data and transform it."""
        return self.scaler.fit_transform(x)

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

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit the scaler to the data and transform it."""
        return self.scaler.fit_transform(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the transformation directly (assumes fitted)."""
        return self.transform(x)


class RobustScaler:
    """Scales features using statistics that are robust to outliers."""

    def __init__(self, quantile_range: tuple = (25.0, 75.0)):
        """Initialize with a RobustScaler from sklearn."""
        self.scaler = SklearnRobustScaler(quantile_range=quantile_range)

    def fit(self, x: np.ndarray) -> "RobustScaler":
        """Fit the scaler to the data."""
        self.scaler.fit(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the data using the fitted scaler."""
        return self.scaler.transform(x)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit the scaler to the data and transform it."""
        return self.scaler.fit_transform(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the transformation directly (assumes fitted)."""
        return self.transform(x)


class CompoundTransform:
    """Chain multiple preprocessing transformations in sequence."""

    def __init__(self, transforms: List[Callable]):
        """Initialize with a list of transformations to apply in sequence."""
        self.transforms = transforms

    def fit(self, x: np.ndarray) -> "CompoundTransform":
        """Fit each transformation in sequence."""
        data = x.copy()
        for transform in self.transforms:
            if hasattr(transform, "fit"):
                transform.fit(data)
                if hasattr(transform, "transform"):
                    data = transform.transform(data)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply each transformation in sequence."""
        data = x.copy()
        for transform in self.transforms:
            if hasattr(transform, "transform"):
                data = transform.transform(data)
            else:
                data = transform(data)
        return data

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        data = x.copy()
        for transform in self.transforms:
            if hasattr(transform, "fit_transform"):
                data = transform.fit_transform(data)
            elif hasattr(transform, "fit") and hasattr(transform, "transform"):
                transform.fit(data)
                data = transform.transform(data)
            else:
                data = transform(data)
        return data

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the transformations directly (assumes fitted)."""
        return self.transform(x)


def create_preprocessing_transform(transform_type: str, **kwargs) -> Callable:
    """
    Factory function to create preprocessing transformations.

    Args:
        transform_type: Type of transformation ('normalize', 'scale', 'robust', 'compound')
        **kwargs: Additional arguments for the transformation

    Returns:
        A preprocessing transformation object
    """
    valid_types = {"normalize", "scale", "robust", "compound"}
    if transform_type not in valid_types:
        raise ValueError(
            f"Invalid preprocessing transform type: {transform_type}. Allowed: {valid_types}"
        )

    if transform_type == "compound":
        transform_list = []
        for transform_spec in kwargs.get("transforms", []):
            transform_name = transform_spec.get("type")
            transform_args = {k: v for k, v in transform_spec.items() if k != "type"}
            transform_list.append(
                create_preprocessing_transform(transform_name, **transform_args)
            )
        return CompoundTransform(transform_list)

    transform_map = {
        "normalize": Normalizer(),
        "scale": StandardScaler(),
        "robust": RobustScaler(
            quantile_range=kwargs.get("quantile_range", (25.0, 75.0))
        ),
    }
    return transform_map[transform_type]
