# src/data/preprocessing_transforms.py
"""
Preprocessing transformations for normalizing and scaling data.
"""
import logging
from typing import Callable, Dict, List, Union

import torch
import decoupler as dc
import numpy as np
import pandas as pd
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

class TFInferenceTransform:
    """
    Transform that performs transcription factor activity inference using decoupler.
    
    This transform converts gene expression data to TF activity scores based on
    a provided regulatory network.
    """
    
    def __init__(
        self, 
        network: pd.DataFrame, 
        method: Union[str, List[str]] = "ulm",
        min_n: int = 10,
        use_raw: bool = False,
        consensus: bool = False,
        cache_results: bool = True
    ):
        """
        Initialize the TF inference transform.
        
        Args:
            network: DataFrame containing the TF regulatory network (e.g., TF-gene interactions)
            method: Inference method(s) to use, e.g., "ulm", "mlm", or a list like ["ulm", "mlm"]
            min_n: Minimum number of targets per TF
            use_raw: Whether to use raw counts
            consensus: Whether to return consensus scores across methods
            cache_results: Whether to cache results to avoid recomputing for the same data
        """
        self.network = network
        self.method = method
        self.min_n = min_n
        self.use_raw = use_raw
        self.consensus = consensus
        self.cache_results = cache_results
        
        # For caching results to avoid redundant computation
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Methods needs to be a list for decoupler
        self.methods = [method] if isinstance(method, str) else method
        
        logger.info(f"Initialized TF inference transform with method(s): {self.methods}")
    
    def _get_cache_key(self, X):
        """Generate a cache key based on the data matrix."""
        if not self.cache_results:
            return None
        # Use shape and a hash of the first and last rows to create a lightweight cache key
        try:
            key = (X.shape, hash(str(X[0].tobytes()) + str(X[-1].tobytes())))
            return key
        except:
            return None
    
    def fit(self, X: np.ndarray, gene_symbols: List[str] = None):
        """
        Fit is a no-op for this transform since inference doesn't require fitting.
        
        Args:
            X: Gene expression matrix (n_samples, n_genes)
            gene_symbols: List of gene symbols corresponding to columns in X
        
        Returns:
            self
        """
        return self
    
    def transform(self, X: np.ndarray, gene_symbols: List[str]) -> np.ndarray:
        """
        Transform gene expression data to TF activity scores.
        
        Args:
            X: Gene expression matrix (n_samples, n_genes)
            gene_symbols: List of gene symbols corresponding to columns in X
        
        Returns:
            TF activity scores matrix (n_samples, n_TFs)
        """
        # Check if we have seen this exact data before
        cache_key = self._get_cache_key(X)
        if cache_key and cache_key in self._cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit for TF inference (hits: {self._cache_hits}, misses: {self._cache_misses})")
            return self._cache[cache_key]
        
        self._cache_misses += 1
        
        # Convert to pandas DataFrame with gene symbols as columns
        X_df = pd.DataFrame(X, columns=gene_symbols)
        
        try:
            # Run decoupler
            res = dc.decouple(
                X_df, 
                self.network, 
                methods=self.methods, 
                consensus=self.consensus,
                min_n=self.min_n, 
                use_raw=self.use_raw
            )
            
            # Convert back to numpy array (extract TF activity matrix)
            tf_activities = dc.cons(res)
            
            # Get activity scores as numpy array
            result = tf_activities.values
            
            # Cache the result
            if cache_key:
                self._cache[cache_key] = result
            
            # Return numpy array of TF activities
            return result
            
        except Exception as e:
            logger.error(f"Error in TF inference: {str(e)}")
            # Return original data if inference fails
            return X
    
    def fit_transform(self, X: np.ndarray, gene_symbols: List[str]) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Gene expression matrix (n_samples, n_genes)
            gene_symbols: List of gene symbols corresponding to columns in X
        
        Returns:
            TF activity scores matrix (n_samples, n_TFs)
        """
        return self.transform(X, gene_symbols)
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Transform a batch of data from a dataset.
        
        This special implementation is designed to work with our Dataset __getitem__ method.
        
        Args:
            data: Dictionary containing 'transcriptomics' tensor and 'gene_symbols' list
            
        Returns:
            Dictionary with 'transcriptomics' replaced by TF activity scores
        """
        # Extract transcriptomics data and gene symbols
        X = data['transcriptomics'].numpy()
        gene_symbols = data['gene_symbols']
        
        # Apply the transform
        if len(X.shape) == 1:  # Single sample
            X = X.reshape(1, -1)
            tf_activities = self.transform(X, gene_symbols)
            
            # Store TF names before reshaping
            # We need to run decouple again to get column names since they're not returned by transform
            X_df = pd.DataFrame(X, columns=gene_symbols)
            res = dc.decouple(
                X_df, 
                self.network, 
                methods=self.methods, 
                consensus=self.consensus,
                min_n=self.min_n, 
                use_raw=self.use_raw
            )
            tf_names = list(dc.cons(res).columns)
            
            # Reshape for a single sample
            tf_activities = tf_activities.reshape(-1)
        else:  # Batch of samples
            tf_activities = self.transform(X, gene_symbols)
            
            # Get TF names
            X_df = pd.DataFrame(X[0:1], columns=gene_symbols)  # Just use first sample to get names
            res = dc.decouple(
                X_df, 
                self.network, 
                methods=self.methods, 
                consensus=self.consensus,
                min_n=self.min_n, 
                use_raw=self.use_raw
            )
            tf_names = list(dc.cons(res).columns)
        
        # Create a new dictionary with transformed data
        new_data = data.copy()
        new_data['transcriptomics'] = torch.tensor(tf_activities, dtype=torch.float32)
        new_data['tf_names'] = tf_names
        
        return new_data
    
    def get_cache_stats(self):
        """Get statistics about cache performance."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache)
        }

class TransformPipeline:
    """Chain multiple preprocessing transformations in sequence to form a pipeline."""

    def __init__(self, transforms: List[Callable]):
        """
        Initialize with a list of transformations to apply in sequence.
        
        Args:
            transforms: List of transformation objects to apply in order
        """
        self.transforms = transforms

    def _copy_data(self, x):
        """Create a copy of the data, handling both numpy arrays and torch tensors."""
        if isinstance(x, np.ndarray):
            return x.copy()
        elif isinstance(x, torch.Tensor):
            return x.clone()
        elif isinstance(x, dict):
            return x.copy()
        return x  # For other types, return as is

    def fit(self, x) -> "TransformPipeline":
        """Fit each transformation in sequence."""
        data = self._copy_data(x)
        for transform in self.transforms:
            if hasattr(transform, "fit"):
                transform.fit(data)
                if hasattr(transform, "transform"):
                    data = transform.transform(data)
        return self

    def transform(self, x):
        """
        Apply each transformation in sequence.
        
        For dictionary inputs needed by TFInferenceTransform, this will use __call__ method.
        """
        data = self._copy_data(x)
        
        # Track if we're working with a dictionary from a dataset
        is_dataset_dict = isinstance(data, dict) and 'transcriptomics' in data
        
        for transform in self.transforms:
            if is_dataset_dict and hasattr(transform, "__call__"):
                # For dataset dictionaries, use __call__ to handle gene_symbols
                data = transform(data)
            elif hasattr(transform, "transform") and not is_dataset_dict:
                # For direct tensors/arrays, use transform method
                data = transform.transform(data)
            else:
                # Fallback to __call__
                data = transform(data)
        
        return data

    def fit_transform(self, x):
        """Fit and transform in one step."""
        self.fit(x)
        return self.transform(x)

    def __call__(self, x):
        """Apply the transformations directly."""
        return self.transform(x)


def create_preprocessing_transform(transform_type: str, **kwargs) -> Callable:
    """
    Factory function to create preprocessing transformations.

    Args:
        transform_type: Type of transformation ('normalize', 'zscore', 'robust', 'pipeline', 'none', 'tf_inference')
        **kwargs: Additional arguments for the transformation

    Returns:
        A preprocessing transformation object
    """
    valid_types = {"normalize", "zscore", "robust", "pipeline", "none", "tf_inference"}
    if transform_type not in valid_types:
        raise ValueError(
            f"Invalid preprocessing transform type: {transform_type}. Allowed: {valid_types}"
        )

    if transform_type == "pipeline":
        transform_list = []
        for transform_spec in kwargs.get("transforms", []):
            transform_name = transform_spec.get("type")
            transform_args = {k: v for k, v in transform_spec.items() if k != "type"}
            transform_list.append(
                create_preprocessing_transform(transform_name, **transform_args)
            )
        return TransformPipeline(transform_list)
    
    if transform_type == "tf_inference":
        if "network" not in kwargs:
            raise ValueError("Network DataFrame is required for tf_inference transform")
        return TFInferenceTransform(
            network=kwargs.get("network"),
            method=kwargs.get("method", "ulm"),
            min_n=kwargs.get("min_n", 10),
            use_raw=kwargs.get("use_raw", False),
            consensus=kwargs.get("consensus", False),
            cache_results=kwargs.get("cache_results", True)
        )

    transform_map = {
        "normalize": Normalizer(),
        "zscore": StandardScaler(),
        "robust": RobustScaler(
            quantile_range=kwargs.get("quantile_range", (25.0, 75.0))
        ),
        "none": None,
    }
    return transform_map[transform_type]
