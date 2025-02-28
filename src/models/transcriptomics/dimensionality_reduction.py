import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class TranscriptomicDimensionalityReducer(nn.Module):
    """Neural autoencoder for dimensionality reduction of transcriptomics data."""

    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.1):
        super(TranscriptomicDimensionalityReducer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, input_dim),
        )
        logger.debug(
            f"Initialized TranscriptomicDimensionalityReducer with latent dim {latent_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


def pca_reducer(data: np.ndarray, n_components: int) -> np.ndarray:
    """Apply PCA for dimensionality reduction."""
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)
    logger.debug(
        f"Applied PCA, explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}"
    )
    return reduced
