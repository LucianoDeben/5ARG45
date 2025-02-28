# models/chemical/descriptors.py

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MolecularDescriptorEncoder(nn.Module):
    """
    Process pre-calculated molecular descriptors for drug response prediction.

    This module encodes molecular descriptors (like molecular weight, logP, etc.)
    into a fixed-size representation that can be combined with transcriptomics data.
    It supports optional normalization and feature selection.

    Attributes:
        input_dim: Number of input molecular descriptors
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimension of the final representation
        normalize: Whether to apply batch normalization
        dropout: Dropout rate
        activation: Activation function to use
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        normalize: bool = True,
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        """
        Initialize the MolecularDescriptorEncoder.

        Args:
            input_dim: Number of input molecular descriptors
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of the final representation
            normalize: Whether to apply batch normalization
            dropout: Dropout rate
            activation: Activation function to use ('relu', 'leaky_relu', 'elu', etc.)
        """
        super(MolecularDescriptorEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.normalize = normalize
        self.dropout = dropout

        # Choose activation function
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation.lower() == "elu":
            self.activation = nn.ELU()
        elif activation.lower() == "selu":
            self.activation = nn.SELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Build network layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if normalize:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        logger.debug(
            f"Initialized MolecularDescriptorEncoder with input_dim={input_dim}, "
            f"output_dim={output_dim}, normalize={normalize}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for molecular descriptors.

        Args:
            x: Tensor [batch_size, input_dim] with molecular descriptors

        Returns:
            Tensor [batch_size, output_dim] with molecular embedding
        """
        return self.network(x)


class IntegratedMolecularEncoder(nn.Module):
    """
    Integrated encoder that combines molecular descriptors with dosage information.

    This module processes molecular descriptors alongside dosage information
    to create a comprehensive molecular representation.

    Attributes:
        descriptor_encoder: The MolecularDescriptorEncoder module
        dosage_encoder: The encoder for dosage information
        output_dim: Dimension of the final representation
    """

    def __init__(
        self,
        descriptor_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dosage_integration: str = "concat",
        normalize: bool = True,
        dropout: float = 0.3,
    ):
        """
        Initialize the IntegratedMolecularEncoder.

        Args:
            descriptor_dim: Number of input molecular descriptors
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of the final representation
            dosage_integration: How to integrate dosage ('concat', 'feature')
            normalize: Whether to apply batch normalization
            dropout: Dropout rate
        """
        super(IntegratedMolecularEncoder, self).__init__()
        self.dosage_integration = dosage_integration.lower()

        if self.dosage_integration == "concat":
            # Concatenate dosage to descriptors before encoding
            self.descriptor_encoder = MolecularDescriptorEncoder(
                input_dim=descriptor_dim + 1,  # +1 for dosage
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                normalize=normalize,
                dropout=dropout,
            )
        elif self.dosage_integration == "feature":
            # Process descriptors first, then integrate dosage
            self.descriptor_encoder = MolecularDescriptorEncoder(
                input_dim=descriptor_dim,
                hidden_dims=hidden_dims,
                output_dim=hidden_dims[-1],  # Intermediate output
                normalize=normalize,
                dropout=dropout,
            )
            # Integrate dosage and project to final dimension
            self.dosage_layer = nn.Linear(1, hidden_dims[-1])
            self.output_projection = nn.Linear(hidden_dims[-1], output_dim)
        else:
            raise ValueError(f"Invalid dosage integration: {dosage_integration}")

        self.output_dim = output_dim
        logger.debug(
            f"Initialized IntegratedMolecularEncoder with dosage_integration={dosage_integration}, "
            f"output_dim={output_dim}"
        )

    def forward(self, descriptors: torch.Tensor, dosage: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for molecular descriptors with dosage.

        Args:
            descriptors: Tensor [batch_size, descriptor_dim] with molecular descriptors
            dosage: Tensor [batch_size, 1] with dosage information

        Returns:
            Tensor [batch_size, output_dim] with molecular embedding
        """
        if self.dosage_integration == "concat":
            # Concatenate dosage with descriptors
            x = torch.cat([descriptors, dosage], dim=1)
            return self.descriptor_encoder(x)
        elif self.dosage_integration == "feature":
            # Process descriptors
            x = self.descriptor_encoder(descriptors)
            # Process and integrate dosage
            d = F.relu(self.dosage_layer(dosage))
            # Multiply for feature-wise scaling
            x = x * torch.sigmoid(d)
            return self.output_projection(x)
