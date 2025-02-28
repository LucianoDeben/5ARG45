# models/transcriptomics/encoders.py
import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TranscriptomicEncoder(nn.Module):
    """
    Neural network encoder for transcriptomics data.

    This encoder converts gene expression data into fixed-size representations
    using fully connected neural networks. It supports various architectures,
    normalization options, and activation functions.

    Attributes:
        input_dim: Number of input genes
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimension of the output representation
        normalize: Whether to apply batch normalization
        dropout: Dropout rate
        activation: Activation function to use
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: Optional[int] = None,
        normalize: bool = True,
        dropout: float = 0.3,
        activation: str = "relu",
        residual: bool = False,
    ):
        """
        Initialize the TranscriptomicEncoder.

        Args:
            input_dim: Number of input genes
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of the output representation (defaults to last hidden_dim)
            normalize: Whether to apply batch normalization
            dropout: Dropout rate
            activation: Activation function to use ('relu', 'leaky_relu', 'elu', etc.)
            residual: Whether to use residual connections between layers
        """
        super(TranscriptomicEncoder, self).__init__()

        # Set output dimension
        self.output_dim = output_dim if output_dim is not None else hidden_dims[-1]
        self.input_dim = input_dim
        self.normalize = normalize
        self.residual = residual

        # Choose activation function
        if activation.lower() == "relu":
            self.activation_fn = nn.ReLU()
        elif activation.lower() == "leaky_relu":
            self.activation_fn = nn.LeakyReLU(0.1)
        elif activation.lower() == "elu":
            self.activation_fn = nn.ELU()
        elif activation.lower() == "gelu":
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Build network architecture
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Add linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Add normalization if requested
            if normalize:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Add activation
            layers.append(self.activation_fn)

            # Add dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Add final projection if needed
        if self.output_dim != hidden_dims[-1]:
            layers.append(nn.Linear(hidden_dims[-1], self.output_dim))

        # Create encoder
        self.encoder = nn.Sequential(*layers)

        # Initialize residual projections if using residual connections
        if residual:
            self.residual_projections = nn.ModuleList()
            current_dim = input_dim
            for hidden_dim in hidden_dims:
                if current_dim != hidden_dim:
                    self.residual_projections.append(nn.Linear(current_dim, hidden_dim))
                else:
                    self.residual_projections.append(nn.Identity())
                current_dim = hidden_dim

        logger.debug(
            f"Initialized TranscriptomicEncoder with input_dim={input_dim}, "
            f"output_dim={self.output_dim}, normalize={normalize}, residual={residual}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for gene expression data.

        Args:
            x: Tensor [batch_size, input_dim] with gene expression values

        Returns:
            Tensor [batch_size, output_dim] with gene expression embedding
        """
        if self.residual:
            # Process with residual connections
            current_input = x
            current_output = x

            for i, layer in enumerate(self.encoder):
                # Apply layer
                current_output = layer(current_output)

                # Apply residual connection after each 'block' (linear+norm+act+dropout)
                if isinstance(layer, nn.Dropout) or (
                    isinstance(layer, nn.ReLU) and i == len(self.encoder) - 1
                ):
                    # Apply residual connection
                    proj_idx = i // 4  # Each block has 4 layers
                    if proj_idx < len(self.residual_projections):
                        residual = self.residual_projections[proj_idx](current_input)
                        current_output = current_output + residual
                        current_input = current_output

            return current_output
        else:
            # Standard processing
            return self.encoder(x)


class CNNTranscriptomicEncoder(nn.Module):
    """
    Convolutional neural network encoder for transcriptomics data.

    This encoder treats gene expression data as a 1D signal and applies
    convolutional layers to extract patterns across gene subsets. It can
    be particularly useful for capturing local patterns in gene expression.

    Attributes:
        input_dim: Number of input genes
        hidden_dims: List of hidden channel dimensions
        output_dim: Dimension of the output representation
        kernel_sizes: List of kernel sizes for convolutional layers
        normalize: Whether to apply batch normalization
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        kernel_sizes: Optional[List[int]] = None,
        normalize: bool = True,
        dropout: float = 0.3,
    ):
        """
        Initialize the CNNTranscriptomicEncoder.

        Args:
            input_dim: Number of input genes
            hidden_dims: List of hidden channel dimensions
            output_dim: Dimension of the output representation
            kernel_sizes: List of kernel sizes for each convolutional layer (default: [5, 5, 3])
            normalize: Whether to apply batch normalization
            dropout: Dropout rate
        """
        super(CNNTranscriptomicEncoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Default kernel sizes if not provided
        if kernel_sizes is None:
            kernel_sizes = [5, 5, 3]

        # Ensure we have enough kernel sizes
        if len(kernel_sizes) < len(hidden_dims):
            kernel_sizes = kernel_sizes + [3] * (len(hidden_dims) - len(kernel_sizes))

        # Build CNN architecture
        self.conv_layers = nn.ModuleList()

        # First layer: input genes -> hidden_dims[0]
        # Reshape input to [batch, 1, input_dim] to treat it as 1D signal
        self.conv_layers.append(
            nn.Conv1d(
                1,
                hidden_dims[0],
                kernel_size=kernel_sizes[0],
                padding=kernel_sizes[0] // 2,
            )
        )
        if normalize:
            self.conv_layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(nn.Dropout(dropout))

        # Additional convolutional layers
        for i in range(1, len(hidden_dims)):
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_dims[i - 1],
                    hidden_dims[i],
                    kernel_size=kernel_sizes[i],
                    padding=kernel_sizes[i] // 2,
                )
            )
            if normalize:
                self.conv_layers.append(nn.BatchNorm1d(hidden_dims[i]))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.Dropout(dropout))

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final projection to output dimension
        self.projection = nn.Linear(hidden_dims[-1], output_dim)

        logger.debug(
            f"Initialized CNNTranscriptomicEncoder with input_dim={input_dim}, "
            f"output_dim={output_dim}, conv_layers={len(hidden_dims)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for gene expression data.

        Args:
            x: Tensor [batch_size, input_dim] with gene expression values

        Returns:
            Tensor [batch_size, output_dim] with gene expression embedding
        """
        # Reshape input to [batch, 1, input_dim]
        x = x.unsqueeze(1)

        # Apply convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Global pooling
        x = self.global_pool(x).squeeze(-1)

        # Project to output dimension
        return self.projection(x)


class BiologicallyInformedEncoder(nn.Module):
    """
    Transcriptomics encoder that incorporates biological prior knowledge.

    This encoder applies optional biological pathway or gene grouping information
    to improve feature extraction from gene expression data. It's designed to
    leverage biological knowledge for more interpretable representations.

    Attributes:
        input_dim: Number of input genes
        pathway_groups: Gene groupings based on biological pathways
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimension of the output representation
    """

    def __init__(
        self,
        input_dim: int,
        pathway_groups: Dict[str, List[int]],
        hidden_dims: List[int],
        output_dim: int,
        normalize: bool = True,
        dropout: float = 0.3,
        aggregation: str = "attention",
    ):
        """
        Initialize the BiologicallyInformedEncoder.

        Args:
            input_dim: Number of input genes
            pathway_groups: Dictionary mapping pathway names to lists of gene indices
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of the output representation
            normalize: Whether to apply batch normalization
            dropout: Dropout rate
            aggregation: Method to aggregate pathway features ('attention', 'mean', 'concat')
        """
        super(BiologicallyInformedEncoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pathway_groups = pathway_groups
        self.aggregation = aggregation.lower()

        # Validate aggregation method
        valid_aggregations = ["attention", "mean", "concat"]
        if self.aggregation not in valid_aggregations:
            raise ValueError(
                f"Invalid aggregation method: {aggregation}. Must be one of {valid_aggregations}"
            )

        # Create pathway-specific encoders
        self.pathway_encoders = nn.ModuleDict()
        for pathway_name, gene_indices in pathway_groups.items():
            # Each pathway gets its own encoder
            pathway_input_dim = len(gene_indices)
            self.pathway_encoders[pathway_name] = TranscriptomicEncoder(
                input_dim=pathway_input_dim,
                hidden_dims=[hidden_dims[0] // 2],  # Smaller network for each pathway
                output_dim=hidden_dims[0] // 2,
                normalize=normalize,
                dropout=dropout,
            )

        # For attention-based aggregation
        if self.aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_dims[0] // 2, 1), nn.Softmax(dim=1)
            )
            self.merged_dim = hidden_dims[0] // 2
        elif self.aggregation == "mean":
            self.merged_dim = hidden_dims[0] // 2
        elif self.aggregation == "concat":
            self.merged_dim = (hidden_dims[0] // 2) * len(pathway_groups)

        # Global encoder for all genes
        self.global_encoder = TranscriptomicEncoder(
            input_dim=input_dim,
            hidden_dims=[hidden_dims[0]],
            output_dim=hidden_dims[0],
            normalize=normalize,
            dropout=dropout,
        )

        # Integration layer
        self.integration = nn.Sequential(
            nn.Linear(self.merged_dim + hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]) if normalize else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], output_dim),
        )

        logger.debug(
            f"Initialized BiologicallyInformedEncoder with {len(pathway_groups)} pathways, "
            f"aggregation={aggregation}, output_dim={output_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for gene expression data.

        Args:
            x: Tensor [batch_size, input_dim] with gene expression values

        Returns:
            Tensor [batch_size, output_dim] with biologically informed embedding
        """
        # Process through global encoder
        global_features = self.global_encoder(x)

        # Process through pathway-specific encoders
        pathway_features = []
        for pathway_name, gene_indices in self.pathway_groups.items():
            # Extract genes for this pathway
            pathway_input = x[:, gene_indices]
            # Process through pathway encoder
            pathway_output = self.pathway_encoders[pathway_name](pathway_input)
            pathway_features.append(pathway_output)

        # Stack pathway features
        if pathway_features:
            pathway_features = torch.stack(
                pathway_features, dim=1
            )  # [batch, n_pathways, feature_dim]

            # Aggregate pathway features
            if self.aggregation == "attention":
                # Calculate attention weights
                attn_weights = self.attention(
                    pathway_features
                )  # [batch, n_pathways, 1]
                # Apply attention
                merged_pathway_features = torch.sum(
                    pathway_features * attn_weights, dim=1
                )  # [batch, feature_dim]
            elif self.aggregation == "mean":
                merged_pathway_features = torch.mean(pathway_features, dim=1)
            elif self.aggregation == "concat":
                merged_pathway_features = pathway_features.view(
                    pathway_features.size(0), -1
                )  # [batch, n_pathways * feature_dim]
        else:
            # If no pathway features, use zeros
            batch_size = x.size(0)
            merged_pathway_features = torch.zeros(
                batch_size, self.merged_dim, device=x.device
            )

        # Combine global and pathway features
        combined = torch.cat([global_features, merged_pathway_features], dim=1)

        # Process through integration layer
        return self.integration(combined)
