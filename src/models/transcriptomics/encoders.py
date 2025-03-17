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


class AttentionTranscriptomicEncoder(nn.Module):
    """
    Self-attention based encoder for transcriptomics data.

    This encoder uses self-attention mechanisms to capture global relationships
    between genes in expression data, allowing it to model complex interactions
    and dependencies that might be missed by MLP or CNN architectures.

    Attributes:
        input_dim: Number of input genes
        hidden_dim: Dimension of hidden representations
        output_dim: Dimension of the output representation
        num_heads: Number of attention heads
        num_layers: Number of transformer encoder layers
        normalize: Whether to apply layer normalization
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        normalize: bool = True,
        dropout: float = 0.2,
        feed_forward_dim: Optional[int] = None,
    ):
        """
        Initialize the AttentionTranscriptomicEncoder.

        Args:
            input_dim: Number of input genes
            hidden_dim: Dimension of hidden representations
            output_dim: Dimension of the output representation
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            normalize: Whether to apply layer normalization
            dropout: Dropout rate
            feed_forward_dim: Dimension of feed-forward network (defaults to 4*hidden_dim)
        """
        super(AttentionTranscriptomicEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        if feed_forward_dim is None:
            feed_forward_dim = 4 * hidden_dim

        # Initial projection of gene expression to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Create transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feed_forward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=normalize,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Final projection to output dimension
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # Learned positions for genes (as we don't have sequential order)
        self.position_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))

        logger.debug(
            f"Initialized AttentionTranscriptomicEncoder with input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, output_dim={output_dim}, "
            f"num_heads={num_heads}, num_layers={num_layers}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for gene expression data.

        This implementation treats each gene as a "token" in the transformer framework,
        allowing the self-attention mechanism to model relationships between genes.

        Args:
            x: Tensor [batch_size, input_dim] with gene expression values

        Returns:
            Tensor [batch_size, output_dim] with gene expression embedding
        """
        batch_size = x.shape[0]

        # Reshape for transformer: [batch_size, input_dim] -> [batch_size, input_dim, 1]
        # This treats each gene as a "token" with a single feature
        x = x.unsqueeze(-1)

        # Project each gene to hidden dimension: [batch_size, input_dim, hidden_dim]
        x = self.input_projection(x)

        # Add positional embedding
        x = x + self.position_embedding

        # Pass through transformer encoder
        # Shape remains [batch_size, input_dim, hidden_dim]
        x = self.transformer_encoder(x)

        # Global pooling across genes
        x = x.mean(dim=1)  # [batch_size, hidden_dim]

        # Project to output dimension
        x = self.output_projection(x)  # [batch_size, output_dim]

        return x


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling for aggregating gene features.

    This module applies multi-head attention with a learnable query to
    aggregate gene-level features into a single representation, giving
    different weights to different genes based on their importance.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize the multi-head attention pooling.

        Args:
            hidden_dim: Dimension of gene representations
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttentionPooling, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Use multi-head attention with a learnable query
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling to gene expressions.

        Args:
            x: Tensor [batch_size, num_genes, hidden_dim] with gene representations

        Returns:
            Tensor [batch_size, hidden_dim] with aggregated gene representation
        """
        batch_size = x.shape[0]

        # Expand query to batch size
        query = self.query.expand(batch_size, -1, -1)

        # Apply attention (query = [batch_size, 1, hidden_dim],
        #                 key/value = [batch_size, num_genes, hidden_dim])
        # Output = [batch_size, 1, hidden_dim]
        output, _ = self.attention(query, x, x)

        # Remove singleton dimension
        output = output.squeeze(1)

        return output


class HierarchicalTranscriptomicEncoder(nn.Module):
    """
    Hierarchical encoder for transcriptomics data using pathway/gene-set organization.

    This encoder organizes genes into biologically meaningful pathways or gene-sets,
    processes each set with a pathway-specific encoder, and then aggregates the
    pathway representations using self-attention mechanisms. This can improve
    interpretability and potentially performance by leveraging known biological structures.
    """

    def __init__(
        self,
        input_dim: int,
        pathway_groups: Dict[str, List[int]],  # Maps pathway names to gene indices
        pathway_hidden_dim: int = 64,
        global_hidden_dim: int = 128,
        output_dim: int = 256,
        pathway_encoder_type: str = "attention",
        global_num_heads: int = 4,
        dropout: float = 0.3,
    ):
        """
        Initialize the HierarchicalTranscriptomicEncoder.

        Args:
            input_dim: Number of input genes
            pathway_groups: Dictionary mapping pathway names to lists of gene indices
            pathway_hidden_dim: Hidden dimension for pathway-specific encoders
            global_hidden_dim: Hidden dimension for global integration
            output_dim: Dimension of the output representation
            pathway_encoder_type: Type of encoder for each pathway ('mlp', 'cnn', 'attention')
            global_num_heads: Number of attention heads for global integration
            dropout: Dropout rate
        """
        super(HierarchicalTranscriptomicEncoder, self).__init__()

        self.input_dim = input_dim
        self.pathway_groups = pathway_groups
        self.output_dim = output_dim

        # Create encoders for each pathway
        self.pathway_encoders = nn.ModuleDict()

        for pathway_name, gene_indices in pathway_groups.items():
            pathway_input_dim = len(gene_indices)

            if pathway_encoder_type == "mlp":
                encoder = TranscriptomicEncoder(
                    input_dim=pathway_input_dim,
                    hidden_dims=[pathway_hidden_dim],
                    output_dim=pathway_hidden_dim,
                    normalize=True,
                    dropout=dropout,
                )
            elif pathway_encoder_type == "cnn":
                encoder = CNNTranscriptomicEncoder(
                    input_dim=pathway_input_dim,
                    hidden_dims=[pathway_hidden_dim, pathway_hidden_dim],
                    output_dim=pathway_hidden_dim,
                    normalize=True,
                    dropout=dropout,
                )
            elif pathway_encoder_type == "attention":
                encoder = AttentionTranscriptomicEncoder(
                    input_dim=pathway_input_dim,
                    hidden_dim=pathway_hidden_dim,
                    output_dim=pathway_hidden_dim,
                    num_heads=2,
                    num_layers=1,
                    dropout=dropout,
                )
            else:
                raise ValueError(
                    f"Unsupported pathway encoder type: {pathway_encoder_type}"
                )

            self.pathway_encoders[pathway_name] = encoder

        # Global integration of pathway representations
        self.num_pathways = len(pathway_groups)

        # Initial projection to ensure uniform dimension
        self.pathway_projection = nn.Linear(pathway_hidden_dim, global_hidden_dim)

        # Global integration via self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=global_hidden_dim,
            nhead=global_num_heads,
            dim_feedforward=global_hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.global_integration = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Attention-based pooling
        self.attention_pooling = MultiHeadAttentionPooling(
            hidden_dim=global_hidden_dim, num_heads=global_num_heads, dropout=dropout
        )

        # Final projection
        self.output_projection = nn.Linear(global_hidden_dim, output_dim)

        logger.debug(
            f"Initialized HierarchicalTranscriptomicEncoder with {self.num_pathways} pathways, "
            f"pathway_hidden_dim={pathway_hidden_dim}, global_hidden_dim={global_hidden_dim}, "
            f"output_dim={output_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for gene expression data.

        Args:
            x: Tensor [batch_size, input_dim] with gene expression values

        Returns:
            Tensor [batch_size, output_dim] with gene expression embedding
        """
        batch_size = x.shape[0]

        # Process each pathway
        pathway_embeddings = []

        for pathway_name, gene_indices in self.pathway_groups.items():
            # Extract genes for this pathway
            pathway_input = x[:, gene_indices]

            # Process with pathway-specific encoder
            pathway_embedding = self.pathway_encoders[pathway_name](pathway_input)
            pathway_embeddings.append(pathway_embedding)

        # Stack pathway embeddings [batch_size, num_pathways, pathway_hidden_dim]
        pathway_embeddings = torch.stack(pathway_embeddings, dim=1)

        # Project to global hidden dimension
        pathway_embeddings = self.pathway_projection(pathway_embeddings)

        # Global integration with self-attention
        global_embeddings = self.global_integration(pathway_embeddings)

        # Attention-based pooling
        pooled_embedding = self.attention_pooling(global_embeddings)

        # Final projection
        output = self.output_projection(pooled_embedding)

        return output

def create_transcriptomic_encoder(
    encoder_type: str,
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of transcriptomic encoders.
    
    Args:
        encoder_type: Type of encoder ("mlp", "cnn", "attention", "hierarchical")
        input_dim: Number of input genes
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimension of output features
        **kwargs: Additional arguments specific to encoder types
    
    Returns:
        A transcriptomic encoder module
    """
    if encoder_type == "mlp":
        return TranscriptomicEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            normalize=kwargs.get("normalize", True),
            dropout=kwargs.get("dropout", 0.3),
            activation=kwargs.get("activation", "relu"),
            residual=kwargs.get("residual", False)
        )
    elif encoder_type == "cnn":
        return CNNTranscriptomicEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            kernel_sizes=kwargs.get("kernel_sizes"),
            normalize=kwargs.get("normalize", True),
            dropout=kwargs.get("dropout", 0.3)
        )
    elif encoder_type == "attention":
        return AttentionTranscriptomicEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dims[0] if hidden_dims else 128,
            output_dim=output_dim,
            num_heads=kwargs.get("num_heads", 4),
            num_layers=kwargs.get("num_layers", 2),
            normalize=kwargs.get("normalize", True),
            dropout=kwargs.get("dropout", 0.2),
            feed_forward_dim=kwargs.get("feed_forward_dim")
        )
    elif encoder_type == "hierarchical":
        # This requires pathway_groups which is a critical parameter
        if "pathway_groups" not in kwargs:
            raise ValueError("pathway_groups is required for hierarchical encoder")
            
        return HierarchicalTranscriptomicEncoder(
            input_dim=input_dim,
            pathway_groups=kwargs["pathway_groups"],
            pathway_hidden_dim=kwargs.get("pathway_hidden_dim", 64),
            global_hidden_dim=kwargs.get("global_hidden_dim", 128),
            output_dim=output_dim,
            pathway_encoder_type=kwargs.get("pathway_encoder_type", "attention"),
            global_num_heads=kwargs.get("global_num_heads", 4),
            dropout=kwargs.get("dropout", 0.3)
        )
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")