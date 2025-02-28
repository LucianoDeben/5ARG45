# models/integration/attention.py
import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between transcriptomics and chemical features.

    This module implements a transformer-style attention mechanism to allow
    one modality to attend to the other, creating integrated representations
    that capture interactions between modalities.

    Attributes:
        input_dim: Dimension of input features (will be projected)
        hidden_dim: Dimension of attention space
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        transcriptomics_dim: int,
        chemical_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_projection: bool = True,
    ):
        """
        Initialize the CrossModalAttention module.

        Args:
            transcriptomics_dim: Dimension of transcriptomics features
            chemical_dim: Dimension of chemical features
            hidden_dim: Dimension of attention space
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_projection: Whether to project inputs to hidden_dim
        """
        super(CrossModalAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Ensure hidden_dim is divisible by num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"Hidden dimension ({hidden_dim}) must be divisible by number of heads ({num_heads})"
            )

        # Input projections if dimensions don't match
        self.use_projection = use_projection
        if use_projection:
            self.t_projection = nn.Linear(transcriptomics_dim, hidden_dim)
            self.c_projection = nn.Linear(chemical_dim, hidden_dim)
        else:
            if transcriptomics_dim != hidden_dim or chemical_dim != hidden_dim:
                raise ValueError(
                    f"When use_projection=False, dimensions must match. "
                    f"Got transcriptomics_dim={transcriptomics_dim}, chemical_dim={chemical_dim}, "
                    f"hidden_dim={hidden_dim}"
                )

        # Multi-head attention components
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Normalization and dropout
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        logger.debug(
            f"Initialized CrossModalAttention with hidden_dim={hidden_dim}, "
            f"num_heads={num_heads}, use_projection={use_projection}"
        )

    def forward(
        self,
        transcriptomics: torch.Tensor,
        chemicals: torch.Tensor,
        direction: str = "both",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply cross-modal attention.

        Args:
            transcriptomics: Tensor [batch_size, transcriptomics_dim] with transcriptomics features
            chemicals: Tensor [batch_size, chemical_dim] with chemical features
            direction: Attention direction ('t2c' for transcriptomics→chemical,
                      'c2t' for chemical→transcriptomics, or 'both' for bidirectional)

        Returns:
            If direction is 'both', returns a tuple of (updated_transcriptomics, updated_chemicals)
            Otherwise, returns the updated features for the target modality
        """
        batch_size = transcriptomics.size(0)

        # Project inputs to hidden_dim if needed
        if self.use_projection:
            t_hidden = self.t_projection(transcriptomics)
            c_hidden = self.c_projection(chemicals)
        else:
            t_hidden = transcriptomics
            c_hidden = chemicals

        # Reshape for multi-head attention: [batch_size, num_heads, head_dim]
        def reshape_for_multihead(x):
            return x.view(batch_size, self.num_heads, self.head_dim)

        # Process based on direction
        if direction in ["c2t", "both"]:
            # Chemical → Transcriptomics attention
            q_t = reshape_for_multihead(self.q_proj(t_hidden))
            k_c = reshape_for_multihead(self.k_proj(c_hidden))
            v_c = reshape_for_multihead(self.v_proj(c_hidden))

            # Attention scores
            attn_scores = torch.bmm(q_t, k_c.transpose(1, 2)) * self.scale
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            # Apply attention
            c2t_context = torch.bmm(attn_probs, v_c)
            c2t_context = c2t_context.contiguous().view(batch_size, self.hidden_dim)
            c2t_context = self.output_proj(c2t_context)

            # Add & norm
            t_hidden_updated = self.layer_norm1(t_hidden + self.dropout(c2t_context))

            # FFN
            t_output = self.layer_norm2(t_hidden_updated + self.ffn(t_hidden_updated))

        if direction in ["t2c", "both"]:
            # Transcriptomics → Chemical attention
            q_c = reshape_for_multihead(self.q_proj(c_hidden))
            k_t = reshape_for_multihead(self.k_proj(t_hidden))
            v_t = reshape_for_multihead(self.v_proj(t_hidden))

            # Attention scores
            attn_scores = torch.bmm(q_c, k_t.transpose(1, 2)) * self.scale
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            # Apply attention
            t2c_context = torch.bmm(attn_probs, v_t)
            t2c_context = t2c_context.contiguous().view(batch_size, self.hidden_dim)
            t2c_context = self.output_proj(t2c_context)

            # Add & norm
            c_hidden_updated = self.layer_norm1(c_hidden + self.dropout(t2c_context))

            # FFN
            c_output = self.layer_norm2(c_hidden_updated + self.ffn(c_hidden_updated))

        # Return based on direction
        if direction == "c2t":
            return t_output
        elif direction == "t2c":
            return c_output
        else:  # both
            return t_output, c_output


class ModalityAttentionFusion(nn.Module):
    """
    Fusion module using self-attention across modalities.

    This module creates a joint representation by treating different modalities
    as tokens in a sequence, applying self-attention, and aggregating the results.

    Attributes:
        modality_dims: Dictionary mapping modality names to their dimensions
        hidden_dim: Dimension of attention space
        output_dim: Dimension of output representation
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        aggregation: str = "mean",
    ):
        """
        Initialize the ModalityAttentionFusion module.

        Args:
            modality_dims: Dictionary mapping modality names to their dimensions
            hidden_dim: Dimension of attention space
            output_dim: Dimension of output representation
            num_heads: Number of attention heads
            dropout: Dropout rate
            aggregation: How to aggregate modality tokens ('mean', 'sum', 'cls')
        """
        super(ModalityAttentionFusion, self).__init__()

        self.modality_names = list(modality_dims.keys())
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.aggregation = aggregation.lower()

        # Validate aggregation method
        valid_aggregations = ["mean", "sum", "cls"]
        if self.aggregation not in valid_aggregations:
            raise ValueError(
                f"Invalid aggregation: {aggregation}. Must be one of {valid_aggregations}"
            )

        # Input projections for each modality
        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, hidden_dim) for name, dim in modality_dims.items()}
        )

        # Add CLS token embedding if using 'cls' aggregation
        if self.aggregation == "cls":
            self.cls_embedding = nn.Parameter(torch.randn(1, hidden_dim))

        # Position embeddings (one for each modality)
        self.pos_embeddings = nn.Parameter(
            torch.randn(
                len(modality_dims) + (1 if self.aggregation == "cls" else 0), hidden_dim
            )
        )

        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization and dropout
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        logger.debug(
            f"Initialized ModalityAttentionFusion with {len(modality_dims)} modalities, "
            f"hidden_dim={hidden_dim}, output_dim={output_dim}, aggregation={aggregation}"
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse multiple modalities using attention.

        Args:
            features: Dictionary mapping modality names to their feature tensors
                     Each tensor should be [batch_size, modality_dim]

        Returns:
            Tensor [batch_size, output_dim] with fused representation
        """
        batch_size = next(iter(features.values())).size(0)

        # Check if all required modalities are present
        missing_modalities = set(self.modality_names) - set(features.keys())
        if missing_modalities:
            raise ValueError(f"Missing modalities: {missing_modalities}")

        # Project each modality to hidden_dim
        projected_features = []
        for name in self.modality_names:
            projected = self.projections[name](features[name])
            projected_features.append(projected)

        # Add CLS token if using 'cls' aggregation
        if self.aggregation == "cls":
            cls_tokens = self.cls_embedding.expand(batch_size, 1, self.hidden_dim)
            projected_features = [cls_tokens] + projected_features

        # Stack modalities into a sequence: [batch_size, num_modalities, hidden_dim]
        sequence = torch.stack(projected_features, dim=1)

        # Add position embeddings
        sequence = sequence + self.pos_embeddings.unsqueeze(0)

        # Apply self-attention
        attn_output, _ = self.attention(sequence, sequence, sequence)
        attn_output = self.dropout(attn_output)
        sequence = self.layer_norm1(sequence + attn_output)

        # Apply feed-forward network
        ffn_output = self.ffn(sequence)
        sequence = self.layer_norm2(sequence + ffn_output)

        # Aggregate sequence based on specified method
        if self.aggregation == "mean":
            fused = torch.mean(sequence, dim=1)
        elif self.aggregation == "sum":
            fused = torch.sum(sequence, dim=1)
        elif self.aggregation == "cls":
            fused = sequence[:, 0]  # Take CLS token

        # Project to output dimension
        return self.output_proj(fused)
