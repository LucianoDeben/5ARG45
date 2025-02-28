# models/integration/fusion.py
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FeatureFusion(nn.Module):
    """
    Fuse transcriptomics and chemical features using various strategies.

    This module implements multiple fusion strategies to combine features
    from different modalities into a single representation.

    Attributes:
        strategy: Fusion strategy to use
        t_dim: Dimension of transcriptomics features
        c_dim: Dimension of chemical features
        output_dim: Dimension of fused representation
    """

    def __init__(
        self,
        t_dim: int,
        c_dim: int,
        output_dim: Optional[int] = None,
        strategy: str = "concat",
        projection: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize the FeatureFusion module.

        Args:
            t_dim: Dimension of transcriptomics features
            c_dim: Dimension of chemical features
            output_dim: Dimension of output features (defaults to t_dim + c_dim for concat)
            strategy: Fusion strategy ('concat', 'add', 'multiply', 'gated', 'film', 'bilinear')
            projection: Whether to project the fused representation
            dropout: Dropout rate for projection layers
        """
        super(FeatureFusion, self).__init__()

        self.strategy = strategy.lower()
        self.t_dim = t_dim
        self.c_dim = c_dim
        self.projection = projection

        # Set default output dimension based on strategy
        if output_dim is None:
            if strategy == "concat":
                output_dim = t_dim + c_dim
            else:
                output_dim = max(t_dim, c_dim)

        self.output_dim = output_dim

        # Validate strategy
        valid_strategies = ["concat", "add", "multiply", "gated", "film", "bilinear"]
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Unknown fusion strategy: {self.strategy}. Must be one of {valid_strategies}"
            )

        # Create dimension-matching projections if needed
        if self.strategy == "add" or self.strategy == "multiply":
            if t_dim != c_dim:
                self.t_match = nn.Linear(t_dim, max(t_dim, c_dim))
                self.c_match = nn.Linear(c_dim, max(t_dim, c_dim))
            else:
                self.t_match = nn.Identity()
                self.c_match = nn.Identity()

        # Strategy-specific layers
        if self.strategy == "gated":
            self.gate = nn.Sequential(nn.Linear(t_dim + c_dim, c_dim), nn.Sigmoid())
        elif self.strategy == "film":
            # FiLM (Feature-wise Linear Modulation)
            self.gamma = nn.Linear(c_dim, t_dim)  # Scale
            self.beta = nn.Linear(c_dim, t_dim)  # Shift
        elif self.strategy == "bilinear":
            # Bilinear fusion
            self.bilinear = nn.Bilinear(t_dim, c_dim, output_dim)

        # Output projection
        if projection:
            in_dim = output_dim
            self.proj = nn.Sequential(
                nn.Linear(in_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        logger.debug(
            f"Initialized FeatureFusion with strategy={strategy}, "
            f"t_dim={t_dim}, c_dim={c_dim}, output_dim={output_dim}"
        )

    def forward(
        self, transcriptomics: torch.Tensor, chemicals: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse transcriptomics and chemical features.

        Args:
            transcriptomics: Tensor [batch_size, t_dim] with transcriptomics features
            chemicals: Tensor [batch_size, c_dim] with chemical features

        Returns:
            Tensor [batch_size, output_dim] with fused representation
        """
        # Basic fusion strategies
        if self.strategy == "concat":
            fused = torch.cat((transcriptomics, chemicals), dim=-1)

        elif self.strategy == "add":
            t_matched = self.t_match(transcriptomics)
            c_matched = self.c_match(chemicals)
            fused = t_matched + c_matched

        elif self.strategy == "multiply":
            t_matched = self.t_match(transcriptomics)
            c_matched = self.c_match(chemicals)
            fused = t_matched * c_matched

        # Advanced fusion strategies
        elif self.strategy == "gated":
            # Gated fusion
            gate_input = torch.cat((transcriptomics, chemicals), dim=-1)
            gate = self.gate(gate_input)
            fused = transcriptomics * gate

        elif self.strategy == "film":
            # FiLM fusion
            gamma = self.gamma(chemicals)
            beta = self.beta(chemicals)
            fused = (1 + gamma) * transcriptomics + beta

        elif self.strategy == "bilinear":
            # Bilinear fusion
            fused = self.bilinear(transcriptomics, chemicals)

        # Apply projection if requested
        if self.projection:
            fused = self.proj(fused)

        return fused


class MultimodalFusion(nn.Module):
    """
    Flexible fusion module for multiple modalities.

    This module supports fusing features from an arbitrary number of modalities
    using various strategies, with support for weighted combinations and
    hierarchical fusion.

    Attributes:
        strategy: Fusion strategy to use
        modality_dims: Dictionary of modality dimensions
        output_dim: Dimension of fused representation
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        output_dim: int,
        strategy: str = "concat",
        projection: bool = True,
        hierarchical: bool = False,
        learnable_weights: bool = False,
        dropout: float = 0.1,
    ):
        """
        Initialize the MultimodalFusion module.

        Args:
            modality_dims: Dictionary mapping modality names to their dimensions
            output_dim: Dimension of output features
            strategy: Fusion strategy ('concat', 'add', 'multiply', 'weighted')
            projection: Whether to project the fused representation
            hierarchical: Whether to fuse modalities hierarchically in pairs
            learnable_weights: Whether to learn weights for weighted fusion
            dropout: Dropout rate for projection layers
        """
        super(MultimodalFusion, self).__init__()

        self.strategy = strategy.lower()
        self.modality_dims = modality_dims
        self.output_dim = output_dim
        self.hierarchical = hierarchical
        self.learnable_weights = learnable_weights

        # Validate strategy
        valid_strategies = ["concat", "add", "multiply", "weighted"]
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Unknown fusion strategy: {self.strategy}. Must be one of {valid_strategies}"
            )

        # Create dimension-matching projections
        self.modality_projections = nn.ModuleDict()

        # For add/multiply/weighted, project all to same dimension
        if self.strategy in ["add", "multiply", "weighted"]:
            max_dim = max(modality_dims.values())
            for name, dim in modality_dims.items():
                if dim != max_dim:
                    self.modality_projections[name] = nn.Linear(dim, max_dim)
                else:
                    self.modality_projections[name] = nn.Identity()

            # Input dimension for projection layer
            self.fusion_dim = max_dim
        else:  # concat
            # Input dimension for projection layer is sum of all dimensions
            self.fusion_dim = sum(modality_dims.values())

        # Learnable weights for weighted fusion
        if self.strategy == "weighted" and learnable_weights:
            self.weights = nn.Parameter(
                torch.ones(len(modality_dims)) / len(modality_dims)
            )

        # Output projection
        if projection:
            self.proj = nn.Sequential(
                nn.Linear(self.fusion_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            if self.fusion_dim != output_dim:
                self.proj = nn.Linear(self.fusion_dim, output_dim)
            else:
                self.proj = nn.Identity()

        # For hierarchical fusion
        if hierarchical and len(modality_dims) > 2:
            # Create a binary tree of fusion modules
            self.build_fusion_tree(list(modality_dims.items()))

        logger.debug(
            f"Initialized MultimodalFusion with strategy={strategy}, "
            f"modalities={list(modality_dims.keys())}, output_dim={output_dim}, "
            f"hierarchical={hierarchical}, learnable_weights={learnable_weights}"
        )

    def build_fusion_tree(self, modalities: List[Tuple[str, int]]):
        """
        Build a binary tree of fusion modules for hierarchical fusion.

        Args:
            modalities: List of (name, dimension) tuples for each modality
        """
        self.fusion_modules = nn.ModuleList()

        # Base case: just two modalities
        if len(modalities) == 2:
            return

        # Build a balanced binary tree of fusion modules
        current_level = []
        for i in range(0, len(modalities) - 1, 2):
            if i + 1 < len(modalities):
                # Create a fusion module for this pair
                name1, dim1 = modalities[i]
                name2, dim2 = modalities[i + 1]

                fusion = FeatureFusion(
                    t_dim=dim1,
                    c_dim=dim2,
                    output_dim=max(dim1, dim2),
                    strategy=self.strategy,
                    projection=True,
                )

                self.fusion_modules.append(fusion)
                current_level.append((f"{name1}_{name2}", max(dim1, dim2)))
            else:
                # Odd number of modalities, pass through
                current_level.append(modalities[i])

        # If we have more than 2 fused representations, continue building the tree
        if len(current_level) > 2:
            self.build_fusion_tree(current_level)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse multiple modalities.

        Args:
            features: Dictionary mapping modality names to their feature tensors
                     Each tensor should be [batch_size, modality_dim]

        Returns:
            Tensor [batch_size, output_dim] with fused representation
        """
        # Check if all required modalities are present
        missing_modalities = set(self.modality_dims.keys()) - set(features.keys())
        if missing_modalities:
            raise ValueError(f"Missing modalities: {missing_modalities}")

        # Project modalities to compatible dimensions if needed
        projected_features = {}
        for name, tensor in features.items():
            if name in self.modality_projections:
                projected_features[name] = self.modality_projections[name](tensor)
            else:
                projected_features[name] = tensor

        # Hierarchical fusion
        if self.hierarchical and len(self.modality_dims) > 2:
            return self._hierarchical_fusion(projected_features)

        # Standard fusion
        if self.strategy == "concat":
            modality_list = [tensor for name, tensor in projected_features.items()]
            fused = torch.cat(modality_list, dim=-1)

        elif self.strategy == "add":
            modality_list = [tensor for name, tensor in projected_features.items()]
            fused = sum(modality_list)

        elif self.strategy == "multiply":
            modality_list = [tensor for name, tensor in projected_features.items()]
            fused = modality_list[0]
            for tensor in modality_list[1:]:
                fused = fused * tensor

        elif self.strategy == "weighted":
            modality_list = [tensor for name, tensor in projected_features.items()]

            if self.learnable_weights:
                # Use learned weights (softmax to ensure they sum to 1)
                weights = F.softmax(self.weights, dim=0)
                fused = sum(w * tensor for w, tensor in zip(weights, modality_list))
            else:
                # Equal weights
                fused = sum(modality_list) / len(modality_list)

        # Apply projection
        return self.proj(fused)

    def _hierarchical_fusion(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply hierarchical fusion by recursively fusing pairs of modalities.

        Args:
            features: Dictionary mapping modality names to their feature tensors

        Returns:
            Tensor with fused representation
        """
        modality_names = list(features.keys())
        feature_list = [features[name] for name in modality_names]

        # For simplicity, handle the case of exactly 3 modalities
        if len(feature_list) == 3:
            # First fuse modalities 0 and 1
            intermediate = self.fusion_modules[0](feature_list[0], feature_list[1])
            # Then fuse the result with modality 2
            return self.proj(torch.cat([intermediate, feature_list[2]], dim=-1))

        # More complex hierarchical fusion would require implementing a full tree traversal
        # This is a simplified implementation
        remaining_features = feature_list
        fusion_idx = 0

        while len(remaining_features) > 1:
            new_features = []

            # Process pairs
            for i in range(0, len(remaining_features) - 1, 2):
                if i + 1 < len(remaining_features):
                    # Fuse this pair
                    fused = self.fusion_modules[fusion_idx](
                        remaining_features[i], remaining_features[i + 1]
                    )
                    new_features.append(fused)
                    fusion_idx += 1
                else:
                    # Odd number, pass through
                    new_features.append(remaining_features[i])

            remaining_features = new_features

        # Apply final projection
        return self.proj(remaining_features[0])
