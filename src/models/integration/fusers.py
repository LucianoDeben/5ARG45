# models/integration/fusers.py
import logging
from typing import Dict, List, Optional, Tuple, Union

from src.models.integration.attention import CrossModalAttention, ModalityAttentionFusion
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

        # Build fusion tree for hierarchical fusion if enabled
        if hierarchical:
            self.fusion_modules = nn.ModuleList()
            self.build_fusion_tree(list(modality_dims.items()))

        logger.debug(
            f"Initialized MultimodalFusion with strategy={strategy}, "
            f"modalities={list(modality_dims.keys())}, output_dim={output_dim}, "
            f"hierarchical={hierarchical}, learnable_weights={learnable_weights}"
        )

    def build_fusion_tree(self, modalities: List[Tuple[str, int]]) -> None:
        """
        Build a binary tree of fusion modules for hierarchical fusion.

        Args:
            modalities: List of (name, dimension) tuples for each modality
        """
        if len(modalities) <= 1:
            return

        # Create pairs for fusion, handling odd numbers by pairing with the last
        current_level = []
        for i in range(0, len(modalities) - 1, 2):
            if i + 1 < len(modalities):
                name1, dim1 = modalities[i]
                name2, dim2 = modalities[i + 1]
                output_dim = max(dim1, dim2)  # Match the largest dimension
                fusion = FeatureFusion(
                    t_dim=dim1,
                    c_dim=dim2,
                    output_dim=output_dim,
                    strategy=self.strategy,
                    projection=True,
                )
                self.fusion_modules.append(fusion)
                current_level.append((f"{name1}_{name2}", output_dim))
            else:
                # Handle odd number of modalities by keeping the last one unchanged
                current_level.append(modalities[i])

        # Recursively build the tree if there are still multiple nodes
        if len(current_level) > 1:
            self.build_fusion_tree(current_level)
        elif current_level:  # Handle the final node (single modality or fused pair)
            self.final_dim = current_level[0][1]
            self.final_projection = nn.Linear(self.final_dim, self.output_dim)

    def _hierarchical_fusion(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply hierarchical fusion by recursively fusing pairs of modalities.

        Args:
            features: Dictionary mapping modality names to their feature tensors

        Returns:
            Tensor [batch_size, output_dim] with fused representation
        """
        modality_names = list(features.keys())
        feature_list = [features[name] for name in modality_names]

        # Project features to compatible dimensions
        projected_features = []
        for feature in feature_list:
            if feature.size(-1) in self.modality_dims.values():
                projected_features.append(feature)
            else:
                # Find the matching projection or use identity
                for name, proj in self.modality_projections.items():
                    if self.modality_dims[name] == feature.size(-1):
                        projected_features.append(proj(feature))
                        break
                else:
                    raise ValueError(
                        f"No matching projection for feature dimension {feature.size(-1)}"
                    )

        # Initialize the list of features to fuse
        current_features = projected_features
        fusion_idx = 0

        while len(current_features) > 1:
            new_features = []
            for i in range(0, len(current_features) - 1, 2):
                if i + 1 < len(current_features):
                    # Fuse this pair using the corresponding fusion module
                    fused = self.fusion_modules[fusion_idx](
                        current_features[i], current_features[i + 1]
                    )
                    new_features.append(fused)
                    fusion_idx += 1
                else:
                    # Handle odd number by keeping the last feature unchanged
                    new_features.append(current_features[i])
            current_features = new_features

        # Apply final projection if we have a single fused feature
        if current_features:
            if hasattr(self, "final_projection"):
                return self.final_projection(current_features[0])
            return self.proj(current_features[0])
        raise ValueError("No features to fuse after hierarchical processing")

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
        if self.hierarchical:
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

def create_feature_fusion(
    fusion_type: str,
    modality_dims: Union[Dict[str, int], Tuple[int, int]],
    output_dim: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of feature fusion modules.
    
    Args:
        fusion_type: Type of fusion ('simple', 'multimodal', 'attention')
        modality_dims: Dimensions of modalities to fuse 
            - For simple fusion: (transcriptomics_dim, chemical_dim)
            - For multimodal: Dictionary of {modality_name: dimension}
        output_dim: Dimension of output features
        **kwargs: Additional arguments specific to fusion types
    
    Returns:
        A feature fusion module
    """
    if fusion_type == "simple":
        # For two-modality fusion
        if isinstance(modality_dims, tuple) and len(modality_dims) == 2:
            return FeatureFusion(
                t_dim=modality_dims[0],
                c_dim=modality_dims[1],
                output_dim=output_dim,
                strategy=kwargs.get("strategy", "concat"),
                projection=kwargs.get("projection", True),
                dropout=kwargs.get("dropout", 0.1)
            )
        else:
            raise ValueError("Simple fusion requires a tuple of two dimensions")
    
    elif fusion_type == "multimodal":
        # For multiple modalities
        if isinstance(modality_dims, dict):
            return MultimodalFusion(
                modality_dims=modality_dims,
                output_dim=output_dim,
                strategy=kwargs.get("strategy", "concat"),
                projection=kwargs.get("projection", True),
                hierarchical=kwargs.get("hierarchical", False),
                learnable_weights=kwargs.get("learnable_weights", False),
                dropout=kwargs.get("dropout", 0.1)
            )
        else:
            raise ValueError("Multimodal fusion requires a dictionary of modality dimensions")
    
    elif fusion_type == "attention":
        # For attention-based fusion
        if isinstance(modality_dims, dict):
            return ModalityAttentionFusion(
                modality_dims=modality_dims,
                hidden_dim=kwargs.get("hidden_dim", 128),
                output_dim=output_dim,
                num_heads=kwargs.get("num_heads", 4),
                dropout=kwargs.get("dropout", 0.1),
                aggregation=kwargs.get("aggregation", "mean")
            )
        else:
            raise ValueError("Attention fusion requires a dictionary of modality dimensions")
    
    elif fusion_type == "cross_modal":
        # For cross-modal attention
        if isinstance(modality_dims, tuple) and len(modality_dims) == 2:
            return CrossModalAttention(
                transcriptomics_dim=modality_dims[0],
                chemical_dim=modality_dims[1],
                hidden_dim=kwargs.get("hidden_dim", 128),
                num_heads=kwargs.get("num_heads", 4),
                dropout=kwargs.get("dropout", 0.1),
                use_projection=kwargs.get("use_projection", True)
            )
        else:
            raise ValueError("Cross-modal fusion requires a tuple of two dimensions")
    
    else:
        raise ValueError(f"Unsupported fusion type: {fusion_type}")