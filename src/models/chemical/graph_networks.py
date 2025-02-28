# models/chemical/graph_networks.py
import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GlobalAttention,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

logger = logging.getLogger(__name__)


class MolecularGraphEncoder(nn.Module):
    """
    Process molecular graphs using Graph Neural Networks.

    This module encodes molecular structures represented as graphs using GNN architectures.
    It supports different GNN layer types (GCN, GAT) and pooling mechanisms.

    Attributes:
        input_dim: Number of input node features
        hidden_dims: List of hidden dimensions for GNN layers
        output_dim: Dimension of the final molecular representation
        gnn_type: Type of GNN layer to use ('gcn' or 'gat')
        pooling: Type of graph pooling to use ('mean', 'add', 'max', or 'attention')
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        gnn_type: str = "gcn",
        pooling: str = "mean",
        dropout: float = 0.3,
    ):
        """
        Initialize the MolecularGraphEncoder.

        Args:
            input_dim: Number of input node features
            hidden_dims: List of hidden dimensions for GNN layers
            output_dim: Dimension of the final molecular representation
            gnn_type: Type of GNN layer to use ('gcn' or 'gat')
            pooling: Type of graph pooling to use ('mean', 'add', 'max', or 'attention')
            dropout: Dropout rate
        """
        super(MolecularGraphEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.gnn_type = gnn_type.lower()
        self.pooling = pooling.lower()
        self.dropout = dropout

        # Validate inputs
        valid_gnn_types = ["gcn", "gat"]
        if self.gnn_type not in valid_gnn_types:
            raise ValueError(
                f"Invalid GNN type: {self.gnn_type}. Must be one of {valid_gnn_types}"
            )

        valid_pooling = ["mean", "add", "max", "attention"]
        if self.pooling not in valid_pooling:
            raise ValueError(
                f"Invalid pooling: {self.pooling}. Must be one of {valid_pooling}"
            )

        # Build GNN layers
        self.convs = nn.ModuleList()
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            if self.gnn_type == "gcn":
                self.convs.append(GCNConv(prev_dim, hidden_dim))
            elif self.gnn_type == "gat":
                self.convs.append(GATConv(prev_dim, hidden_dim, heads=1))
            prev_dim = hidden_dim

        # Setup graph pooling
        if self.pooling == "attention":
            self.pool_gate_nn = nn.Linear(hidden_dims[-1], 1)
            self.pool = GlobalAttention(self.pool_gate_nn)

        # Final projection to output dimension
        self.project = nn.Linear(hidden_dims[-1], output_dim)
        logger.debug(
            f"Initialized MolecularGraphEncoder with {gnn_type} layers, {pooling} pooling, "
            f"input_dim={input_dim}, output_dim={output_dim}"
        )

    def forward(self, data):
        """
        Forward pass for molecular graph data.

        Args:
            data: PyTorch Geometric Data object or batch containing:
                - x: Node feature matrix [num_nodes, input_dim]
                - edge_index: Graph connectivity [2, num_edges]
                - batch: Batch assignment vector [num_nodes]

        Returns:
            Tensor of shape [batch_size, output_dim] with molecular embeddings
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Process through GNN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply graph pooling
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "add":
            x = global_add_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "attention":
            x = self.pool(x, batch)

        # Project to output dimension
        x = self.project(x)
        return x


class DosageIntegration(nn.Module):
    """
    Integrates dosage information with molecular graph representations.

    This module combines molecular graph embeddings with dosage information
    to create a comprehensive representation for drug response prediction.

    Attributes:
        integration_type: How to integrate dosage ('concat', 'scale', or 'bilinear')
        graph_dim: Dimension of the graph representation
        dosage_dim: Dimension of the dosage information (typically 1)
        output_dim: Dimension of the output representation
    """

    def __init__(
        self,
        graph_dim: int,
        output_dim: int,
        integration_type: str = "concat",
        dosage_dim: int = 1,
    ):
        """
        Initialize the DosageIntegration module.

        Args:
            graph_dim: Dimension of the graph representation
            output_dim: Dimension of the output representation
            integration_type: How to integrate dosage ('concat', 'scale', or 'bilinear')
            dosage_dim: Dimension of the dosage information (typically 1)
        """
        super(DosageIntegration, self).__init__()
        self.integration_type = integration_type.lower()
        self.graph_dim = graph_dim
        self.dosage_dim = dosage_dim
        self.output_dim = output_dim

        valid_types = ["concat", "scale", "bilinear"]
        if self.integration_type not in valid_types:
            raise ValueError(
                f"Invalid integration type: {integration_type}. Must be one of {valid_types}"
            )

        if self.integration_type == "concat":
            self.projection = nn.Linear(graph_dim + dosage_dim, output_dim)
        elif self.integration_type == "scale":
            self.dosage_gates = nn.Linear(dosage_dim, graph_dim)
            self.projection = nn.Linear(graph_dim, output_dim)
        elif self.integration_type == "bilinear":
            self.bilinear = nn.Bilinear(graph_dim, dosage_dim, graph_dim)
            self.projection = nn.Linear(graph_dim, output_dim)

    def forward(
        self, graph_embedding: torch.Tensor, dosage: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine graph embedding with dosage information.

        Args:
            graph_embedding: Tensor [batch_size, graph_dim] with molecular graph embeddings
            dosage: Tensor [batch_size, dosage_dim] with dosage information

        Returns:
            Tensor [batch_size, output_dim] with combined representation
        """
        if self.integration_type == "concat":
            x = torch.cat([graph_embedding, dosage], dim=-1)
            return self.projection(x)
        elif self.integration_type == "scale":
            scaling = torch.sigmoid(self.dosage_gates(dosage))
            x = graph_embedding * scaling
            return self.projection(x)
        elif self.integration_type == "bilinear":
            x = self.bilinear(graph_embedding, dosage)
            x = F.relu(x)
            return self.projection(x)
