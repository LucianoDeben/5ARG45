import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "prelu": nn.PReLU,
}

NORM_LAYERS = {"batchnorm": nn.BatchNorm1d, "layernorm": nn.LayerNorm, "none": None}


class FlexibleFCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[512, 256, 128, 64],
        output_dim=1,
        activation_fn="relu",
        dropout_prob=0.2,
        residual=True,
        norm_type="batchnorm",
        weight_init="xavier",
    ):
        """
        Flexible Fully-Connected Neural Network.

        Args:
            input_dim (int): Number of input features.
            hidden_dims (list of int): Sizes of hidden layers.
            output_dim (int): Output dimension (1 for regression).
            activation_fn (str): Activation function name (e.g. "relu", "gelu", "silu").
            dropout_prob (float): Dropout probability.
            residual (bool): Whether to use residual connections between layers.
            norm_type (str): Normalization type: "batchnorm", "layernorm", "none".
            weight_init (str): Weight initialization method: "xavier" or "kaiming".

        """
        super(FlexibleFCNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.residual = residual
        self.activation_fn = (
            ACTIVATIONS[activation_fn.lower()]
            if activation_fn.lower() in ACTIVATIONS
            else nn.ReLU
        )
        self.norm_type = norm_type.lower()
        self.dropout_prob = dropout_prob

        # Layers
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            # Normalization layer
            if NORM_LAYERS[self.norm_type] is not None:
                if self.norm_type == "layernorm":
                    # LayerNorm is applied per sample per feature, so dimension = hidden_dim
                    self.norms.append(NORM_LAYERS[self.norm_type](dims[i + 1]))
                else:
                    # BatchNorm expects features as dimension for 1D data
                    self.norms.append(NORM_LAYERS[self.norm_type](dims[i + 1]))
            else:
                self.norms.append(nn.Identity())

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.act = self.activation_fn()

        # Initialize weights
        self._initialize_weights(weight_init)

    def _initialize_weights(self, method="xavier"):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif method == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                else:
                    # Default to xavier if unknown
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Forward pass with optional residual connections
        out = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            prev_out = out
            out = layer(out)
            out = norm(out)
            out = self.act(out)
            out = self.dropout(out)
            if self.residual and out.shape == prev_out.shape:
                out = out + prev_out  # Residual connection

        out = self.output_layer(out)
        return out


class SparseKnowledgeNetwork(nn.Module):
    def __init__(
        self,
        gene_tf_matrix: torch.Tensor,
        hidden_dims: list,
        output_dim: int = 1,
        first_activation: str = "tanh",
        downstream_activation: str = "relu",
        dropout_prob: float = 0.2,
        weight_init: str = "xavier",
        use_batchnorm: bool = True,
    ):
        """
        Knowledge-Informed Sparse Network with trainable gene-TF interactions.

        Args:
            gene_tf_matrix (torch.Tensor): Precomputed gene-TF matrix of shape (num_genes, num_tfs).
            hidden_dims (list of int): Sizes of additional hidden layers after TF activations.
            output_dim (int): Number of output features (1 for regression).
            first_activation (str): Activation function after the gene-to-TF layer (Tanh or Sigmoid).
            downstream_activation (str): Activation function for downstream layers (e.g., ReLU).
            dropout_prob (float): Dropout probability.
            weight_init (str): Weight initialization method ("xavier" or "kaiming").
            use_batchnorm (bool): Whether to use batch normalization in downstream layers.
        """
        super(SparseKnowledgeNetwork, self).__init__()
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.use_batchnorm = use_batchnorm

        # Store the precomputed gene-TF matrix
        self.gene_tf_matrix = gene_tf_matrix

        # Validate first activation function
        if first_activation.lower() not in {"tanh", "sigmoid"}:
            raise ValueError("First activation must be 'tanh' or 'sigmoid'.")
        self.first_activation = getattr(F, first_activation.lower())

        # Downstream activation function
        self.downstream_activation = getattr(F, downstream_activation.lower())

        # Define the hidden layers after the TF layer
        tf_dim = self.gene_tf_matrix.shape[1]  # Number of TFs
        hidden_dims = [tf_dim] + hidden_dims
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batchnorm else None

        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if use_batchnorm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

        # Initialize weights
        self._initialize_weights(weight_init)

    def _initialize_weights(self, method="xavier"):
        """
        Initialize model weights.
        """
        for layer in list(self.hidden_layers) + [self.output_layer]:
            if method == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif method == "kaiming":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            else:
                raise ValueError(f"Unknown weight initialization method: {method}")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass for the sparse knowledge-informed network.

        Args:
            x (torch.Tensor): Input tensor (gene expression data).

        Returns:
            torch.Tensor: Model output.
        """
        # Gene-to-TF interaction layer
        tf_activations = torch.matmul(x, self.gene_tf_matrix)

        # Apply the first activation function (Tanh or Sigmoid)
        tf_activations = self.first_activation(tf_activations)

        # Apply additional hidden layers
        hidden_activations = tf_activations
        for i, layer in enumerate(self.hidden_layers):
            hidden_activations = layer(hidden_activations)
            if self.use_batchnorm:
                hidden_activations = self.batch_norms[i](hidden_activations)
            hidden_activations = self.downstream_activation(hidden_activations)
            hidden_activations = self.dropout(hidden_activations)

        # Output layer
        output = self.output_layer(hidden_activations)
        return output


