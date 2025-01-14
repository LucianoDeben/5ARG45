import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SparseKnowledgeNetwork(FlexibleFCNN):
    def __init__(
        self,
        gene_tf_matrix: torch.Tensor,
        input_dim: int,
        hidden_dims: list,
        output_dim: int = 1,
        first_activation: str = "tanh",
        downstream_activation: str = "relu",
        dropout_prob: float = 0.2,
        residual: bool = True,
        norm_type: str = "batchnorm",
        weight_init: str = "xavier",
    ):
        """
        Knowledge-Informed Sparse Network with Flexible FCNN capabilities.

        Args:
            gene_tf_matrix (torch.Tensor): Binary (-1, 0, 1) gene-TF connection matrix.
            input_dim (int): Number of input genes (columns in gene_tf_matrix).
            hidden_dims (list of int): Sizes of additional hidden layers after TF activations.
            output_dim (int): Number of output features (1 for regression).
            first_activation (str): Activation function after the gene-to-TF layer.
            downstream_activation (str): Activation function for downstream layers.
            dropout_prob (float): Dropout probability.
            residual (bool): Whether to use residual connections in downstream layers.
            norm_type (str): Normalization type ("batchnorm" or "layernorm").
            weight_init (str): Weight initialization method ("xavier" or "kaiming").
        """
        # Validate gene_tf_matrix dimensions
        if gene_tf_matrix.shape[0] != input_dim:
            raise ValueError(
                f"Input dimension ({input_dim}) does not match gene-to-TF matrix rows ({gene_tf_matrix.shape[0]})."
            )

        # Initialize the base FlexibleFCNN
        tf_dim = gene_tf_matrix.shape[1]
        super(SparseKnowledgeNetwork, self).__init__(
            input_dim=tf_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation_fn=downstream_activation,
            dropout_prob=dropout_prob,
            residual=residual,
            norm_type=norm_type,
            weight_init=weight_init,
        )

        # Knowledge-informed gene-to-TF layer
        self.gene_to_tf_weights = nn.Parameter(gene_tf_matrix.clone().float())

        # First activation function for the gene-to-TF layer
        if first_activation.lower() not in {"tanh", "sigmoid"}:
            raise ValueError("First activation must be 'tanh' or 'sigmoid'.")
        self.first_activation = getattr(F, first_activation.lower())

    def forward(self, x):
        """
        Forward pass for SparseKnowledgeNetwork.

        Args:
            x (torch.Tensor): Input tensor (gene expression data).

        Returns:
            torch.Tensor: Model output.
        """
        # Trainable gene-to-TF interaction layer
        tf_activations = torch.matmul(x, self.gene_to_tf_weights)

        # Apply first activation (e.g., Tanh or Sigmoid)
        tf_activations = self.first_activation(tf_activations)

        # Pass through FlexibleFCNN (hidden layers + output layer)
        return super(SparseKnowledgeNetwork, self).forward(tf_activations)
