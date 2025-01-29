import pandas as pd
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
        residual=False,
        norm_type="batchnorm",
        weight_init="kaiming",
    ):
        super(FlexibleFCNN, self).__init__()
        self.residual = residual

        # Ensure hidden_dims are consistent for residual
        if residual:
            hidden_dims = [hidden_dims[0]] * len(hidden_dims)

        self.activation = ACTIVATIONS.get(activation_fn.lower(), nn.ReLU)()
        self.norm_type = norm_type.lower()

        # Build layers
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            norm = (
                NORM_LAYERS.get(self.norm_type, nn.Identity)(dims[i + 1])
                if norm_type != "none"
                else nn.Identity()
            )
            self.norms.append(norm)

        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.output = nn.Linear(hidden_dims[-1], output_dim)

        self._initialize_weights(weight_init)

    def _initialize_weights(self, method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = x
        for layer, norm in zip(self.layers, self.norms):
            identity = out  # Save for residual

            # Main path
            out = layer(out)
            out = norm(out)

            # Optional residual
            if self.residual and (out.shape == identity.shape):
                out = out + identity

            out = self.activation(out)
            out = self.dropout(out)

        return self.output(out)


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
        super(SparseKnowledgeNetwork, self).__init__()
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.use_batchnorm = use_batchnorm

        # Store the precomputed gene-TF matrix as a trainable parameter
        self.gene_tf_matrix = nn.Parameter(gene_tf_matrix)

        # First activation function
        FIRST_ACTIVATIONS = {"tanh": torch.tanh, "sigmoid": torch.sigmoid}
        self.first_activation = FIRST_ACTIVATIONS.get(first_activation.lower())
        if self.first_activation is None:
            raise ValueError("First activation must be 'tanh' or 'sigmoid'.")

        # Downstream activation function
        DOWNSTREAM_ACTIVATIONS = {"relu": F.relu, "gelu": F.gelu, "silu": F.silu}
        self.downstream_activation = DOWNSTREAM_ACTIVATIONS.get(
            downstream_activation.lower()
        )
        if self.downstream_activation is None:
            raise ValueError("Unknown downstream activation.")

        # Define the hidden layers after the TF layer
        tf_dim = self.gene_tf_matrix.shape[1]
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
        # Gene-to-TF interaction layer
        tf_activations = torch.matmul(x, self.gene_tf_matrix)

        # Apply the first activation function
        tf_activations = self.first_activation(tf_activations)

        # Pass through hidden layers
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
