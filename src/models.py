import torch
import torch.nn as nn


class FlexibleFCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layers=(512, 256, 128),
        dropout_rate=0.3,
        activation=nn.GELU,
        output_activation=None,
        regularization_weight=1e-4,
    ):
        """
        A flexible fully connected neural network (FCNN) with advanced features.

        Args:
            input_dim (int): Input dimension of the data.
            hidden_layers (tuple): Dimensions of hidden layers.
            dropout_rate (float): Dropout rate for regularization.
            activation (nn.Module): Activation function for hidden layers.
            output_activation (nn.Module or None): Activation function for the output layer (e.g., nn.Sigmoid).
            regularization_weight (float): Weight for L2 regularization.
        """
        super(FlexibleFCNN, self).__init__()

        self.layers = nn.ModuleList()
        self.film_scale = nn.ModuleList()
        self.film_shift = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.activation = activation()
        self.output_activation = output_activation() if output_activation else None

        prev_dim = input_dim
        for layer_dim in hidden_layers:
            self.layers.append(nn.Linear(prev_dim, layer_dim))
            self.layer_norms.append(nn.LayerNorm(layer_dim))
            self.film_scale.append(nn.Linear(prev_dim, prev_dim))
            self.film_shift.append(nn.Linear(prev_dim, prev_dim))
            prev_dim = layer_dim

        self.output_layer = nn.Linear(prev_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

        self.regularization_weight = regularization_weight

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Applies Xavier initialization to all layers."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            x = layer(x)  # Apply the linear layer
            if i > 0:  # Apply FiLM only after the first layer
                scale = torch.sigmoid(self.film_scale[i](x))
                shift = self.film_shift[i](x)
                x = scale * x + shift
            x = norm(x)  # Layer normalization
            x = self.activation(x)  # Non-linear activation
            x = self.dropout(x)  # Dropout
        x = self.output_layer(x)  # Output layer
        return x
