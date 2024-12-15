import math

import torch
import torch.nn as nn

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
        hidden_dims=[512, 256, 128],
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


class MLPMixer(nn.Module):
    def __init__(
        self,
        num_mixers=4,
        token_dim=128,
        channel_dim=128,
        hidden_dim=128,
        dropout_prob=0.2,
        output_dim=1,
    ):
        super(MLPMixer, self).__init__()

        self.num_mixers = num_mixers
        self.token_dim = token_dim
        self.channel_dim = channel_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.output_dim = output_dim

        # Removed input_dim from arguments as per previous suggestion.
        # We'll set input_dim dynamically in the pipeline before creating the model.

        # We'll define input_dim in forward or pipeline.

        # For now, let's assume we can pass input_dim to forward or set it dynamically later.

    def build(self, input_dim):
        # This method can be called after knowing input_dim.
        self.input_dim = input_dim
        self.token_embedding = nn.Linear(1, self.hidden_dim)

        self.mixer_layers = nn.ModuleList([])
        for _ in range(self.num_mixers):
            self.mixer_layers.append(
                MixerLayer(
                    self.input_dim,
                    self.hidden_dim,
                    self.token_dim,
                    self.channel_dim,
                    self.dropout_prob,
                )
            )

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # x: (B, input_dim)
        # Ensure build was called
        if not hasattr(self, "input_dim"):
            raise RuntimeError(
                "MLPMixer: build() was not called to set input_dim before forward."
            )

        x = x.unsqueeze(-1)  # (B, input_dim, 1)
        x = self.token_embedding(x)  # (B, input_dim, hidden_dim)

        for layer in self.mixer_layers:
            x = layer(x)

        x = self.norm(x)  # (B, input_dim, hidden_dim)
        x = x.mean(dim=1)  # Pool over tokens -> (B, hidden_dim)
        x = self.head(x)  # (B, output_dim)
        return x


class MixerLayer(nn.Module):
    def __init__(self, num_tokens, hidden_dim, token_dim, channel_dim, dropout_prob):
        super(MixerLayer, self).__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim), MLP(num_tokens, token_dim, dropout_prob)
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim), MLP(hidden_dim, channel_dim, dropout_prob)
        )

    def forward(self, x):
        # x: (B, Tokens, Channels)
        # Token mixing: transpose to (B, Channels, Tokens), apply MLP, transpose back
        y = x.transpose(1, 2)  # (B, Channels, Tokens)
        y = self.token_mixing(y)  # MLP over Tokens dimension
        y = y.transpose(1, 2)  # back to (B, Tokens, Channels)
        x = x + y

        # Channel mixing: MLP over the Channels dimension
        y = self.channel_mixing(x)  # MLP over Channels
        x = x + y
        return x


class MLP(nn.Module):
    def __init__(self, dim, expansion_dim, dropout_prob):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, expansion_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(expansion_dim, dim),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        return self.net(x)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        output_dim=1,
    ):
        super(TransformerRegressor, self).__init__()

        # Positional encoding for tokens if needed
        self.pos_embedding = PositionalEncoding(d_model, dropout, max_len=input_dim)

        # Project input genes into d_model dimension
        self.input_proj = nn.Linear(1, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (B, input_dim)
        # Treat each gene as a token: shape (B, Tokens)
        # Add channel dimension: (B, Tokens, 1)
        x = x.unsqueeze(-1)
        # Project to d_model
        x = self.input_proj(x)  # (B, Tokens, d_model)

        # Transformer expects (Tokens, B, d_model)
        x = x.transpose(0, 1)  # (Tokens, B, d_model)

        x = self.pos_embedding(x)  # add positional encoding
        x = self.transformer_encoder(x)  # (Tokens, B, d_model)

        x = x.mean(dim=0)  # pool over tokens (genes)
        x = self.norm(x)
        x = self.output_layer(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (Tokens, B, d_model)
        length = x.size(0)
        x = x + self.pe[:length, :]
        return self.dropout(x)


class CNNRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        num_filters=64,
        kernel_size=7,
        num_layers=3,
        dropout_prob=0.2,
        output_dim=1,
    ):
        super(CNNRegressor, self).__init__()

        layers = []
        in_channels = 1  # One "channel" of gene expression
        # We'll treat input as (B, 1, input_dim)
        for i in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    num_filters,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(num_filters))
            layers.append(nn.Dropout(dropout_prob))
            in_channels = num_filters

        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(num_filters, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, output_dim),
        )
        self.input_dim = input_dim

    def forward(self, x):
        # x: (B, input_dim)
        x = x.unsqueeze(1)  # (B, 1, input_dim)
        x = self.conv_layers(x)  # (B, num_filters, input_dim)
        # Pool over input_dim
        x = x.mean(dim=2)  # (B, num_filters)
        x = self.fc(x)  # (B, output_dim)
        return x
