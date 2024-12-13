import torch
import torch.nn as nn

ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "identity": nn.Identity,
    "prelu": nn.PReLU,
    "elu": nn.ELU,
}


class FlexibleFCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        activation_fn="relu",
        dropout_prob=0.0,
        residual=False,
        use_batchnorm=True,
    ):
        super(FlexibleFCNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.residual = residual
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob

        if isinstance(activation_fn, str):
            if activation_fn.lower() not in ACTIVATION_MAP:
                raise ValueError(f"Unknown activation function {activation_fn}")
            self.activation_fn = ACTIVATION_MAP[activation_fn.lower()]()
        else:
            self.activation_fn = activation_fn

        layer_dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None
        self.dropouts = nn.ModuleList()

        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            self.layers.append(nn.Linear(in_dim, out_dim))

            if use_batchnorm:
                self.bns.append(nn.BatchNorm1d(out_dim))

            if dropout_prob > 0.0:
                self.dropouts.append(nn.Dropout(dropout_prob))
            else:
                self.dropouts.append(nn.Identity())

        self.output_layer = nn.Linear(layer_dims[-1], output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            residual_input = x
            x = layer(x)

            if self.use_batchnorm:
                x = self.bns[i](x)

            x = self.dropouts[i](x)
            x = self.activation_fn(x)

            if self.residual and residual_input.shape == x.shape:
                x = x + residual_input

        x = self.output_layer(x)
        return x.squeeze()  # Ensure the output has the correct shape

    def get_regularization_loss(self, l1_lambda=0.0, l2_lambda=0.0):
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        if l1_lambda == 0.0 and l2_lambda == 0.0:
            return reg_loss

        for layer in self.layers:
            if l1_lambda > 0.0:
                reg_loss += l1_lambda * torch.sum(torch.abs(layer.weight))
            if l2_lambda > 0.0:
                reg_loss += l2_lambda * torch.sum(layer.weight**2)

        if l1_lambda > 0.0:
            reg_loss += l1_lambda * torch.sum(torch.abs(self.output_layer.weight))
        if l2_lambda > 0.0:
            reg_loss += l2_lambda * torch.sum(self.output_layer.weight**2)

        return reg_loss
