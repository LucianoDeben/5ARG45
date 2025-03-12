from typing import Dict, Union
import torch 
import torch.nn as nn

# Define activation and normalization options
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

    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward pass handling both tensor and dictionary inputs."""
        # Handle dictionary inputs similar to RidgeRegression
        if isinstance(x, dict):
            if "transcriptomics" in x and "chemical" in x:
                processed_x = torch.cat([x["transcriptomics"], x["chemical"]], dim=1)
            elif "transcriptomics" in x:
                processed_x = x["transcriptomics"]
            elif "chemical" in x:
                processed_x = x["chemical"]
            else:
                raise ValueError("Input dict must contain 'transcriptomics' and/or 'chemical' keys")
        else:
            processed_x = x
            
        # Existing forward pass logic
        out = processed_x
        for layer, norm in zip(self.layers, self.norms):
            identity = out
            out = layer(out)
            out = norm(out)
            if self.residual and (out.shape == identity.shape):
                out = out + identity
            out = self.activation(out)
            out = self.dropout(out)
        return self.output(out)