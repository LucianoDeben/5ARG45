# models/unimodal/flexible_fcnn.py
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple, Union

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
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128, 64],
        output_dim: int = 1,
        activation_fn: str = "relu",
        dropout_prob: float = 0.2,
        residual: bool = False,
        norm_type: str = "batchnorm",
        weight_init: str = "kaiming",
        feature_names: Optional[List[str]] = None,  # Added for interpretability
    ):
        """
        A flexible fully connected neural network (FCNN) with support for interpretability and uncertainty.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dims (List[int]): List of hidden layer dimensions.
            output_dim (int): Dimension of the output (default 1 for regression).
            activation_fn (str): Activation function name (e.g., 'relu', 'gelu').
            dropout_prob (float): Dropout probability for regularization and uncertainty estimation.
            residual (bool): Whether to use residual connections (requires consistent hidden dims).
            norm_type (str): Type of normalization ('batchnorm', 'layernorm', 'none').
            weight_init (str): Weight initialization method ('kaiming', 'xavier').
            feature_names (Optional[List[str]]): Names of input features for interpretability.
        """
        super(FlexibleFCNN, self).__init__()
        self.residual = residual
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_names = feature_names or [f"feat_{i}" for i in range(input_dim)]

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

    def _initialize_weights(self, method: str) -> None:
        """
        Initialize weights of linear layers.

        Args:
            method (str): Initialization method ('kaiming', 'xavier').
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass of the FCNN. Handles both tensor and dict inputs for multimodal compatibility.

        Args:
            x (Union[torch.Tensor, Dict[str, torch.Tensor]]): Input tensor or dict with 'transcriptomics'/'chemical' keys.

        Returns:
            torch.Tensor: Model predictions.
        """
        if isinstance(x, dict):
            # Handle multimodal inputs by concatenating or processing as needed
            if "transcriptomics" in x and "chemical" in x:
                x = torch.cat([x["transcriptomics"], x["chemical"]], dim=1)
            elif "transcriptomics" in x:
                x = x["transcriptomics"]
            elif "chemical" in x:
                x = x["chemical"]
            else:
                raise ValueError("Input dict must contain 'transcriptomics' and/or 'chemical' keys")

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

    def get_feature_importance(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        method: str = "integrated_gradients",
        n_steps: int = 50,
        subset_size: Optional[int] = None,
        cache_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Compute feature importance using Captum's interpretability methods.

        Args:
            inputs (Union[torch.Tensor, Dict[str, torch.Tensor]]): Input data for attribution.
            method (str): Captum method to use ('integrated_gradients', 'saliency', 'deeplift').
            n_steps (int): Number of steps for integrated gradients approximation.
            subset_size (Optional[int]): Number of samples to use (for performance).
            cache_path (Optional[str]): Path to cache results, if any.

        Returns:
            Dict[str, float]: Mapping of feature names to their importance scores.
        """
        from captum.attr import IntegratedGradients, Saliency, DeepLift
        import json
        import os

        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)

        self.eval()  # Ensure model is in evaluation mode for attribution

        # Handle subset for performance
        if subset_size is not None and isinstance(inputs, torch.Tensor):
            indices = torch.randperm(inputs.size(0))[:subset_size]
            inputs = inputs[indices]
        elif subset_size is not None and isinstance(inputs, dict):
            indices = torch.randperm(inputs[list(inputs.keys())[0]].size(0))[:subset_size]
            inputs = {k: v[indices] for k, v in inputs.items()}

        # Select attribution method
        if method.lower() == "integrated_gradients":
            attr_method = IntegratedGradients(self)
        elif method.lower() == "saliency":
            attr_method = Saliency(self)
        elif method.lower() == "deeplift":
            attr_method = DeepLift(self)
        else:
            raise ValueError(f"Unsupported attribution method: {method}")

        # Handle multimodal inputs
        if isinstance(inputs, dict):
            # For now, we'll attribute to the concatenated input (transcriptomics + chemical)
            # If separate attribution is needed, this would need to be split
            combined_input = torch.cat([inputs["transcriptomics"], inputs["chemical"]], dim=1)
            combined_input = combined_input.to(self.device).requires_grad_(True)
            baseline = torch.zeros_like(combined_input).to(self.device)
        else:
            combined_input = inputs.to(self.device).requires_grad_(True)
            baseline = torch.zeros_like(combined_input).to(self.device)

        # Compute attributions
        if method.lower() == "integrated_gradients":
            attributions = attr_method.attribute(
                combined_input,
                baselines=baseline,
                target=0,  # For regression, attribute to the output scalar
                n_steps=n_steps,
            )
        else:
            attributions = attr_method.attribute(
                combined_input,
                target=0,  # For regression, attribute to the output scalar
            )

        # Average attributions across samples and normalize
        attr_mean = attributions.abs().mean(dim=0).cpu().numpy()
        attr_sum = attr_mean.sum()
        if attr_sum > 0:
            attr_mean /= attr_sum  # Normalize to sum to 1
            
        # Map to feature names
        importance = dict(zip(self.feature_names, attr_mean))
        
        if cache_path:
            with open(cache_path, 'w') as f:
                json.dump(importance, f, indent=2)
        return importance

    def predict_with_uncertainty(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        n_samples: int = 10,
        subset_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform Monte Carlo Dropout to estimate prediction uncertainty.

        Args:
            inputs (Union[torch.Tensor, Dict[str, torch.Tensor]]): Input data.
            n_samples (int): Number of Monte Carlo samples.
            subset_size (Optional[int]): Number of samples to use (for performance).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean predictions and standard deviation of predictions.
        """
        self.train()  # Enable dropout during inference

        # Handle subset for performance
        if subset_size is not None and isinstance(inputs, torch.Tensor):
            indices = torch.randperm(inputs.size(0))[:subset_size]
            inputs = inputs[indices]
        elif subset_size is not None and isinstance(inputs, dict):
            indices = torch.randperm(inputs[list(inputs.keys())[0]].size(0))[:subset_size]
            inputs = {k: v[indices] for k, v in inputs.items()}

        preds = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self(inputs)
            preds.append(pred)
        preds = torch.stack(preds)
        mean_preds = preds.mean(dim=0)
        std_preds = preds.std(dim=0)
        self.eval()  # Return to evaluation mode
        return mean_preds, std_preds