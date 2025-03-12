# src/models/unimodal/ridge_regression.py
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple, Union

class RidgeRegression(nn.Module):
    """
    Pure Ridge Regression model (linear model with L2 regularization) implemented in PyTorch.
    Supports feature importance and uncertainty estimation.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        alpha: float = 1.0,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize the Ridge Regression model.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output (default 1 for regression).
            alpha (float): L2 regularization strength.
            feature_names (Optional[List[str]]): Names of input features for interpretability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha  # L2 regularization strength
        self.linear = nn.Linear(input_dim, output_dim)
        self.feature_names = feature_names or [f"feat_{i}" for i in range(input_dim)]

        # Initialize weights
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass of the Ridge Regression model. Handles both tensor and dict inputs.

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
        return self.linear(x)

    def get_param_groups(self) -> List[Dict[str, any]]:
        """
        Return parameter groups for optimizer configuration.
        Weight decay (L2 regularization) only applied to weights, not biases.

        Returns:
            List[Dict[str, any]]: Parameter groups for optimization.
        """
        return [
            {'params': self.linear.weight, 'weight_decay': self.alpha},
            {'params': self.linear.bias, 'weight_decay': 0.0}
        ]

    def get_feature_importance(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        method: str = "integrated_gradients",
        n_steps: int = 50,
        subset_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute feature importance using Captum's interpretability methods.

        Args:
            inputs (Union[torch.Tensor, Dict[str, torch.Tensor]]): Input data for attribution.
            method (str): Captum method to use ('integrated_gradients', 'saliency', 'deeplift').
            n_steps (int): Number of steps for integrated gradients approximation.
            subset_size (Optional[int]): Number of samples to use (for performance).

        Returns:
            Dict[str, float]: Mapping of feature names to their importance scores.
        """
        from captum.attr import IntegratedGradients, Saliency, DeepLift

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
        return dict(zip(self.feature_names, attr_mean))

    def predict_with_uncertainty(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        X_train: Optional[torch.Tensor] = None,
        subset_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty using the posterior variance of ridge regression.

        Args:
            inputs (Union[torch.Tensor, Dict[str, torch.Tensor]]): Input data for prediction.
            X_train (Optional[torch.Tensor]): Training data used to estimate noise variance.
                Required for uncertainty estimation.
            subset_size (Optional[int]): Number of samples to use (for performance).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean predictions and standard deviation of predictions.

        Note:
            Uncertainty estimation requires X_train to compute the noise variance.
            If X_train is not provided, returns zero uncertainty (mean prediction only).
        """
        self.eval()  # Ensure model is in evaluation mode

        # Handle subset for performance
        if subset_size is not None and isinstance(inputs, torch.Tensor):
            indices = torch.randperm(inputs.size(0))[:subset_size]
            inputs = inputs[indices]
        elif subset_size is not None and isinstance(inputs, dict):
            indices = torch.randperm(inputs[list(inputs.keys())[0]].size(0))[:subset_size]
            inputs = {k: v[indices] for k, v in inputs.items()}

        # Handle multimodal inputs
        if isinstance(inputs, dict):
            X = torch.cat([inputs["transcriptomics"], inputs["chemical"]], dim=1)
        else:
            X = inputs

        # Compute mean predictions
        mean_preds = self(X)

        # Compute uncertainty (posterior variance) if training data is provided
        if X_train is not None:
            # Weight matrix (W = (X^T X + alpha I)^(-1) X^T y, but we use the trained weights directly)
            W = self.linear.weight  # Shape: [output_dim, input_dim]
            
            # Compute the design matrix inverse term: (X_train^T X_train + alpha I)^(-1)
            XTX = torch.matmul(X_train.T, X_train)  # Shape: [input_dim, input_dim]
            I = torch.eye(self.input_dim, device=X_train.device)
            XTX_plus_alphaI = XTX + self.alpha * I
            try:
                XTX_plus_alphaI_inv = torch.inverse(XTX_plus_alphaI)  # Shape: [input_dim, input_dim]
            except RuntimeError:
                # Add small jitter for numerical stability if inversion fails
                XTX_plus_alphaI += 1e-6 * I
                XTX_plus_alphaI_inv = torch.inverse(XTX_plus_alphaI)

            # Predictive variance for each input: sigma^2 * (1 + X (X^T X + alpha I)^(-1) X^T)
            # First, estimate noise variance (sigma^2) from training residuals
            with torch.no_grad():
                y_train_pred = self(X_train)
                residuals = y_train_pred - X_train  # Assuming X_train includes targets for simplicity
                sigma2 = torch.mean(residuals ** 2)  # Scalar

            # Compute variance for new inputs
            X_inv_XT = torch.matmul(X, torch.matmul(XTX_plus_alphaI_inv, X.T))  # Shape: [batch_size, batch_size]
            pred_variance = sigma2 * (1 + torch.diagonal(X_inv_XT))  # Shape: [batch_size]
            std_preds = torch.sqrt(pred_variance).view(-1, 1)  # Match output shape
        else:
            # No uncertainty if X_train not provided
            std_preds = torch.zeros_like(mean_preds)

        return mean_preds, std_preds