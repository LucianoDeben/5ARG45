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
    ):
        """
        Initialize the Ridge Regression model.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output (default 1 for regression).
            alpha (float): L2 regularization strength.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha  # L2 regularization strength
        self.linear = nn.Linear(input_dim, output_dim)
        self.X_train_cached = None  # Cache for training data

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
        # Process input to get tensor x
        if isinstance(x, dict):
            # Handle multimodal inputs by concatenating or processing as needed
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
            
        # Cache training data during training
        if self.training and self.X_train_cached is None:
            self.X_train_cached = processed_x.detach().clone()
            
        return self.linear(processed_x)

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

    def cache_training_data(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Cache training data for uncertainty estimation.
        
        Args:
            X (torch.Tensor): Input features.
            y (Optional[torch.Tensor]): Target values.
        """
        self.X_train_cached = X
        if y is not None:
            self.y_train_cached = y