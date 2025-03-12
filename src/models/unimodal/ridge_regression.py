# src/models/unimodal/ridge_regression.py
import torch
import torch.nn as nn

class RidgeRegression(nn.Module):
    """
    Pure Ridge Regression model (linear model with L2 regularization)
    implemented in PyTorch.
    """
    def __init__(self, input_dim, output_dim=1, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.alpha = alpha  # L2 regularization strength
        
        # Initialize weights
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.linear(x)
    
    def get_param_groups(self):
        """
        Return parameter groups for optimizer configuration.
        Weight decay (L2 regularization) only applied to weights, not biases.
        """
        return [
            {'params': self.linear.weight, 'weight_decay': self.alpha},
            {'params': self.linear.bias, 'weight_decay': 0.0}
        ]