# models/prediction/viability_prediction.py
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ViabilityPredictor(nn.Module):
    """
    Predict cell viability from integrated multimodal features.

    This module implements prediction heads for cell viability from
    integrated features. It supports multiple architectures, calibration
    options, and uncertainty estimation.

    Attributes:
        input_dim: Dimension of input features
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        activation: Activation function to use
        use_batch_norm: Whether to use batch normalization
        output_activation: Activation for the output (sigmoid, none)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True,
        output_activation: str = "sigmoid",
        uncertainty: bool = False,
    ):
        """
        Initialize the ViabilityPredictor.

        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            activation: Activation function ('relu', 'leaky_relu', 'elu', etc.)
            use_batch_norm: Whether to use batch normalization
            output_activation: Activation for the output ('sigmoid', 'none')
            uncertainty: Whether to estimate prediction uncertainty
        """
        super(ViabilityPredictor, self).__init__()

        self.input_dim = input_dim
        self.output_activation = output_activation.lower()
        self.uncertainty = uncertainty

        # Choose activation function
        if activation.lower() == "relu":
            self.activation_fn = nn.ReLU()
        elif activation.lower() == "leaky_relu":
            self.activation_fn = nn.LeakyReLU(0.1)
        elif activation.lower() == "elu":
            self.activation_fn = nn.ELU()
        elif activation.lower() == "gelu":
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Validate output activation
        valid_output_activations = ["sigmoid", "none"]
        if self.output_activation not in valid_output_activations:
            raise ValueError(
                f"Invalid output activation: {output_activation}. Must be one of {valid_output_activations}"
            )

        # Build network layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization (if requested)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(self.activation_fn)

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer(s)
        if uncertainty:
            # For uncertainty estimation, predict both mean and variance
            self.network = nn.Sequential(*layers)
            self.mean_head = nn.Linear(hidden_dims[-1], 1)
            self.var_head = nn.Sequential(
                nn.Linear(hidden_dims[-1], 1), nn.Softplus()  # Ensure positive variance
            )
        else:
            # Standard deterministic output
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)

        logger.debug(
            f"Initialized ViabilityPredictor with input_dim={input_dim}, "
            f"hidden_dims={hidden_dims}, uncertainty={uncertainty}"
        )

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for viability prediction.

        Args:
            x: Tensor [batch_size, input_dim] with integrated features

        Returns:
            If uncertainty is False: Tensor [batch_size, 1] with viability predictions
            If uncertainty is True: Tuple of (mean, variance) tensors
        """
        if self.uncertainty:
            # Process through shared network
            features = self.network(x)

            # Predict mean and variance
            mean = self.mean_head(features)
            var = self.var_head(features)

            # Apply output activation to mean if needed
            if self.output_activation == "sigmoid":
                mean = torch.sigmoid(mean)

            return mean, var
        else:
            # Standard deterministic prediction
            outputs = self.network(x)

            # Apply output activation if needed
            if self.output_activation == "sigmoid":
                outputs = torch.sigmoid(outputs)

            return outputs

    def predict_with_uncertainty(
        self, x: torch.Tensor, samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions with Monte Carlo dropout uncertainty estimation.

        This method uses MC dropout to estimate prediction uncertainty even for
        models not specifically trained with uncertainty outputs.

        Args:
            x: Input tensor [batch_size, input_dim]
            samples: Number of Monte Carlo samples

        Returns:
            Tuple of (mean_prediction, std_deviation) tensors
        """
        if self.uncertainty:
            # For models already trained with uncertainty
            mean, var = self.forward(x)
            return mean, torch.sqrt(var)

        # Enable dropout at inference time
        self.train()

        # Collect multiple predictions
        predictions = []
        for _ in range(samples):
            pred = self.forward(x)
            predictions.append(pred)

        # Restore eval mode
        self.eval()

        # Stack predictions [samples, batch_size, 1]
        stacked = torch.stack(predictions)

        # Calculate mean and standard deviation
        mean_pred = torch.mean(stacked, dim=0)
        std_dev = torch.std(stacked, dim=0)

        return mean_pred, std_dev


class MultiTaskViabilityPredictor(nn.Module):
    """
    Multi-task prediction head for viability and related metrics.

    This module predicts multiple related outputs (e.g., viability, IC50)
    from the same integrated features, leveraging shared representations
    for improved performance.

    Attributes:
        input_dim: Dimension of input features
        task_names: List of prediction task names
        shared_hidden_dims: Dimensions for shared hidden layers
        task_specific_dims: Dimensions for task-specific layers
    """

    def __init__(
        self,
        input_dim: int,
        task_names: List[str],
        shared_hidden_dims: List[int] = [256, 128],
        task_specific_dims: List[int] = [64],
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        """
        Initialize the MultiTaskViabilityPredictor.

        Args:
            input_dim: Dimension of input features
            task_names: List of prediction task names (e.g., ['viability', 'ic50'])
            shared_hidden_dims: Dimensions for shared hidden layers
            task_specific_dims: Dimensions for task-specific layers
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(MultiTaskViabilityPredictor, self).__init__()

        self.input_dim = input_dim
        self.task_names = task_names

        # Shared network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in shared_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.shared_network = nn.Sequential(*layers)

        # Task-specific network heads
        self.task_heads = nn.ModuleDict()
        for task in task_names:
            head_layers = []
            task_prev_dim = shared_hidden_dims[-1]

            for task_dim in task_specific_dims:
                head_layers.append(nn.Linear(task_prev_dim, task_dim))
                if use_batch_norm:
                    head_layers.append(nn.BatchNorm1d(task_dim))
                head_layers.append(nn.ReLU())
                head_layers.append(nn.Dropout(dropout))
                task_prev_dim = task_dim

            # Output layer for this task
            head_layers.append(nn.Linear(task_prev_dim, 1))

            # Add sigmoid for viability but not for IC50 (which is log-scaled)
            if task.lower() == "viability":
                head_layers.append(nn.Sigmoid())

            self.task_heads[task] = nn.Sequential(*head_layers)

        logger.debug(
            f"Initialized MultiTaskViabilityPredictor with tasks={task_names}, "
            f"input_dim={input_dim}, shared_dims={shared_hidden_dims}"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-task prediction.

        Args:
            x: Tensor [batch_size, input_dim] with integrated features

        Returns:
            Dictionary mapping task names to prediction tensors
        """
        # Process through shared network
        shared_features = self.shared_network(x)

        # Process through task-specific heads
        predictions = {}
        for task in self.task_names:
            predictions[task] = self.task_heads[task](shared_features)

        return predictions
