import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import (
    DeepLift,
    FeatureAblation,
    GradientShap,
    IntegratedGradients,
    Saliency,
    ShapleyValueSampling,
)

logger = logging.getLogger(__name__)


class FeatureInterpreter:
    """
    Interpret feature importance for model predictions using Captum.

    This class provides multiple attribution methods to explain model predictions
    and identify the most important features for each modality.

    Attributes:
        model: The PyTorch model to interpret
        modality_dims: Dictionary mapping modality names to feature dimensions
        attribution_methods: Dictionary of available attribution methods
    """

    def __init__(
        self,
        model: torch.nn.Module,
        modality_dims: Optional[Dict[str, Tuple[int, int]]] = None,
        device: str = "cpu",
    ):
        """
        Initialize the FeatureInterpreter.

        Args:
            model: The PyTorch model to interpret
            modality_dims: Dictionary mapping modality names to (start_idx, end_idx) tuples
                          for interpreting specific modalities
            device: Device to use for computations ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.modality_dims = modality_dims

        # Initialize attribution methods
        self.attribution_methods = {
            "integrated_gradients": IntegratedGradients(model),
            "saliency": Saliency(model),
            "deep_lift": DeepLift(model),
            "gradient_shap": GradientShap(model),
            "feature_ablation": FeatureAblation(model),
            "shapley_sampling": ShapleyValueSampling(model),
        }

        logger.debug(
            f"Initialized FeatureInterpreter with {len(self.attribution_methods)} "
            f"attribution methods, modalities: {list(modality_dims.keys()) if modality_dims else 'None'}"
        )

    def compute_attributions(
        self,
        inputs: torch.Tensor,
        method: str = "integrated_gradients",
        target: Optional[int] = None,
        n_steps: int = 50,
        n_samples: int = 25,
    ) -> torch.Tensor:
        """
        Compute feature attributions using the specified method.

        Args:
            inputs: Input tensor [batch_size, features]
            method: Attribution method to use
            target: Target for attribution (e.g., class index)
            n_steps: Number of steps for IntegratedGradients
            n_samples: Number of samples for sampling-based methods

        Returns:
            Tensor with attribution scores for each feature
        """
        # Ensure input requires gradient
        inputs = inputs.clone().detach().to(self.device).requires_grad_(True)

        # Set model to eval mode
        self.model.eval()

        # Get the appropriate attribution method
        if method not in self.attribution_methods:
            raise ValueError(
                f"Unsupported attribution method: {method}. Available methods: {list(self.attribution_methods.keys())}"
            )

        attribution_fn = self.attribution_methods[method]

        # Compute attributions based on method
        try:
            if method == "integrated_gradients":
                # Create baseline (zeros)
                baseline = torch.zeros_like(inputs).to(self.device)
                attributions = attribution_fn.attribute(
                    inputs,
                    baselines=baseline,
                    target=target,
                    n_steps=n_steps,
                )
            elif method == "saliency":
                attributions = attribution_fn.attribute(inputs, target=target)
            elif method == "deep_lift":
                baseline = torch.zeros_like(inputs).to(self.device)
                attributions = attribution_fn.attribute(
                    inputs, baselines=baseline, target=target
                )
            elif method == "gradient_shap":
                baseline = torch.zeros_like(inputs).to(self.device)
                attributions = attribution_fn.attribute(
                    inputs,
                    baselines=baseline,
                    n_samples=n_samples,
                    stdevs=0.1,
                    target=target,
                )
            elif method == "feature_ablation":
                attributions = attribution_fn.attribute(inputs, target=target)
            elif method == "shapley_sampling":
                baseline = torch.zeros_like(inputs).to(self.device)
                attributions = attribution_fn.attribute(
                    inputs,
                    baselines=baseline,
                    n_samples=n_samples,
                    target=target,
                )

            logger.debug(
                f"Computed attributions using {method}, shape: {attributions.shape}"
            )
            return attributions

        except Exception as e:
            logger.error(f"Error computing attributions with {method}: {str(e)}")
            raise

    def explain(
        self,
        inputs: torch.Tensor,
        method: str = "integrated_gradients",
        target: Optional[int] = None,
        modality: Optional[str] = None,
        top_k: int = 10,
        visualize: bool = True,
    ) -> Dict[str, Union[np.ndarray, plt.Figure]]:
        """
        Generate and optionally visualize feature importance scores.

        Args:
            inputs: Input tensor to explain
            method: Attribution method to use
            target: Target for attribution (e.g., class index)
            modality: Specific modality to explain (or None for all)
            top_k: Number of top features to highlight
            visualize: Whether to generate visualization

        Returns:
            Dictionary with attribution results and visualizations
        """
        # Compute attributions
        attributions = self.compute_attributions(inputs, method, target)
        attributions = attributions.detach().cpu().numpy()

        # Apply modality filter if specified
        if modality is not None and self.modality_dims is not None:
            if modality not in self.modality_dims:
                raise ValueError(
                    f"Unknown modality: {modality}. Available: {list(self.modality_dims.keys())}"
                )

            start_idx, end_idx = self.modality_dims[modality]
            modality_attributions = attributions[:, start_idx:end_idx]
            modality_mask = np.zeros_like(attributions)
            modality_mask[:, start_idx:end_idx] = modality_attributions
            attributions = modality_mask

        # Calculate absolute attributions and find top-k features
        abs_attributions = np.abs(attributions)
        batch_size = abs_attributions.shape[0]

        # Calculate mean across batch
        mean_attributions = np.mean(abs_attributions, axis=0)
        top_k_indices = np.argsort(mean_attributions)[-top_k:][::-1]
        top_k_values = mean_attributions[top_k_indices]

        result = {
            "attributions": attributions,
            "mean_attributions": mean_attributions,
            "top_k_indices": top_k_indices,
            "top_k_values": top_k_values,
        }

        # Generate visualization if requested
        if visualize:
            fig = self._visualize_attributions(
                attributions, top_k_indices, top_k_values, method, modality
            )
            result["visualization"] = fig

        return result

    def _visualize_attributions(
        self,
        attributions: np.ndarray,
        top_k_indices: np.ndarray,
        top_k_values: np.ndarray,
        method: str,
        modality: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create visualization of attribution results.

        Args:
            attributions: Attribution values
            top_k_indices: Indices of top-k features
            top_k_values: Values of top-k features
            method: Attribution method used
            modality: Modality being visualized (if applicable)

        Returns:
            Matplotlib figure with visualization
        """
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot heatmap of attributions for first sample
        abs_attr = np.abs(attributions[0])
        im = ax1.imshow(abs_attr.reshape(1, -1), cmap="viridis", aspect="auto")
        ax1.set_title(f"Attribution Heatmap ({method})")
        ax1.set_xlabel("Feature Index")
        fig.colorbar(im, ax=ax1, label="Attribution Magnitude")

        # Plot top-k features
        ax2.barh(range(len(top_k_indices)), top_k_values)
        ax2.set_yticks(range(len(top_k_indices)))
        ax2.set_yticklabels([f"Feature {idx}" for idx in top_k_indices])
        ax2.set_title(f"Top {len(top_k_indices)} Features")
        ax2.set_xlabel("Mean Attribution")

        # Set overall title
        title = f"Feature Attributions - {method.replace('_', ' ').title()}"
        if modality:
            title += f" - {modality}"
        fig.suptitle(title)

        fig.tight_layout()
        return fig


class MultimodalInterpreter:
    """
    Interpret feature importance across different modalities.

    This class extends FeatureInterpreter to provide modality-specific
    interpretation and visualization for multimodal models.

    Attributes:
        model: The PyTorch model to interpret
        modality_dims: Dictionary mapping modality names to feature dimensions
        attribute_modalities: Whether to attribute each modality separately
    """

    def __init__(
        self,
        model: torch.nn.Module,
        modality_dims: Dict[str, Tuple[int, int]],
        device: str = "cpu",
    ):
        """
        Initialize the MultimodalInterpreter.

        Args:
            model: The PyTorch model to interpret
            modality_dims: Dictionary mapping modality names to (start_idx, end_idx) tuples
            device: Device to use for computations
        """
        self.model = model
        self.modality_dims = modality_dims
        self.device = device

        # Create base interpreter
        self.interpreter = FeatureInterpreter(model, modality_dims, device)

        logger.debug(
            f"Initialized MultimodalInterpreter for {len(modality_dims)} modalities: "
            f"{list(modality_dims.keys())}"
        )

    def explain_modalities(
        self,
        inputs: torch.Tensor,
        method: str = "integrated_gradients",
        target: Optional[int] = None,
        top_k: int = 10,
        visualize: bool = True,
    ) -> Dict[str, Dict]:
        """
        Generate explanations for each modality.

        Args:
            inputs: Input tensor to explain
            method: Attribution method to use
            target: Target for attribution
            top_k: Number of top features to highlight
            visualize: Whether to generate visualizations

        Returns:
            Dictionary mapping modality names to attribution results
        """
        # Get global attributions
        global_results = self.interpreter.explain(
            inputs, method, target, modality=None, top_k=top_k, visualize=visualize
        )

        # Get modality-specific attributions
        modality_results = {"global": global_results}

        for modality in self.modality_dims.keys():
            modality_result = self.interpreter.explain(
                inputs,
                method,
                target,
                modality=modality,
                top_k=top_k,
                visualize=visualize,
            )
            modality_results[modality] = modality_result

        return modality_results

    def compare_modalities(
        self,
        inputs: torch.Tensor,
        method: str = "integrated_gradients",
        target: Optional[int] = None,
    ) -> Dict[str, Union[Dict, plt.Figure]]:
        """
        Compare the importance of different modalities.

        Args:
            inputs: Input tensor to explain
            method: Attribution method to use
            target: Target for attribution

        Returns:
            Dictionary with modality comparison results and visualization
        """
        # Get attributions
        attributions = self.interpreter.compute_attributions(inputs, method, target)
        attributions = attributions.detach().cpu().numpy()

        # Calculate importance per modality
        modality_importance = {}
        for modality, (start_idx, end_idx) in self.modality_dims.items():
            modality_attr = attributions[:, start_idx:end_idx]
            modality_importance[modality] = np.mean(np.abs(modality_attr))

        # Normalize to percentages
        total_importance = sum(modality_importance.values())
        modality_percentages = {
            modality: (importance / total_importance) * 100
            for modality, importance in modality_importance.items()
        }

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        modalities = list(modality_percentages.keys())
        percentages = [modality_percentages[m] for m in modalities]

        ax.bar(modalities, percentages)
        ax.set_ylabel("Relative Importance (%)")
        ax.set_title(
            f"Modality Importance Comparison - {method.replace('_', ' ').title()}"
        )

        for i, p in enumerate(percentages):
            ax.annotate(f"{p:.1f}%", (i, p + 1), ha="center")

        # Return results
        result = {
            "modality_importance": modality_importance,
            "modality_percentages": modality_percentages,
            "visualization": fig,
        }

        return result
