# inference/deployment.py
import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelDeployment:
    """
    Model deployment utilities for serving predictions.

    Features:
    - Model packaging for deployment
    - Serialization and deserialization
    - Optimized inference
    """

    def __init__(
        self,
        model: nn.Module,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize model deployment service.

        Args:
            model: PyTorch model
            model_path: Path to save/load model
            metadata: Model metadata
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model = model
        self.model_path = model_path
        self.metadata = metadata or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # Set model to evaluation mode
        self.model.eval()

    def package_model(self, include_source: bool = False) -> str:
        """
        Package model for deployment.

        Args:
            include_source: Whether to include model source code

        Returns:
            Path to packaged model
        """
        package_dir = os.path.dirname(self.model_path)
        os.makedirs(package_dir, exist_ok=True)

        # Save model state dict
        torch.save(self.model.state_dict(), self.model_path)

        # Save metadata
        metadata_path = os.path.join(package_dir, "metadata.json")
        import json

        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        # Include source code if requested
        if include_source and hasattr(self.model, "__module__"):
            import importlib
            import inspect

            module = importlib.import_module(self.model.__module__)
            source = inspect.getsource(module)

            source_path = os.path.join(package_dir, "model_source.py")
            with open(source_path, "w") as f:
                f.write(source)

        logger.info(f"Model packaged to {package_dir}")
        return self.model_path

    @classmethod
    def load_packaged_model(
        cls,
        model_class: type,
        model_path: str,
        model_args: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> "ModelDeployment":
        """
        Load packaged model.

        Args:
            model_class: Model class
            model_path: Path to model
            model_args: Arguments for model initialization
            device: Device to use

        Returns:
            ModelDeployment instance
        """
        package_dir = os.path.dirname(model_path)

        # Load metadata
        metadata_path = os.path.join(package_dir, "metadata.json")
        import json

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Initialize model
        model_args = model_args or {}
        model = model_class(**model_args)

        # Load model state
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        return cls(model, model_path, metadata, device)

    def optimize_model(self, quantize: bool = False) -> None:
        """
        Optimize model for inference.

        Args:
            quantize: Whether to quantize the model
        """
        self.model.eval()

        if quantize:
            # Quantize model
            try:
                # Static Quantization
                from torch.quantization import get_default_qconfig, quantize_jit

                qconfig = get_default_qconfig("fbgemm")

                # Prepare model for quantization
                if hasattr(self.model, "prepare_for_quantization"):
                    self.model.prepare_for_quantization(qconfig)

                # Trace model
                example_input = self._get_example_input()
                traced_model = torch.jit.trace(self.model, example_input)

                # Quantize model
                quantized_model = quantize_jit(
                    traced_model, {"": qconfig}, inplace=False
                )
                self.model = quantized_model

                logger.info("Model quantized successfully")
            except Exception as e:
                logger.error(f"Failed to quantize model: {str(e)}")

    def _get_example_input(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get example input for model optimization."""
        # Check if model has an example_input method
        if hasattr(self.model, "example_input"):
            return self.model.example_input(device=self.device)

        # Try to infer input format
        if (
            hasattr(self.model, "input_format")
            and self.model.input_format == "multimodal"
        ):
            # Multimodal input (dictionary)
            if hasattr(self.model, "transcriptomics_dim") and hasattr(
                self.model, "chemical_dim"
            ):
                return {
                    "transcriptomics": torch.randn(
                        1, self.model.transcriptomics_dim, device=self.device
                    ),
                    "molecular": torch.randn(
                        1, self.model.chemical_dim, device=self.device
                    ),
                }
            else:
                # Generic multimodal input
                return {
                    "transcriptomics": torch.randn(
                        1, 978, device=self.device
                    ),  # Landmark genes
                    "molecular": torch.randn(
                        1, 100, device=self.device
                    ),  # Arbitrary chemical features
                }
        else:
            # Default to single tensor input
            input_dim = getattr(
                self.model, "input_dim", 978
            )  # Default to landmark genes
            return torch.randn(1, input_dim, device=self.device)

    def serve(self, data: Union[Dict[str, Any], np.ndarray]) -> np.ndarray:
        """
        Serve model for inference.

        Args:
            data: Input data (dictionary or array)

        Returns:
            Numpy array of predictions
        """
        with torch.no_grad():
            # Preprocess input
            if isinstance(data, dict):
                # Convert to tensors and move to device
                inputs = {}
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        inputs[key] = torch.tensor(value, dtype=torch.float32).to(
                            self.device
                        )
                    elif isinstance(value, torch.Tensor):
                        inputs[key] = value.to(self.device)
                    else:
                        inputs[key] = value
            else:
                # Convert to tensor and move to device
                inputs = torch.tensor(data, dtype=torch.float32).to(self.device)

            # Add batch dimension if needed
            if isinstance(inputs, torch.Tensor) and inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)

            # Forward pass
            outputs = self.model(inputs)

            # Return as numpy array
            return outputs.cpu().numpy()
