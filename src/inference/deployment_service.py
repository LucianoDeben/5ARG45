# inference/deployment_service.py
import json
import logging
import os
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.config_utils import load_config
from utils.storage import CheckpointManager

logger = logging.getLogger(__name__)


class DeploymentService:
    """
    Manages model deployment for serving predictions in multimodal drug response tasks.

    Handles model packaging, loading, optimization (e.g., quantization), and real-time inference,
    configured via a configuration dictionary.

    Attributes:
        model: PyTorch model instance.
        model_path: Path to save/load the model.
        metadata: Model metadata.
        device: Device for computation ('cuda' or 'cpu').
        config: Configuration dictionary for deployment parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the DeploymentService.

        Args:
            model: PyTorch model for deployment.
            model_path: Path to save/load the model.
            metadata: Optional model metadata dictionary.
            device: Device to use ('cuda', 'cpu', or None for auto-detection).
            config: Configuration dictionary (loads default if None).
        """
        self.config = config or load_config("config.yaml")
        self.model = model
        self.model_path = model_path
        self.metadata = metadata or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device and set to evaluation mode
        self.model.to(self.device).eval()

    def package_model(self, include_source: bool = False) -> str:
        """
        Package the model for deployment, including metadata and optional source code.

        Args:
            include_source: Whether to include model source code.

        Returns:
            Path to packaged model directory.
        """
        deploy_cfg = self.config.get("deployment", {})
        package_dir = os.path.dirname(self.model_path) or "models/deployed"
        os.makedirs(package_dir, exist_ok=True)

        # Save model state
        CheckpointManager.save(self.model.state_dict(), self.model_path)
        logger.info(f"Saved model state to {self.model_path}")

        # Save metadata
        metadata_path = os.path.join(package_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

        # Include source code if requested
        if include_source and hasattr(self.model, "__module__"):
            import importlib
            import inspect

            module = importlib.import_module(self.model.__module__)
            source = inspect.getsource(module)
            source_path = os.path.join(package_dir, "model_source.py")
            with open(source_path, "w") as f:
                f.write(source)
            logger.info(f"Saved model source to {source_path}")

        return package_dir

    @classmethod
    def load_packaged_model(
        cls,
        model_class: type,
        model_path: str,
        model_args: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> "DeploymentService":
        """
        Load a packaged model for deployment.

        Args:
            model_class: Model class to instantiate.
            model_path: Path to packaged model state.
            model_args: Arguments for model initialization.
            device: Device to use.
            config: Configuration dictionary.

        Returns:
            DeploymentService instance.
        """
        config = config or load_config("config.yaml")
        package_dir = os.path.dirname(model_path)

        # Load metadata
        metadata_path = os.path.join(package_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Initialize model
        model_args = model_args or {}
        model = model_class(**model_args)

        # Load model state
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        ).eval()

        return cls(model, model_path, metadata, device, config)

    def optimize_model(
        self, quantize: bool = False, dynamic_quantize: bool = False
    ) -> None:
        """
        Optimize the model for inference, including quantization options.

        Args:
            quantize: Whether to use static quantization.
            dynamic_quantize: Whether to use dynamic quantization.
        """
        deploy_cfg = self.config.get("deployment", {})
        self.model.eval()

        if quantize or dynamic_quantize:
            try:
                from torch.quantization import (
                    convert,
                    get_default_qconfig,
                    prepare,
                    quantize_dynamic,
                )

                # Choose quantization strategy
                if dynamic_quantize:
                    self.model = quantize_dynamic(
                        self.model,
                        {torch.nn.Linear},
                        dtype=torch.qint8,
                        inplace=False,
                    )
                    logger.info("Applied dynamic quantization")
                else:  # static quantization
                    qconfig = get_default_qconfig("fbgemm")
                    self.model = prepare(self.model, {""}, qconfig)
                    example_input = self._get_example_input()
                    self.model(example_input)  # calibrate
                    self.model = convert(self.model, inplace=False)
                    logger.info("Applied static quantization")

            except Exception as e:
                logger.error(f"Failed to optimize model: {str(e)}")
                raise

    def _get_example_input(self) -> Dict[str, torch.Tensor]:
        """Create example input for model optimization, aligned with config."""
        model_cfg = self.config.get("model", {})
        transcriptomics_dim = model_cfg.get("transcriptomics_output_dim", 978)
        chemical_dim = model_cfg.get("chemical_output_dim", 100)

        return {
            "transcriptomics": torch.randn(1, transcriptomics_dim, device=self.device),
            "molecular": torch.randn(1, chemical_dim, device=self.device),
        }

    def serve(self, data_loader: DataLoader) -> np.ndarray:
        """
        Serve predictions for a DataLoader, optimized for batch processing.

        Args:
            data_loader: DataLoader with data from MultimodalDrugDataset.

        Returns:
            Numpy array of predictions.
        """
        with torch.no_grad():
            all_outputs = []
            for batch in data_loader:
                if not isinstance(batch, dict) or "viability" not in batch:
                    raise ValueError(
                        "Batch must be a dict with 'viability' and multimodal keys"
                    )

                inputs = {
                    k: v.to(self.device) for k, v in batch.items() if k != "viability"
                }
                outputs = self.model(inputs)
                all_outputs.append(outputs.cpu().numpy())

        return np.concatenate(all_outputs)
