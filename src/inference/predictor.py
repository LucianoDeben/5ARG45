# inference/predictor.py
import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.storage import ModelCheckpoint

logger = logging.getLogger(__name__)


class Predictor:
    """
    Predictor for making inferences with trained models.

    Features:
    - Batch prediction
    - Model loading from checkpoints
    - Prediction export and visualization
    - Ensemble prediction support
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        ensemble_paths: Optional[List[str]] = None,
    ):
        """
        Initialize the predictor.

        Args:
            model: PyTorch model
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            checkpoint_path: Path to model checkpoint
            ensemble_paths: List of checkpoint paths for ensemble prediction
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Set up model
        self.model = model.to(self.device)

        # Load checkpoint if provided
        if checkpoint_path:
            checkpoint = ModelCheckpoint.load_checkpoint(
                checkpoint_path, model=self.model, map_location=self.device
            )
            logger.info(f"Loaded checkpoint from {checkpoint_path}")

        # Set up ensemble models if provided
        self.ensemble_models = []
        if ensemble_paths:
            for i, path in enumerate(ensemble_paths):
                # Clone the model for ensemble
                ensemble_model = type(model)(
                    **model.config if hasattr(model, "config") else {}
                )
                ensemble_model = ensemble_model.to(self.device)

                # Load checkpoint
                checkpoint = ModelCheckpoint.load_checkpoint(
                    path, model=ensemble_model, map_location=self.device
                )

                self.ensemble_models.append(ensemble_model)
                logger.info(
                    f"Loaded ensemble model {i+1}/{len(ensemble_paths)} from {path}"
                )

        # Set models to evaluation mode
        self.model.eval()
        for model in self.ensemble_models:
            model.eval()

    def predict(
        self,
        data_loader: DataLoader,
        use_ensemble: bool = False,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions for input data.

        Args:
            data_loader: DataLoader with input data
            use_ensemble: Whether to use ensemble prediction
            return_std: Whether to return prediction standard deviation (requires ensemble)

        Returns:
            Numpy array of predictions, or tuple of (predictions, std_devs) if return_std=True
        """
        if return_std and not (use_ensemble and self.ensemble_models):
            logger.warning(
                "Standard deviation requires ensemble prediction; using single model"
            )
            return_std = False

        all_outputs = []
        std_devs = [] if return_std else None

        with torch.no_grad():
            for batch in data_loader:
                # Get batch data
                if isinstance(batch, dict):
                    # For MultimodalDataset returning dictionary
                    inputs = batch
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # For standard (X, y) tuple
                    inputs, _ = batch
                else:
                    inputs = batch

                # Move data to device
                if isinstance(inputs, dict):
                    inputs = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }
                else:
                    inputs = inputs.to(self.device)

                if use_ensemble and self.ensemble_models:
                    # Collect predictions from all ensemble models
                    ensemble_outputs = []

                    # Main model prediction
                    outputs = self.model(inputs)
                    ensemble_outputs.append(outputs.cpu().numpy())

                    # Ensemble models predictions
                    for model in self.ensemble_models:
                        outputs = model(inputs)
                        ensemble_outputs.append(outputs.cpu().numpy())

                    # Stack predictions
                    stacked_outputs = np.stack(ensemble_outputs, axis=0)

                    # Calculate mean and std
                    mean_outputs = np.mean(stacked_outputs, axis=0)
                    if return_std:
                        batch_std = np.std(stacked_outputs, axis=0)
                        std_devs.append(batch_std)

                    all_outputs.append(mean_outputs)
                else:
                    # Single model prediction
                    outputs = self.model(inputs)
                    all_outputs.append(outputs.cpu().numpy())

        # Concatenate predictions
        predictions = np.concatenate(all_outputs)

        if return_std:
            std_deviations = np.concatenate(std_devs)
            return predictions, std_deviations
        else:
            return predictions

    def predict_with_metadata(
        self,
        data_loader: DataLoader,
        metadata_cols: List[str],
        use_ensemble: bool = False,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions with associated metadata.

        Args:
            data_loader: DataLoader with input data
            metadata_cols: List of metadata columns to include
            use_ensemble: Whether to use ensemble prediction
            output_path: Path to save predictions CSV

        Returns:
            DataFrame with predictions and metadata
        """
        # Ensure DataLoader dataset has metadata attribute
        if not hasattr(data_loader.dataset, "metadata"):
            raise ValueError("DataLoader dataset must have metadata attribute")

        metadata = data_loader.dataset.metadata

        # Validate metadata columns
        missing_cols = [col for col in metadata_cols if col not in metadata.columns]
        if missing_cols:
            raise ValueError(f"Missing metadata columns: {missing_cols}")

        # Generate predictions
        if use_ensemble and self.ensemble_models:
            predictions, std_devs = self.predict(
                data_loader, use_ensemble=True, return_std=True
            )
        else:
            predictions = self.predict(data_loader, use_ensemble=False)
            std_devs = None

        # Create result DataFrame
        result_dict = {"prediction": predictions.flatten()}

        if std_devs is not None:
            result_dict["std_dev"] = std_devs.flatten()

        # Add metadata columns
        for col in metadata_cols:
            result_dict[col] = metadata[col].values

        result_df = pd.DataFrame(result_dict)

        # Save to CSV if requested
        if output_path:
            result_df.to_csv(output_path, index=False)
            logger.info(f"Saved predictions to {output_path}")

        return result_df

    def predict_batch(
        self,
        batch: Union[Dict[str, torch.Tensor], torch.Tensor],
        use_ensemble: bool = False,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions for a single batch.

        Args:
            batch: Input batch (dictionary or tensor)
            use_ensemble: Whether to use ensemble prediction
            return_std: Whether to return prediction standard deviation

        Returns:
            Numpy array of predictions, or tuple of (predictions, std_devs)
        """
        with torch.no_grad():
            # Move data to device
            if isinstance(batch, dict):
                inputs = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            else:
                inputs = batch.to(self.device)

            if use_ensemble and self.ensemble_models:
                # Collect predictions from all ensemble models
                ensemble_outputs = []

                # Main model prediction
                outputs = self.model(inputs)
                ensemble_outputs.append(outputs.cpu().numpy())

                # Ensemble models predictions
                for model in self.ensemble_models:
                    outputs = model(inputs)
                    ensemble_outputs.append(outputs.cpu().numpy())

                # Stack predictions
                stacked_outputs = np.stack(ensemble_outputs, axis=0)

                # Calculate mean and std
                predictions = np.mean(stacked_outputs, axis=0)

                if return_std:
                    std_deviations = np.std(stacked_outputs, axis=0)
                    return predictions, std_deviations
                else:
                    return predictions
            else:
                # Single model prediction
                outputs = self.model(inputs)
                return outputs.cpu().numpy()

    def export_model(self, output_path: str, format: str = "pytorch") -> str:
        """
        Export the model for deployment.

        Args:
            output_path: Path to save the exported model
            format: Export format ('pytorch', 'onnx', 'torchscript')

        Returns:
            Path to exported model
        """
        if format == "pytorch":
            # Export PyTorch model
            torch.save(self.model.state_dict(), output_path)
            logger.info(f"Exported PyTorch model to {output_path}")

            # Save model architecture
            if hasattr(self.model, "config"):
                config_path = os.path.splitext(output_path)[0] + "_config.json"
                with open(config_path, "w") as f:
                    json.dump(self.model.config, f, indent=2)
                logger.info(f"Saved model configuration to {config_path}")

        elif format == "onnx":
            # Export to ONNX format
            try:
                import onnx

                dummy_input = self._create_dummy_input()

                torch.onnx.export(
                    self.model,
                    dummy_input,
                    output_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "output": {0: "batch_size"},
                    },
                )

                # Verify the model
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)

                logger.info(f"Exported ONNX model to {output_path}")
            except ImportError:
                logger.error("ONNX export requires the 'onnx' package")
                raise

        elif format == "torchscript":
            # Export to TorchScript
            dummy_input = self._create_dummy_input()

            traced_script_module = torch.jit.trace(self.model, dummy_input)
            traced_script_module.save(output_path)

            logger.info(f"Exported TorchScript model to {output_path}")

        else:
            raise ValueError(f"Unsupported export format: {format}")

        return output_path

    def _create_dummy_input(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Create dummy input for model export."""
        # Check if model has a dummy_input method
        if hasattr(self.model, "dummy_input"):
            return self.model.dummy_input(batch_size=1, device=self.device)

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
