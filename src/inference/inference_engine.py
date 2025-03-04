# inference/inference_engine.py
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.config_utils import load_config
from utils.storage import CheckpointManager

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Handles inference tasks for trained multimodal drug response models.

    Supports batch prediction, ensemble prediction, model export, and metadata integration,
    configured via a configuration dictionary and aligned with MultimodalDrugDataset.

    Attributes:
        model: PyTorch model instance.
        device: Device for computation ('cuda' or 'cpu').
        config: Configuration dictionary for inference parameters.
        ensemble_models: List of ensemble model instances.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        ensemble_paths: Optional[List[str]] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the InferenceEngine.

        Args:
            model: PyTorch model for inference.
            device: Device to use ('cuda', 'cpu', or None for auto-detection).
            checkpoint_path: Path to load a single model checkpoint.
            ensemble_paths: List of checkpoint paths for ensemble prediction.
            config: Configuration dictionary (loads default if None).
        """
        self.config = config or load_config("config.yaml")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

        # Load checkpoint if provided
        if checkpoint_path:
            checkpoint = CheckpointManager.load(checkpoint_path, self.model)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")

        # Set up ensemble models if provided
        self.ensemble_models = []
        infer_cfg = self.config.get("inference", {})
        max_ensemble = infer_cfg.get("max_ensemble", 5)
        if ensemble_paths:
            if len(ensemble_paths) > max_ensemble:
                logger.warning(f"Limiting ensemble to {max_ensemble} models")
                ensemble_paths = ensemble_paths[:max_ensemble]

            for i, path in enumerate(ensemble_paths):
                ensemble_model = (
                    type(model)(**(model.config if hasattr(model, "config") else {}))
                    .to(self.device)
                    .eval()
                )
                CheckpointManager.load(path, ensemble_model)
                self.ensemble_models.append(ensemble_model)
                logger.info(
                    f"Loaded ensemble model {i+1}/{len(ensemble_paths)} from {path}"
                )

    def predict(
        self,
        data_loader: DataLoader,
        use_ensemble: bool = False,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions for input data using a DataLoader.

        Args:
            data_loader: DataLoader with data from MultimodalDrugDataset.
            use_ensemble: Whether to use ensemble averaging for predictions.
            return_std: Whether to return prediction standard deviation (requires ensemble).

        Returns:
            Numpy array of predictions, or tuple of (predictions, std_devs) if return_std=True.
        """
        if return_std and not (use_ensemble and self.ensemble_models):
            logger.warning("Standard deviation requires ensemble; using single model")
            return_std = False

        all_outputs = []
        std_devs = [] if return_std else None

        with torch.no_grad():
            for batch in data_loader:
                if not isinstance(batch, dict) or "viability" not in batch:
                    raise ValueError(
                        "Batch must be a dict with 'viability' and multimodal keys"
                    )

                inputs = {
                    k: v.to(self.device) for k, v in batch.items() if k != "viability"
                }

                if use_ensemble and self.ensemble_models:
                    ensemble_outputs = []
                    outputs = self.model(inputs)
                    ensemble_outputs.append(outputs.cpu().numpy())

                    for model in self.ensemble_models:
                        outputs = model(inputs)
                        ensemble_outputs.append(outputs.cpu().numpy())

                    stacked_outputs = np.stack(ensemble_outputs, axis=0)
                    predictions = np.mean(stacked_outputs, axis=0)

                    if return_std:
                        std_devs.append(np.std(stacked_outputs, axis=0))
                else:
                    outputs = self.model(inputs)
                    predictions = outputs.cpu().numpy()

                all_outputs.append(predictions)

        predictions = np.concatenate(all_outputs)
        return (predictions, np.concatenate(std_devs)) if return_std else predictions

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
            data_loader: DataLoader with data from MultimodalDrugDataset.
            metadata_cols: List of metadata column names from dataset.metadata.
            use_ensemble: Whether to use ensemble prediction.
            output_path: Path to save predictions CSV (defaults to config if None).

        Returns:
            DataFrame with predictions, standard deviations (if ensemble), and metadata.
        """
        if not hasattr(data_loader.dataset, "metadata"):
            raise ValueError("DataLoader dataset must have metadata attribute")

        metadata = data_loader.dataset.metadata
        missing_cols = [col for col in metadata_cols if col not in metadata.columns]
        if missing_cols:
            raise ValueError(f"Missing metadata columns: {missing_cols}")

        infer_cfg = self.config.get("inference", {})
        output_path = output_path or infer_cfg.get(
            "output_path", "results/predictions.csv"
        )

        predictions = self.predict(
            data_loader, use_ensemble=use_ensemble, return_std=use_ensemble
        )
        result_dict = {"prediction": predictions.flatten()}

        if use_ensemble and isinstance(predictions, tuple):
            predictions, std_devs = predictions
            result_dict["std_dev"] = std_devs.flatten()

        for col in metadata_cols:
            result_dict[col] = metadata[col].values

        result_df = pd.DataFrame(result_dict)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)
            logger.info(f"Saved predictions to {output_path}")

        return result_df

    def export_model(self, output_path: str, format: str = "pytorch") -> str:
        """
        Export the model for deployment.

        Args:
            output_path: Path to save the exported model.
            format: Export format ('pytorch', 'onnx', 'torchscript').

        Returns:
            Path to exported model.
        """
        infer_cfg = self.config.get("inference", {})
        supported_formats = infer_cfg.get(
            "export_formats", ["pytorch", "onnx", "torchscript"]
        )
        if format not in supported_formats:
            raise ValueError(
                f"Unsupported export format: {format}. Use {supported_formats}"
            )

        if format == "pytorch":
            torch.save(self.model.state_dict(), output_path)
            if hasattr(self.model, "config"):
                config_path = os.path.splitext(output_path)[0] + "_config.json"
                with open(config_path, "w") as f:
                    json.dump(self.model.config, f, indent=2)
            logger.info(f"Exported PyTorch model to {output_path}")

        elif format == "onnx":
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
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                logger.info(f"Exported ONNX model to {output_path}")
            except ImportError:
                logger.error("ONNX export requires the 'onnx' package")
                raise

        elif format == "torchscript":
            dummy_input = self._create_dummy_input()
            traced_script_module = torch.jit.trace(self.model, dummy_input)
            traced_script_module.save(output_path)
            logger.info(f"Exported TorchScript model to {output_path}")

        return output_path

    def _create_dummy_input(self) -> Dict[str, torch.Tensor]:
        """Create dummy input for model export, aligned with config."""
        model_cfg = self.config.get("model", {})
        transcriptomics_dim = model_cfg.get("transcriptomics_output_dim", 978)
        chemical_dim = model_cfg.get("chemical_output_dim", 100)

        return {
            "transcriptomics": torch.randn(1, transcriptomics_dim, device=self.device),
            "molecular": torch.randn(1, chemical_dim, device=self.device),
        }
