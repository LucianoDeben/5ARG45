# utils/logging_monitoring.py
import logging
import os
from typing import Any, Dict, Optional

import wandb

# Set up a logger instance for this module
logger = logging.getLogger(__name__)


class LoggingMonitoring:
    """Handles logging and monitoring for the Multimodal Drug Response Prediction Framework, with W&B integration."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Logging & Monitoring module.

        Args:
            config: Configuration dictionary from the Configuration Module containing experiment and logging settings.
        """
        self.config = config
        self.track = config.get("experiment", {}).get(
            "track", False
        )  # Toggle W&B tracking
        self.run = None

        # Configure local logging
        log_level = config.get("logging", {}).get("level", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),  # Log to console
                logging.FileHandler("training.log"),  # Log to file
            ],
        )

        # Initialize W&B if tracking is enabled
        if self.track:
            try:
                self.run = wandb.init(
                    project=config["experiment"]["project_name"],
                    name=config["experiment"]["run_name"],
                    config=config,  # Log entire config as hyperparameters
                )
                logger.info(f"Initialized W&B run: {self.run.name}")
            except Exception as e:
                logger.error(f"Failed to initialize W&B: {e}")
                self.track = False  # Disable W&B if initialization fails

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to both local logger and W&B.

        Args:
            metrics: Dictionary of metrics (e.g., {'loss': 0.5, 'accuracy': 0.9}).
            step: Optional step or epoch number for tracking progress.
        """
        # Log metrics locally
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")

        # Log to W&B if enabled
        if self.track and self.run:
            try:
                self.run.log(metrics, step=step)
            except Exception as e:
                logger.error(f"Failed to log metrics to W&B: {e}")

    def log_artifact(
        self, artifact_path: str, artifact_name: str, artifact_type: str
    ) -> None:
        """
        Log an artifact (e.g., model checkpoint, plot) to W&B.

        Args:
            artifact_path: Path to the artifact file.
            artifact_name: Name of the artifact.
            artifact_type: Type of artifact (e.g., 'model', 'plot').
        """
        if self.track and self.run:
            try:
                artifact = wandb.Artifact(artifact_name, type=artifact_type)
                artifact.add_file(artifact_path)
                self.run.log_artifact(artifact)
                logger.info(f"Logged artifact {artifact_name} to W&B")
            except Exception as e:
                logger.error(f"Failed to log artifact to W&B: {e}")
        else:
            logger.info(
                f"Artifact {artifact_name} saved locally at {artifact_path} (W&B disabled)"
            )

    def finish(self) -> None:
        """Clean up and finish the W&B run if tracking is enabled."""
        if self.track and self.run:
            try:
                self.run.finish()
                logger.info("Finished W&B run")
            except Exception as e:
                logger.error(f"Failed to finish W&B run: {e}")
