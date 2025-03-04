#!/usr/bin/env python
# main.py - Multimodal Drug Response Prediction Framework

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim

from .config.config_utils import get_paths, init_wandb, load_config, setup_logging
from .data.datasets import DatasetFactory
from .data.feature_transforms import create_feature_transform
from .data.loaders import GCTXDataLoader
from .data.preprocessing_transforms import create_preprocessing_transform
from .evaluation.evaluator import Evaluator
from .models.chemical.descriptors import MolecularDescriptorEncoder
from .models.integration.fusion import FeatureFusion
from .models.prediction.viability_prediction import ViabilityPredictor
from .models.transcriptomics.encoders import TranscriptomicEncoder
from .training.trainer import MultiRunTrainer
from .utils.logging import ExperimentLogger


class MultimodalDrugResponseModel(nn.Module):
    """
    Multimodal neural network for drug response prediction.

    Combines transcriptomics and chemical feature encoders with a fusion module
    and prediction head to predict cell viability.
    """

    def __init__(
        self,
        transcriptomics_input_dim: int = 978,
        chemical_input_dim: int = 1025,
        transcriptomics_hidden_dims: list = [512, 256],
        chemical_hidden_dims: list = [256, 128],
        transcriptomics_output_dim: int = 128,
        chemical_output_dim: int = 128,
        fusion_output_dim: int = 128,
        fusion_strategy: str = "concat",
        predictor_hidden_dims: list = [64, 32],
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        """Initialize the multimodal network."""
        super(MultimodalDrugResponseModel, self).__init__()

        # Store configuration
        self.config = {
            "transcriptomics_input_dim": transcriptomics_input_dim,
            "chemical_input_dim": chemical_input_dim,
            "transcriptomics_hidden_dims": transcriptomics_hidden_dims,
            "chemical_hidden_dims": chemical_hidden_dims,
            "transcriptomics_output_dim": transcriptomics_output_dim,
            "chemical_output_dim": chemical_output_dim,
            "fusion_output_dim": fusion_output_dim,
            "fusion_strategy": fusion_strategy,
            "predictor_hidden_dims": predictor_hidden_dims,
            "dropout": dropout,
            "activation": activation,
        }

        # Transcriptomics encoder
        self.transcriptomics_encoder = TranscriptomicEncoder(
            input_dim=transcriptomics_input_dim,
            hidden_dims=transcriptomics_hidden_dims,
            output_dim=transcriptomics_output_dim,
            dropout=dropout,
            activation=activation,
            normalize=True,
        )

        # Chemical encoder (handles molecular descriptors with dosage)
        self.chemical_encoder = MolecularDescriptorEncoder(
            input_dim=chemical_input_dim,
            hidden_dims=chemical_hidden_dims,
            output_dim=chemical_output_dim,
            dropout=dropout,
            activation=activation,
        )

        # Fusion module
        if fusion_strategy == "concat":
            fusion_input_dim = transcriptomics_output_dim + chemical_output_dim
        else:
            fusion_input_dim = max(transcriptomics_output_dim, chemical_output_dim)

        self.fusion = FeatureFusion(
            t_dim=transcriptomics_output_dim,
            c_dim=chemical_output_dim,
            output_dim=fusion_output_dim,
            strategy=fusion_strategy,
            dropout=dropout,
        )

        # Prediction head
        self.predictor = ViabilityPredictor(
            input_dim=fusion_output_dim,
            hidden_dims=predictor_hidden_dims,
            dropout=dropout,
            activation=activation,
            output_activation="sigmoid",  # Use sigmoid for viability prediction
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for multimodal input."""
        # Extract inputs
        transcriptomics = inputs["transcriptomics"]
        molecular = inputs["molecular"]

        # Process through encoders
        transcriptomics_features = self.transcriptomics_encoder(transcriptomics)
        chemical_features = self.chemical_encoder(molecular)

        # Fuse features
        fused_features = self.fusion(transcriptomics_features, chemical_features)

        # Make prediction
        predictions = self.predictor(fused_features)

        return predictions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multimodal Drug Response Prediction")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of training runs")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs per run"
    )
    parser.add_argument(
        "--nrows", type=int, default=1000, help="Number of data rows to use"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Set up logging
    setup_logging()

    # Load configuration
    config = load_config(args.config)

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Set up paths
    paths = get_paths(config)
    output_dir = args.output_dir or paths["results_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamp for run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize experiment logger
    exp_logger = ExperimentLogger(
        experiment_name=f"multimodal_drug_response_{timestamp}",
        config=config,
        log_dir=str(paths["log_dir"]),
        use_tensorboard=True,
        use_wandb=not args.no_wandb,
        wandb_project=config["experiment"]["project_name"],
    )

    # Initialize W&B if enabled
    if not args.no_wandb:
        init_wandb(config)

    # Load data
    logging.info("Loading data...")
    gctx_file = config["data"]["gctx_file"]

    # Initialize data loaders with chunking
    gctx_loader = GCTXDataLoader(gctx_file, preload_metadata=True)

    # Create preprocessing transforms
    preprocess_transform = create_preprocessing_transform("scale")

    # Create feature transforms
    molecular_transform = create_feature_transform(
        "fingerprint",
        fingerprint_size=1024,
        fingerprint_radius=2,
    )

    # Create datasets
    logging.info("Creating datasets...")
    train_ds, val_ds, test_ds = DatasetFactory.create_and_split_multimodal(
        gctx_loader=gctx_loader,
        feature_space="landmark",  # Use landmark genes only
        nrows=args.nrows,  # Limit to specified number of rows
        test_size=0.2,
        val_size=0.1,
        random_state=args.seed,
        group_by="cell_id",  # Group by cell line to prevent data leakage
        stratify_by="viability",  # Stratify by outcome
        transform_molecular=molecular_transform,
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Define model architecture
    model_kwargs = {
        "transcriptomics_input_dim": train_ds[0]["transcriptomics"].shape[0],
        "chemical_input_dim": train_ds[0]["molecular"].shape[0],
        "transcriptomics_hidden_dims": [512, 256],
        "chemical_hidden_dims": [256, 128],
        "transcriptomics_output_dim": 128,
        "chemical_output_dim": 128,
        "fusion_output_dim": 128,
        "fusion_strategy": "concat",
        "predictor_hidden_dims": [64, 32],
        "dropout": 0.3,
        "activation": "relu",
    }

    # Define optimizer and scheduler classes
    optimizer_class = torch.optim.Adam
    scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau

    # Define loss function
    criterion = nn.MSELoss()

    # Create multi-run trainer
    logging.info(f"Setting up multi-run training ({args.runs} runs)...")
    multi_trainer = MultiRunTrainer(
        model_class=MultimodalDrugResponseModel,
        model_kwargs=model_kwargs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer_class=optimizer_class,
        scheduler_class=scheduler_class,
        criterion=criterion,
        exp_logger=exp_logger,
        config=config,
        num_runs=args.runs,
        save_models=True,
        output_dir=output_dir,
    )

    # Run multi-run training
    logging.info(f"Starting multi-run training for {args.epochs} epochs per run...")
    aggregate_results = multi_trainer.run_training(epochs=args.epochs)

    # Log aggregate results
    logging.info("Multi-run training completed. Aggregate results:")
    for phase in ["val", "test"]:
        logging.info(f"\n{phase.upper()} METRICS:")
        for metric, value in aggregate_results[phase].items():
            logging.info(f"  {metric}: {value:.4f}")

    # Create evaluator and run additional evaluations
    logging.info("Running detailed evaluation...")
    evaluator = Evaluator(
        model=multi_trainer.trained_models[
            0
        ],  # Use the first trained model for basic evaluation
        exp_logger=exp_logger,
        config=config,
    )

    # Evaluate by cell lines
    cell_results = evaluator.evaluate_by_group(
        test_loader,
        group_column="cell_id",
        output_dir=os.path.join(output_dir, "evaluation"),
    )

    # Multi-model evaluation for statistical analysis
    multi_model_results = evaluator.multi_run_evaluate(
        models=multi_trainer.trained_models,
        data_loader=test_loader,
        output_dir=os.path.join(output_dir, "evaluation"),
        prefix="final_",
    )

    logging.info("Evaluation completed. Results saved to output directory.")

    # Close experiment logger
    exp_logger.close()
    logging.info(f"Complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
