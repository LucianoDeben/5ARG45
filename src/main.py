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

from src.config.config_utils import get_paths, init_wandb, load_config, setup_logging
from src.data.datasets import DatasetFactory
from src.data.feature_transforms import (
    MorganFingerprintTransform,
    create_feature_transform,
)
from src.data.loaders import GCTXDataLoader
from src.evaluation.evaluator import Evaluator
from src.models.multimodal_models import MultimodalViabilityPredictor
from src.training.trainer import MultiRunTrainer
from src.utils.logging import ExperimentLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multimodal Drug Response Prediction")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of training runs")
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


def configure_environment(args, config: Dict[str, Any]):
    """Configure computational environment."""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device


def prepare_data(args, config: Dict[str, Any]):
    """Prepare datasets for training."""
    logging.info("Loading data...")
    gctx_file = config["data"]["gctx_file"]

    gctx_loader = GCTXDataLoader(gctx_file, preload_metadata=True)

    logging.info("Creating datasets...")
    transform_molecular = create_feature_transform(
        "fingerprint", fingerprint_size=1024, fingerprint_radius=2
    )

    try:
        train_ds, val_ds, test_ds = DatasetFactory.create_and_split_multimodal(
            gctx_loader=gctx_loader,
            feature_space="landmark",
            nrows=args.nrows,
            test_size=0.2,
            val_size=0.1,
            random_state=args.seed,
            group_by="cell_mfc_name",
            stratify_by="viability",
            transform_transcriptomics=None,  # No scaling needed for Mod Z scores
            transform_molecular=transform_molecular,
        )
        logging.info(
            f"Created datasets with {len(train_ds)}/{len(val_ds)}/{len(test_ds)} samples"
        )
    except Exception as e:
        logging.error(f"Error in dataset creation: {e}")
        raise

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_ds


def configure_model(train_ds):
    """Configure model architecture parameters."""
    first_sample = train_ds[0]

    transcriptomics_input_dim = first_sample["transcriptomics"].shape[0]
    molecular_input_dim = first_sample["molecular"].shape[0]  # Includes dosage

    model_kwargs = {
        "transcriptomics_input_dim": transcriptomics_input_dim,
        "molecular_input_dim": molecular_input_dim,
        "transcriptomics_hidden_dims": [512, 256],
        "molecular_hidden_dims": [256, 128],
        "transcriptomics_output_dim": 128,
        "molecular_output_dim": 128,
        "fusion_strategy": "concat",
        "predictor_hidden_dims": [64, 32],
        "dropout": 0.3,
    }
    return model_kwargs


def main():
    """Main entry point for the drug response prediction framework."""
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    device = configure_environment(args, config)

    paths = get_paths(config)
    output_dir = args.output_dir or paths["results_dir"]
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_logger = ExperimentLogger(
        experiment_name=f"multimodal_drug_response_{timestamp}",
        config=config,
        log_dir=str(paths["log_dir"]),
        use_tensorboard=True,
        use_wandb=not args.no_wandb,
        wandb_project=config["experiment"]["project_name"],
    )

    if not args.no_wandb:
        init_wandb(config)

    train_loader, val_loader, test_loader, train_ds = prepare_data(args, config)
    model_kwargs = configure_model(train_ds)

    optimizer_class = optim.Adam
    scheduler_class = optim.lr_scheduler.ReduceLROnPlateau
    criterion = nn.MSELoss()

    logging.info(f"Setting up multi-run training ({args.runs} runs)...")
    multi_trainer = MultiRunTrainer(
        model_class=MultimodalViabilityPredictor,
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
        device=device,
    )

    logging.info(f"Starting multi-run training for {args.epochs} epochs per run...")
    aggregate_results = multi_trainer.run_training(epochs=args.epochs)

    logging.info("Multi-run training completed. Aggregate results:")
    for phase in ["val", "test"]:
        logging.info(f"\n{phase.upper()} METRICS:")
        for metric, value in aggregate_results[phase].items():
            logging.info(f"  {metric}: {value:.4f}")

    logging.info("Running detailed evaluation...")
    evaluator = Evaluator(
        model=multi_trainer.trained_models[0],
        exp_logger=exp_logger,
        config=config,
        device=device,
    )

    cell_results = evaluator.evaluate_by_group(
        test_loader,
        group_column="cell_mfc_name",
        output_dir=os.path.join(output_dir, "evaluation"),
    )

    multi_model_results = evaluator.multi_run_evaluate(
        models=multi_trainer.trained_models,
        data_loader=test_loader,
        output_dir=os.path.join(output_dir, "evaluation"),
        prefix="final_",
    )

    logging.info("Evaluation completed. Results saved to output directory.")
    exp_logger.close()
    logging.info(f"Complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
