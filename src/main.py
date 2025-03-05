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
from src.data.feature_transforms import create_feature_transform
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
        "--epochs", type=int, default=10, help="Number of epochs per run"
    )
    parser.add_argument(
        "--nrows", type=int, default=1000, help="Number of data rows to use"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
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
    try:
        gctx_file = config["data"]["gctx_file"]
        gctx_loader = GCTXDataLoader(gctx_file, preload_metadata=True)
    except KeyError as e:
        logging.error(f"Missing configuration key: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    logging.info("Creating datasets...")
    # Get feature transform parameters from config if available
    fp_size = config.get("model", {}).get("fingerprint_size", 1024)
    fp_radius = config.get("model", {}).get("fingerprint_radius", 2)
    
    transform_molecular = create_feature_transform(
        "fingerprint", fingerprint_size=fp_size, fingerprint_radius=fp_radius
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

    # Use args.batch_size consistently for all data loaders
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
        test_ds,  # FIXED: Now correctly using test_ds instead of val_ds
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_ds


def configure_model(train_ds, config: Dict[str, Any]):
    """Configure model architecture parameters."""
    first_sample = train_ds[0]

    transcriptomics_input_dim = first_sample["transcriptomics"].shape[0]
    molecular_input_dim = first_sample["molecular"].shape[0]  # Includes dosage

    # Get model parameters from config if available, otherwise use defaults
    model_config = config.get("model", {})
    
    model_kwargs = {
        "transcriptomics_input_dim": transcriptomics_input_dim,
        "molecular_input_dim": molecular_input_dim,
        "transcriptomics_hidden_dims": model_config.get("transcriptomics_hidden_dims", [512, 256]),
        "molecular_hidden_dims": model_config.get("molecular_hidden_dims", [256, 128]),
        "transcriptomics_output_dim": model_config.get("transcriptomics_output_dim", 128),
        "molecular_output_dim": model_config.get("molecular_output_dim", 128),
        "fusion_strategy": model_config.get("fusion_strategy", "concat"),
        "predictor_hidden_dims": model_config.get("predictor_hidden_dims", [64, 32]),
        "dropout": model_config.get("dropout", 0.3),
    }
    
    logging.info(f"Model configuration: {model_kwargs}")
    return model_kwargs


def main():
    """Main entry point for the drug response prediction framework."""
    args = parse_args()
    setup_logging()
    
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return
    
    device = configure_environment(args, config)

    try:
        paths = get_paths(config)
        output_dir = args.output_dir or paths["results_dir"]
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"Error setting up output directories: {e}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"multimodal_drug_response_{timestamp}"
    
    # Setup experiment logging
    try:
        exp_logger = ExperimentLogger(
            experiment_name=exp_name,
            config=config,
            log_dir=str(paths["log_dir"]),
            use_tensorboard=True,
            use_wandb=not args.no_wandb,
            wandb_project=config["experiment"].get("project_name", "drug_response_prediction"),
        )

        if not args.no_wandb:
            init_wandb(config)
    except Exception as e:
        logging.error(f"Error setting up experiment logging: {e}")
        return

    try:
        train_loader, val_loader, test_loader, train_ds = prepare_data(args, config)
    except Exception as e:
        logging.error(f"Failed to prepare data: {e}")
        return

    model_kwargs = configure_model(train_ds, config)

    # Update config with command-line arguments if provided
    if args.lr:
        # Make sure the training section exists
        if "training" not in config:
            config["training"] = {}
        config["training"]["learning_rate"] = args.lr
        logging.info(f"Using learning rate from command line: {args.lr}")

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
    try:
        aggregate_results = multi_trainer.run_training(epochs=args.epochs)
    except Exception as e:
        logging.error(f"Error during training: {e}")
        exp_logger.close()
        return

    logging.info("Multi-run training completed. Aggregate results:")
    for phase in ["val", "test"]:
        logging.info(f"\n{phase.upper()} METRICS:")
        for metric, value in aggregate_results[phase].items():
            logging.info(f"  {metric}: {value:.4f}")

    logging.info("Running detailed evaluation...")
    try:
        evaluator = Evaluator(
            model=multi_trainer.trained_models[0],
            exp_logger=exp_logger,
            config=config,
            device=device,
        )

        # First run a basic evaluation to ensure everything works
        basic_results = evaluator.evaluate(
            test_loader, 
            output_dir=os.path.join(output_dir, "evaluation"),
            prefix="basic_"
        )
        logging.info(f"Basic evaluation results: {', '.join([f'{k}: {v:.4f}' for k, v in basic_results.items()])}")

        # Then try to evaluate by cell line
        try:
            cell_results = evaluator.evaluate_by_group(
                test_loader,
                group_column="cell_mfc_name",
                output_dir=os.path.join(output_dir, "evaluation"),
                prefix="cellwise_"
            )
            logging.info(f"Cell-line evaluation complete. Analyzed {len(cell_results)} cell lines.")
        except Exception as cell_error:
            logging.error(f"Error during cell-line evaluation: {cell_error}")

        # Run multi-model evaluation
        multi_model_results = evaluator.multi_run_evaluate(
            models=multi_trainer.trained_models,
            data_loader=test_loader,
            output_dir=os.path.join(output_dir, "evaluation"),
            prefix="final_",
        )
        logging.info("Multi-model evaluation complete.")
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
    
    exp_logger.close()
    logging.info(f"Complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()