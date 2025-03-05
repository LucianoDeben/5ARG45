#!/usr/bin/env python
# src/main.py - Multimodal Drug Response Prediction Framework

import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.config.config_utils import get_paths, init_wandb, load_config, setup_logging
from src.data.data_preparation import prepare_datasets
from src.evaluation.evaluator import Evaluator
from src.models.model_configuration import configure_model
from src.training.trainer import MultiRunTrainer
from src.utils.logging import ExperimentLogger

def parse_args():
    """Parse command-line arguments, allowing overrides of config values."""
    parser = argparse.ArgumentParser(description="Multimodal Drug Response Prediction")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--runs", type=int, help="Number of training runs (overrides config)")
    parser.add_argument("--epochs", type=int, help="Number of epochs per run (overrides config)")
    parser.add_argument("--nrows", type=int, help="Number of data rows to use (overrides config)")
    parser.add_argument("--batch_size", type=int, help="Batch size (overrides config)")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config)")
    parser.add_argument("--lr", type=float, help="Learning rate (overrides config)")
    return parser.parse_args()

def configure_environment(config: dict, args) -> torch.device:
    """Configure computational environment based on config and optional args overrides."""
    seed = args.seed if args.seed is not None else config["data"]["random_seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device

def main():
    """Orchestrate the multimodal drug response prediction workflow."""
    # Parse arguments and setup logging
    args = parse_args()
    setup_logging()

    # Load config as the ground truth
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return

    # Apply command-line overrides where provided
    config["training"]["num_runs"] = args.runs if args.runs is not None else config["training"].get("num_runs", 5)
    config["training"]["epochs"] = args.epochs if args.epochs is not None else config["training"]["epochs"]
    config["data"]["nrows"] = args.nrows if args.nrows is not None else config["data"].get("nrows")
    config["training"]["batch_size"] = args.batch_size if args.batch_size is not None else config["training"]["batch_size"]
    config["data"]["random_seed"] = args.seed if args.seed is not None else config["data"]["random_seed"]
    config["paths"]["results_dir"] = args.output_dir if args.output_dir else config["paths"]["results_dir"]
    config["training"]["learning_rate"] = args.lr if args.lr is not None else config["training"]["learning_rate"]

    # Configure environment
    device = configure_environment(config, args)

    # Setup paths and output directory
    output_dir = Path(config["paths"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"multimodal_drug_response_{timestamp}"

    # Setup experiment logging
    try:
        exp_logger = ExperimentLogger(
            experiment_name=exp_name,
            config=config,
            log_dir=config["paths"]["log_dir"],
            use_tensorboard=True,
            use_wandb=not args.no_wandb,
            wandb_project=config["experiment"]["project_name"],
        )
        if not args.no_wandb:
            init_wandb(config)
    except Exception as e:
        logging.error(f"Error setting up experiment logging: {e}")
        return

    # Prepare datasets
    try:
        train_loader, val_loader, test_loader = prepare_datasets(config)
    except Exception as e:
        logging.error(f"Failed to prepare datasets: {e}")
        return

    # Configure model
    try:
        model = configure_model(train_loader.dataset, config)
    except Exception as e:
        logging.error(f"Failed to configure model: {e}")
        return

    # Setup training components
    optimizer_class = optim.Adam
    scheduler_class = optim.lr_scheduler.ReduceLROnPlateau
    criterion = nn.MSELoss()

    # Setup and run multi-run training
    logging.info(f"Setting up multi-run training ({config['training']['num_runs']} runs)...")
    multi_trainer = MultiRunTrainer(
        model_class=lambda **kwargs: model,  # Use pre-configured model instance
        model_kwargs={},  # No additional kwargs needed since model is pre-instantiated
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer_class=optimizer_class,
        scheduler_class=scheduler_class,
        criterion=criterion,
        exp_logger=exp_logger,
        config=config,
        num_runs=config["training"]["num_runs"],
        save_models=True,
        output_dir=output_dir,
        device=device,
    )

    logging.info(f"Starting multi-run training for {config['training']['epochs']} epochs per run...")
    try:
        aggregate_results = multi_trainer.run_training(epochs=config["training"]["epochs"])
    except Exception as e:
        logging.error(f"Error during training: {e}")
        exp_logger.close()
        return

    logging.info("Multi-run training completed. Aggregate results:")
    for phase in ["val", "test"]:
        logging.info(f"\n{phase.upper()} METRICS:")
        for metric, value in aggregate_results[phase].items():
            logging.info(f"  {metric}: {value:.4f}")

    # Run evaluation
    logging.info("Running detailed evaluation...")
    try:
        evaluator = Evaluator(
            model=multi_trainer.trained_models[0],  # Use the first trained model
            exp_logger=exp_logger,
            config=config,
            device=device,
        )

        # Basic evaluation
        basic_results = evaluator.evaluate(
            test_loader,
            output_dir=output_dir / "evaluation",
            prefix="basic_"
        )
        logging.info(f"Basic evaluation results: {basic_results}")

        # Cell-line specific evaluation
        cell_results = evaluator.evaluate_by_group(
            test_loader,
            group_column="cell_mfc_name",
            output_dir=output_dir / "evaluation",
            prefix="cellwise_"
        )
        logging.info(f"Cell-line evaluation complete. Analyzed {len(cell_results)} cell lines.")

        # Multi-model evaluation
        multi_model_results = evaluator.multi_run_evaluate(
            models=multi_trainer.trained_models,
            data_loader=test_loader,
            output_dir=output_dir / "evaluation",
            prefix="final_",
        )
        logging.info("Multi-model evaluation complete.")
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")

    exp_logger.close()
    logging.info(f"Complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()