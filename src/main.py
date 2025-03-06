#!/usr/bin/env python
# src/main.py - Multimodal Drug Response Prediction Framework

import argparse
import logging
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.config.config_utils import get_paths, init_wandb, load_config, setup_logging
from src.data.data_preparation import prepare_datasets
from src.evaluation.evaluator import Evaluator
from src.models.model_configuration import configure_model
from src.training.trainer import MultiRunTrainer
from src.utils.experiment_tracker import ExperimentTracker


# def parse_args():
#     """Parse command-line arguments, allowing overrides of config values."""
#     parser = argparse.ArgumentParser(description="Multimodal Drug Response Prediction")
#     parser.add_argument(
#         "--config", type=str, default="config.yaml", help="Path to config file"
#     )
#     parser.add_argument(
#         "--runs", type=int, help="Number of training runs (overrides config)"
#     )
#     parser.add_argument(
#         "--epochs", type=int, help="Number of epochs per run (overrides config)"
#     )
#     parser.add_argument(
#         "--nrows", type=int, help="Number of data rows to use (overrides config)"
#     )
#     parser.add_argument("--batch_size", type=int, help="Batch size (overrides config)")
#     parser.add_argument("--seed", type=int, help="Random seed (overrides config)")
#     parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
#     parser.add_argument(
#         "--output_dir", type=str, help="Output directory (overrides config)"
#     )
#     parser.add_argument("--lr", type=float, help="Learning rate (overrides config)")
#     return parser.parse_args()


# def configure_environment(config: dict, args) -> torch.device:
#     """Configure computational environment based on config and optional args overrides."""
#     seed = args.seed if args.seed is not None else config["data"]["random_seed"]
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logging.info(f"Using device: {device}")
#     return device


def get_optimizer_class(optimizer_name: str):
    """Return the optimizer class based on config specification."""
    optimizers = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
    }
    if optimizer_name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return optimizers[optimizer_name]


def get_scheduler_class(scheduler_type: str, config: dict):
    """Return a scheduler factory function with config parameters."""
    schedulers = {
        "step": partial(
            optim.lr_scheduler.StepLR,
            step_size=config["training"]["lr_scheduler"]["step_size"],
            gamma=config["training"]["lr_scheduler"].get("gamma", 0.1),
        ),
        "cosine": partial(
            optim.lr_scheduler.CosineAnnealingLR,
            T_max=config["training"]["epochs"],  # Assuming total epochs as T_max
            eta_min=config["training"]["lr_scheduler"].get("min_lr", 0.0),
        ),
        "plateau": partial(
            optim.lr_scheduler.ReduceLROnPlateau,
            mode="min",
            factor=config["training"]["lr_scheduler"].get("factor", 0.1),
            patience=config["training"]["lr_scheduler"].get("patience", 10),
        ),
    }
    if scheduler_type not in schedulers:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    return schedulers[scheduler_type]


# def get_loss_function(loss_name: str):
#     """Return the loss function based on config specification."""
#     losses = {
#         "mse": nn.MSELoss(),
#         "mae": nn.L1Loss(),
#         "smooth_l1": nn.SmoothL1Loss(),
#     }
#     if loss_name not in losses:
#         raise ValueError(f"Unsupported loss function: {loss_name}")
#     return losses[loss_name]


def main():
    """Orchestrate the multimodal drug response prediction workflow."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting multimodal drug response prediction workflow...")
    
    # Load configuration
    try:
        config = load_config("config.yaml")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # Setup paths and output directory (uses config['paths'])
    output_dir = Path(config["paths"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"multimodal_drug_response_{timestamp}"

    # Setup experiment logging (uses config['experiment'] and config['paths'])
    try:
        exp_logger = ExperimentTracker(
            experiment_name=exp_name,
            config=config,
            log_dir=config["paths"]["log_dir"],
            use_tensorboard=False,
            use_wandb= False,
            wandb_project=config["experiment"]["project_name"],
        )
        if config["experiment"]["track"]:
            init_wandb(config)
    except Exception as e:
        logger.error(f"Error setting up experiment logging: {e}")
        return
    try:
        train_loader, val_loader, test_loader = prepare_datasets(config)
    except Exception as e:
        logger.error(f"Failed to prepare datasets: {e}")
        return

    try:
        model = configure_model(train_loader.dataset, config)
    except Exception as e:
        logger.error(f"Failed to configure model: {e}")
        return

    try:
        optimizer_class = optim.Adam
        
        scheduler_factory = get_scheduler_class(
            config["training"]["lr_scheduler"]["type"], config
        )
        criterion = nn.MSELoss()
    except Exception as e:
        logger.error(f"Error configuring training components: {e}")
        return
    
    # Set the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup and run multi-run training
    logger.info(
        f"Setting up multi-run training ({config['training']['num_runs']} runs)..."
    )
    multi_trainer = MultiRunTrainer(
        model_class=lambda **kwargs: model,
        model_kwargs={}, 
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer_class=optimizer_class,
        scheduler_class=scheduler_factory, 
        criterion=criterion,
        experiment_tracker=exp_logger,
        config=config,
        num_runs=config["training"]["num_runs"],
        save_models=True,
        output_dir=output_dir,
        device=device,
    )

    logger.info(
        f"Starting multi-run training for {config['training']['epochs']} epochs per run..."
    )
    try:
        aggregate_results = multi_trainer.run_training(
            epochs=config["training"]["epochs"]
        )
    except Exception as e:
        logger.error(f"Error during training: {e}")
        exp_logger.close()
        return

    logger.info("Multi-run training completed. Aggregate results:")
    for phase in ["val", "test"]:
        logger.info(f"\n{phase.upper()} METRICS:")
        for metric, value in aggregate_results[phase].items():
            logger.info(f"  {metric}: {value:.4f}")

    # Run evaluation (uses config['evaluation'])
    logger.info("Running detailed evaluation...")
    try:
        evaluator = Evaluator(
            model=multi_trainer.trained_models[0],  # Use the first trained model
            experiment_tracker=exp_logger,
            config=config,
            device=device,
        )

        # Basic evaluation
        basic_results = evaluator.evaluate(
            test_loader, output_dir=output_dir / "evaluation", prefix="basic_"
        )
        logger.info(f"Basic evaluation results: {basic_results}")

        # Cell-line specific evaluation
        cell_results = evaluator.evaluate_by_group(
            test_loader,
            group_column="cell_mfc_name",
            output_dir=output_dir / "evaluation",
            prefix="cellwise_",
        )
        logger.info(
            f"Cell-line evaluation complete. Analyzed {len(cell_results)} cell lines."
        )

        # Multi-model evaluation
        multi_model_results = evaluator.multi_run_evaluate(
            models=multi_trainer.trained_models,
            data_loader=test_loader,
            output_dir=output_dir / "evaluation",
            prefix="final_",
        )
        logger.info("Multi-model evaluation complete.")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")

    exp_logger.close()
    logger.info(f"Complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
