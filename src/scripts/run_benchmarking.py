# scripts/run_multirun_experiment.py
import os
import argparse
import yaml
import torch
import torch.nn as nn
import logging
from datetime import datetime
from src.config.config_utils import setup_logging, get_default_config
from src.data.loaders import GCTXDataLoader
from src.data.datasets import DatasetFactory
from src.training.multi_run_trainer import MultiRunTrainer
from src.models.transcriptomics_module import TranscriptomicsModule

# Import your FlexibleFCNN model
from src.models.unimodal.flexible_fcnn import FlexibleFCNN

def main(config_path=None):
    # Setup logging
    logger = setup_logging()
    
    # Load configuration if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = get_default_config()
    
    # Load LINCS and MixSeq data
    logger.info("Loading datasets...")
    lincs_loader = GCTXDataLoader(config["data"]["lincs_file"])
    mixseq_loader = GCTXDataLoader(config["data"]["mixseq_file"])
    
    # Create dataset for MixSeq as external test set
    logger.info("Creating MixSeq dataset for external testing...")
    _, _, test_ds_mixseq = DatasetFactory.create_and_split_transcriptomics(
        gctx_loader=mixseq_loader,
        feature_space=config["data"]["feature_space"],
        nrows=config["data"]["nrows"],
        test_size=0.9, # Use most of the data for testing, this is a workaround for MixSeq for now
        val_size=0.05,
        random_state=config["training"]["random_state"],
        chunk_size=config["data"]["chunk_size"],
        group_by=config["training"]["group_by"],
        stratify_by=config["training"]["stratify_by"],
    )
    
    # Create DataLoader for external test set
    mixseq_loader = torch.utils.data.DataLoader(
        test_ds_mixseq,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"]
    )
    
    # Get the input dimension from a sample or use from config
    input_dim = config["model"]["transcriptomics_input_dim"]
    logger.info(f"Using input dimension from config: {input_dim}")
    
    # Create model
    model = FlexibleFCNN(
        input_dim=input_dim,
        hidden_dims=config["model"]["transcriptomics_hidden_dims"] + [config["model"]["transcriptomics_output_dim"]],  # Combine hidden and output dims
        output_dim=1,
        activation_fn=config["model"]["activation"],
        dropout_prob=config["model"]["dropout"],
        residual=False,  # Set based on your needs
        norm_type="batchnorm" if config["model"]["use_batch_norm"] else "none",
        weight_init="kaiming",  # Default
    )
    
    # Create Lightning module
    module_kwargs = {
        "model": model,
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "grad_clip": config["training"]["max_grad_norm"] if config["training"]["clip_grad_norm"] else None,
    }
    
    # Create timestamp and run name for experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config["experiment"]["run_name"].replace("${timestamp}", timestamp)
    if "${RUN_NAME:-" in run_name:
        # Handle the variable substitution syntax
        run_name = f"run_{timestamp}"
    
    # Set up MultiRunTrainer
    trainer = MultiRunTrainer(
        model_class=TranscriptomicsModule,
        model_kwargs=module_kwargs,
        gctx_loader=lincs_loader,
        dataset_kwargs={
            "feature_space": config["data"]["feature_space"],
            "nrows": config["data"]["nrows"],
            "test_size": config["training"]["test_size"],
            "val_size": config["training"]["val_size"],
            "chunk_size": config["data"]["chunk_size"],
            "group_by": config["training"]["group_by"],
            "stratify_by": config["training"]["stratify_by"],
        },
        dataloader_kwargs={
            "batch_size": config["training"]["batch_size"],
            "num_workers": config["data"]["num_workers"],
            "shuffle": True,
        },
        num_runs=config["training"]["num_runs"],
        max_epochs=config["training"]["epochs"],
        patience=config["training"]["patience"],
        output_dir=config["paths"]["results_dir"],
        experiment_name=f"{config['experiment']['project_name']}_{run_name}",
        base_seed=config["training"]["random_state"],
        gpu_if_available=(config["inference"]["device"] == "cuda"),
        external_test_loaders={"mixseq": mixseq_loader},
    )
    
    # Run training
    logger.info("Starting multi-run training...")
    results = trainer.train()
    
    # Print final results
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETE")
    logger.info("="*50)
    logger.info("\nAggregated Test Metrics:")
    for metric, value in results["mean"].items():
        if metric.startswith("test_"):
            logger.info(f"{metric.replace('test_', '')}: {value:.4f} ± {results['std'][metric]:.4f}")
    
    logger.info("\nExternal Test Metrics (MixSeq):")
    for metric, value in results["mean"].items():
        if metric.startswith("mixseq_"):
            logger.info(f"{metric.replace('mixseq_', '')}: {value:.4f} ± {results['std'][metric]:.4f}")
    
    logger.info(f"\nResults saved to {trainer.experiment_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-run training experiment")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)